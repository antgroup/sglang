"""Startup JIT manifest build rules (srt/utils/jit_manifest.py); no GPU, no compiler."""

import hashlib
import json
import shlex
import sys
import tempfile
import threading
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.moe_runner import humming_overlay
from sglang.srt.utils import jit_manifest
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _humming(key, patch_mode, t=0.0):
    return {
        "t": t,
        "kind": "humming",
        "compiler": "NVRTCCompiler",
        "code": "#include <humming/kernel/humming.cuh>\nkernel<1>",
        "sm": "90a",
        "expr": f"kernel<{key}>",
        "no_fast_math": False,
        "key": key,
        "patch": patch_mode,
    }


def _deep_gemm_entry(root: Path, name: str, staging: str) -> None:
    entry = root / "cache" / name
    entry.mkdir(parents=True)
    meta = {
        "command": f"/usr/local/cuda/bin/nvcc {staging}/kernel.cu --cubin "
        f"--output-file {staging}/kernel.cubin --gpu-architecture=sm_90a -O3",
        "compiler_info": {"path": "/usr/local/cuda/bin/nvcc", "version": "v"},
        "config": {"include_dirs": []},
    }
    (entry / "meta.json").write_text(json.dumps(meta))
    (entry / "kernel.cu").write_text("// kernel")
    (entry / "kernel.cubin").write_bytes(b"\0")
    (entry / ".committed").touch()


class TestHummingPatchMode(CustomTestCase):
    def test_entry_kept_only_with_one_patch_mode(self):
        """A humming cubin may be byte-patched after nvrtc (postprocess_cubin); replaying
        with the wrong patch publishes a wrong cubin under the right cache key, which
        every rank then loads. An entry seen with two patch modes is dropped."""
        cases = {
            "unpatched": ([_humming("b", None), _humming("b", None)], (None,)),
            "patched": ([_humming("c", "mma_e3m4_b")], ("mma_e3m4_b",)),
            "conflicting": ([_humming("d", None), _humming("d", "mma_e0m3_a")], ()),
        }
        for name, (rows, expected) in cases.items():
            with self.subTest(case=name):
                merged = jit_manifest._merge_humming(rows)
                self.assertEqual(tuple(e["patch"] for _, e in merged), expected)


class TestTritonReplay(CustomTestCase):
    def test_libdevice_follows_the_imported_triton_package(self):
        """Replay must not pass another installation's libdevice to the compiler."""
        old = "/old/site-packages/triton/backends/nvidia/lib/libdevice.10.bc"
        missing = "/old/triton/backends/nvidia/lib/missing.bc"
        custom = "/custom/libdevice.10.bc"
        options = {
            "extern_libs": [
                ["libdevice", old],
                ["missing", missing],
                ["custom", custom],
            ]
        }
        spec = {
            "signature": {},
            "constant_keys": [],
            "constant_vals": [],
            "attrs_keys": [],
            "attrs_vals": [],
            "options": options,
            "target": {"backend": "cuda", "arch": 90, "warp_size": 32},
        }
        entry = {"fn": "replay_kernel:k", "spec": {}}
        tables = {}
        for key, value in spec.items():
            if key in jit_manifest._TRITON_TABLE_FIELDS:
                entry["spec"][key] = 0
                tables[f"triton_{key}"] = [value]
            else:
                entry["spec"][key] = value
        with tempfile.TemporaryDirectory() as root:
            package = Path(root) / "triton"
            local = package / "backends/nvidia/lib/libdevice.10.bc"
            local.parent.mkdir(parents=True)
            local.touch()
            modules = {
                name: types.ModuleType(name)
                for name in (
                    "triton",
                    "triton.language",
                    "triton.backends",
                    "triton.backends.compiler",
                    "triton.compiler",
                    "triton.runtime",
                    "triton.runtime.jit",
                    "replay_kernel",
                )
            }
            triton = modules["triton"]
            triton.__file__ = str(package / "__init__.py")
            triton.compile = Mock()
            triton.language = modules["triton.language"]
            modules["triton.backends.compiler"].GPUTarget = lambda **kw: kw
            modules["triton.compiler"].ASTSource = lambda *args: args
            jit = modules["triton.runtime.jit"]
            jit._triton_jit_function_registry = {
                "replay_kernel:k": types.SimpleNamespace(is_gluon=lambda: False)
            }
            jit.convert_to_tuple_if_list = tuple
            modules["triton.runtime"].jit = jit
            with patch.dict(sys.modules, modules):
                self.assertEqual(jit_manifest._compile_triton(entry, tables), "ok")
                actual = triton.compile.call_args.kwargs["options"]["extern_libs"]
                self.assertEqual(
                    actual,
                    (
                        ["libdevice", str(local)],
                        ["missing", missing],
                        ["custom", custom],
                    ),
                )
                local.unlink()
                self.assertEqual(jit_manifest._compile_triton(entry, tables), "ok")
                self.assertEqual(
                    triton.compile.call_args.kwargs["options"]["extern_libs"],
                    tuple(options["extern_libs"]),
                )
        self.assertEqual(options["extern_libs"][0][1], old)


class TestDeepGemmEntry(CustomTestCase):
    def test_recorded_command_is_rebased_on_the_replay_staging_dir(self):
        """The recorded nvcc command names the recording rank's staging dir for both the
        source and the cubin; replay must point both at its own staging dir."""
        with tempfile.TemporaryDirectory() as root:
            old = "/cache/deep_gemm/tmp/1569-69b521d9-b8fad2a4-db0220eb"
            _deep_gemm_entry(Path(root), "sm90_fp8_gemm_1d2d.0123abcd", old)
            [(_, scanned)] = jit_manifest._scan_deep_gemm(Path(root))
            command = scanned["meta"]["command"].replace(
                jit_manifest._DG_STAGING_TOKEN, "/new/staging"
            )
            argv = shlex.split(command)
            self.assertEqual(argv[1], "/new/staging/kernel.cu")
            self.assertEqual(
                argv[argv.index("--output-file") + 1], "/new/staging/kernel.cubin"
            )
            self.assertNotIn(old, command)


class TestManifestTables(CustomTestCase):
    def test_build_then_merge_round_trips(self):
        """merge expands the interned tables (Triton spec fields, DeepGEMM meta, humming
        source lines) and re-interns them; a lossy step there would publish entries
        under cache keys no rank looks up."""
        spec = {
            "name": "k",
            "key": "x",
            "signature": {"x": "*fp32"},
            "constant_keys": [["BLOCK"]],
            "constant_vals": [128],
            "attrs_keys": [],
            "attrs_vals": [],
            "options": {"num_warps": 4},
            "target": {"backend": "cuda", "arch": 90, "warp_size": 32},
        }
        rows = [
            {"t": 1.0, "kind": "triton", "fn": "m:k", "spec": json.dumps(spec)},
            {"t": 2.0, "kind": "sglang_jit", "args": ["a"], "kwargs": {"x": [1]}},
            {"t": 3.0, "kind": "tilelang", "fn": "m:f", "args": [1], "kwargs": {}},
            _humming("h", None, t=4.0),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            records, dg = Path(tmp) / "records", Path(tmp) / "dg"
            records.mkdir()
            (records / "1.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in rows)
            )
            _deep_gemm_entry(dg, "sm90_gemm.ab", "/s/tmp/1-2")
            built = jit_manifest.build_manifest(records=records, deep_gemm_cache=dg)
            out = Path(tmp) / "m.json"
            jit_manifest.write_manifest(built, out)
            manifest = json.loads(out.read_text())
        self.assertEqual(
            sorted(e["kind"] for e in manifest["entries"]),
            ["deep_gemm", "humming", "sglang_jit", "tilelang", "triton"],
        )
        self.assertEqual(jit_manifest.merge_manifests([manifest, manifest]), manifest)


class TestRecorder(CustomTestCase):
    def test_cli_cannot_create_recordings_when_startup_is_off(self):
        """The standalone record command used to bypass the engine's startup gate."""
        for enabled, disabled in ((False, ""), (True, "startup")):
            with (
                self.subTest(enabled=enabled, disabled=disabled),
                tempfile.TemporaryDirectory() as cache,
                envs.SGLANG_CACHE_DIR.override(cache),
                envs.SGLANG_OPT_DSV41_SM90.override(enabled),
                envs.SGLANG_OPT_DSV41_SM90_DISABLE.override(disabled),
            ):
                with self.assertRaises(SystemExit):
                    jit_manifest.main(["record"])
                self.assertFalse(jit_manifest.record_dir().exists())

    def test_records_only_while_the_record_dir_exists(self):
        # An engine without the record dir must not install hooks or write records.
        with (
            tempfile.TemporaryDirectory() as cache,
            envs.SGLANG_CACHE_DIR.override(cache),
            envs.SGLANG_OPT_DSV41_SM90.override(True),
            envs.SGLANG_OPT_DSV41_SM90_DISABLE.override(""),
            patch.object(jit_manifest, "_record_path", None),
            patch.object(jit_manifest, "_record_lock", None),
            patch.object(jit_manifest, "_tilelang_seen", None),
            patch.object(jit_manifest, "_install_triton_recorder") as triton,
            patch.object(jit_manifest, "_install_humming_recorder"),
        ):
            jit_manifest.maybe_install_recorder()
            self.assertFalse(jit_manifest.is_recording())
            triton.assert_not_called()
            jit_manifest.main(["record"])
            jit_manifest.maybe_install_recorder()
            self.assertTrue(jit_manifest.is_recording())
            self.assertEqual(
                jit_manifest._record_path.parent, jit_manifest.record_dir()
            )
            triton.assert_called_once_with()


@contextmanager
def _fake_humming():
    """humming.jit.compiler with the real cache-key inputs: a cubin's key covers the
    include dirs and the header stamp the overlay folds its digest into."""

    builds = []

    class Compiler:
        include_dirs = staticmethod(lambda: ["/humming/include"])
        cuh_last_update_time = staticmethod(lambda: "stock-mtimes")

        @classmethod
        def compile(
            cls,
            code,
            sm_version,
            kernel_expr,
            disable_fast_math=False,
            postprocess_cubin=None,
        ):
            key_inputs = (
                code,
                tuple(Compiler.include_dirs()),
                Compiler.cuh_last_update_time(),
            )
            key = hashlib.md5(repr(key_inputs).encode()).hexdigest()
            builds.append(key)
            return f"/cache/humming/{key}/kernel.cubin"

    class NVRTCCompiler(Compiler):
        pass

    compiler = types.ModuleType("humming.jit.compiler")
    compiler.Compiler, compiler.NVRTCCompiler = Compiler, NVRTCCompiler
    cubin = types.ModuleType("humming.utils.cubin")
    cubin.patch_cubin = lambda cubin_path, mode, **kwargs: None
    modules = {
        "humming": types.ModuleType("humming"),
        "humming.jit": types.ModuleType("humming.jit"),
        "humming.jit.compiler": compiler,
        "humming.utils": types.ModuleType("humming.utils"),
        "humming.utils.cubin": cubin,
    }
    modules["humming.jit"].compiler = compiler
    modules["humming.utils"].cubin = cubin
    with (
        patch.dict(sys.modules, modules),
        patch.object(jit_manifest, "_humming_base", None),
    ):
        yield Compiler, NVRTCCompiler, builds


class TestHummingOverlay(CustomTestCase):
    def test_replay_rebuilds_each_kernel_under_its_recorded_overlay(self):
        """The overlay's headers and digest are part of a humming cubin's cache key; replay
        must rebuild an overlay kernel under that overlay, a stock kernel under the stock
        headers (also after an overlay entry), and skip an overlay this tree lacks or whose
        headers changed, which no rank would look up."""
        with (
            tempfile.TemporaryDirectory() as tmp,
            _fake_humming() as (compiler, nvrtc, builds),
            patch.object(humming_overlay, "OVERLAY_ROOT", tmp),
            patch.object(jit_manifest, "_record_path", Path(tmp) / "rec.jsonl"),
            patch.object(jit_manifest, "_record_lock", threading.Lock()),
        ):
            (Path(tmp) / "v_bm32" / "kernel").mkdir(parents=True)
            header = Path(tmp) / "v_bm32" / "kernel" / "humming.cuh"
            header.write_text("// overlay")
            stock = dict(compiler.__dict__)
            jit_manifest._install_humming_recorder()
            nvrtc.compile("stock", "90a", "k<0>")
            humming_overlay.install_include_overlay(compiler, str(Path(tmp) / "v_bm32"))
            with patch.object(humming_overlay, "_installed", "v_bm32"):
                nvrtc.compile("bm32", "90a", "k<1>")
            for name in ("include_dirs", "cuh_last_update_time"):
                setattr(compiler, name, stock[name])
            recorded = list(builds)

            rows = [
                json.loads(line)
                for line in (Path(tmp) / "rec.jsonl").read_text().splitlines()
            ]
            self.assertEqual([r["overlay"] for r in rows], [None, "v_bm32"])
            manifest = jit_manifest._intern(
                [e for _, e in jit_manifest._merge_humming(rows)]
            )
            tables = jit_manifest._load_tables(manifest)
            entries = {e["expr"]: e for e in manifest["entries"]}
            builds.clear()
            for expr in ("k<1>", "k<0>"):
                self.assertEqual(
                    jit_manifest._compile_humming(entries[expr], tables), "ok"
                )
            self.assertEqual(builds, recorded[::-1])

            header.write_text("// edited")
            jit_manifest._digest.cache_clear()
            self.assertEqual(
                jit_manifest._compile_humming(entries["k<1>"], tables), "overlay"
            )
            missing = {**entries["k<1>"], "overlay": "v_gone"}
            self.assertEqual(jit_manifest._compile_humming(missing, tables), "overlay")
            self.assertEqual(len(builds), 2)


if __name__ == "__main__":
    unittest.main()
