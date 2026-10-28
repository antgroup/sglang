"""Header overlay for humming's JIT GEMM kernels on the DeepSeek-V4.1 SM90 path.

humming JIT-compiles its kernels from the CUDA headers in ``humming/include``. The directory
``humming_overlay_headers/<name>`` holds modified copies of a few of those headers for one
humming-kernels release; installing an overlay puts it in front of humming's include path, so
``#include <humming/...>`` resolves to the copy when one exists, and adds the overlay's content
hash to humming's JIT cache key, so overlay and stock kernels never share a cache entry.

Every overlay keeps the GEMM math: outputs are bitwise identical to stock 0.1.12.
  * v0_1_12_bm32 (every role but prefill):
    - datatype/dequant_fused.cuh: MXFP4 -> FP8 fused-e8m0 dequant with fewer ALU-pipe
      instructions (single selector mask, raw prmt, IDP4A scale extraction; as in 0.1.15).
    - kernel/humming.cuh: for block_m <= 32, main loop with compile-time stage ids and the
      next-stage wait one warp iteration later (as in 0.1.15).
    - mma/wgmma.cuh + kernel/humming.cuh: an indexed-MoE m-block of block_m 16 / 32 issues the
      narrowest of m64n8 / n16 / n24 / n32 k32 covering its valid rows, so 17..32-row experts
      take one m-block (one weight pass) without padding the tensor work.
  * v0_1_12_ntail (prefill) is bm32 plus:
    - the same per-block narrow wgmma for block_m 48 / 64 tiles with one warpgroup along M and
      one wgmma per k32 slab (warp N 16): a prefill expert's last m-block issues m64n{8g} for
      its g valid 8-row groups (about a tenth of the routed prefill MoE tensor work is padding);
    - K-tail skip: when K is padded past the last k32 slab (TP8 w2: K 288 -> 320), that
      all-padding slab's wgmma is not issued (its activations are zero-filled, its weights zero).
    It reads the bm32 overlay's tile tables after its own (TABLE_ALIASES).
"""

from __future__ import annotations

import glob
import hashlib
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

OVERLAY_ROOT = os.path.join(os.path.dirname(__file__), "humming_overlay_headers")
# humming-kernels version -> overlay directory per role (copies of that release's headers).
OVERLAY_DIRS = {"0.1.12": {"prefill": "v0_1_12_ntail", "default": "v0_1_12_bm32"}}
# Overlays whose kernels differ from another overlay's only in speed read that overlay's tile
# tables after their own.
TABLE_ALIASES = {"v0_1_12_ntail": "v0_1_12_bm32"}

_installed: Optional[str] = None
_install_attempted = False


def installed_overlay_tables() -> tuple[str, ...]:
    """Overlay names in the tile-table file names for the installed overlay, its own first."""
    if _installed is None:
        return ()
    alias = TABLE_ALIASES.get(_installed)
    return (_installed,) if alias is None else (_installed, alias)


def overlay_dir_for(humming_version: str, role: str) -> Optional[str]:
    dirs = OVERLAY_DIRS.get(humming_version)
    if dirs is None:
        return None
    return os.path.join(OVERLAY_ROOT, dirs.get(role, dirs["default"]))


def overlay_digest(path: str) -> str:
    md5 = hashlib.md5()
    for filename in sorted(
        glob.glob(os.path.join(path, "**", "*.cuh"), recursive=True)
    ):
        md5.update(os.path.relpath(filename, path).encode())
        with open(filename, "rb") as f:
            md5.update(f.read())
    return md5.hexdigest()


def install_include_overlay(compiler_cls, path: str) -> None:
    """Prepend ``path`` to ``compiler_cls``'s include dirs and key its JIT cache on the overlay."""
    digest = overlay_digest(path)
    base_include_dirs = compiler_cls.include_dirs
    base_header_stamp = compiler_cls.cuh_last_update_time
    compiler_cls.include_dirs = staticmethod(lambda: [path, *base_include_dirs()])
    compiler_cls.cuh_last_update_time = staticmethod(
        lambda: f"{base_header_stamp()}$$sglang-overlay:{digest}"
    )


def install_sm90_overlay(*, role: str) -> None:
    """Install the role's overlay once per process.

    Must run before the first humming kernel of the process is built (the MoE weight
    preparation calls it first); kernels built earlier keep the stock headers."""
    global _installed, _install_attempted
    if _install_attempted:
        return
    _install_attempted = True

    import humming
    from humming.jit.compiler import Compiler

    path = overlay_dir_for(humming.__version__, role)
    if path is None:
        logger.warning(
            f"No humming SM90 header overlay for humming-kernels {humming.__version__} "
            f"(have {sorted(OVERLAY_DIRS)}); stock headers and heuristic tiles"
        )
        return
    install_include_overlay(Compiler, path)
    _installed = os.path.basename(path)
    logger.info(f"humming header overlay installed: {path}")
