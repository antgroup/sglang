"""DeepSeek-V4.1 SM90 optimizations: one switch and the subsystems it turns on.

``SGLANG_OPT_DSV41_SM90`` turns on the H20-tuned DeepSeek-V4.1 paths. While it is
off, or for another model or another architecture, the set is empty and every
caller keeps the upstream path. ``SGLANG_OPT_DSV41_SM90_DISABLE`` drops subsystems
by name (triage, ablations); ``SGLANG_OPT_DSV41_SM90_LOSSY`` opts into the paths
whose outputs are not bitwise to the ones they replace. Callers ask
``dsv41_sm90_on(<subsystem>)`` once at construction and keep the answer.
"""

import logging

import msgspec

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_disagg, get_flags
from sglang.srt.utils.common import is_sm90_supported

logger = logging.getLogger(__name__)

# One name per ported subsystem, as the DISABLE list spells them.
SUBSYSTEMS = (
    "startup",
    "moe",
    "fp8_gemm",
    "indexer",
    "cp",
    "mhc",
    "attn_proj",
    "prefill_graph",
    "mix_decode",
    "decode",
    "lm_head",
    "runtime",
    "pd",
    "epd",
)

# Not bitwise to the path they replace; on only when the LOSSY list names them.
LOSSY_OPTIONS = ("fp8_lm_head_target",)

ROLES = ("mix", "prefill", "decode", "encoder")

_MODEL_TYPE = "deepseek_v41"


class Dsv41Sm90Selection(msgspec.Struct, frozen=True):
    subsystems: frozenset[str] = frozenset()
    lossy: frozenset[str] = frozenset()
    role: str = ""
    # Why the set is empty although the switch is on; "" otherwise.
    off_reason: str = ""


def _parse_names(*, names, allowed, env_name: str) -> frozenset[str]:
    unknown = sorted(set(names) - set(allowed))
    if unknown:
        raise ValueError(
            f"{env_name}: unknown name(s) {unknown}; expected a subset of {list(allowed)}"
        )
    return frozenset(names)


def _role(*, disaggregation_mode: str, encoder_only: bool) -> str:
    if encoder_only:
        return "encoder"
    if disaggregation_mode in ("prefill", "decode"):
        return disaggregation_mode
    return "mix"


def resolve_dsv41_sm90(
    *,
    enabled: bool,
    model_type: str,
    sm90: bool,
    disable: tuple,
    lossy: tuple,
    disaggregation_mode: str,
    encoder_only: bool,
) -> Dsv41Sm90Selection:
    if not enabled:
        return Dsv41Sm90Selection()
    disabled = _parse_names(
        names=disable, allowed=SUBSYSTEMS, env_name="SGLANG_OPT_DSV41_SM90_DISABLE"
    )
    lossy_on = _parse_names(
        names=lossy, allowed=LOSSY_OPTIONS, env_name="SGLANG_OPT_DSV41_SM90_LOSSY"
    )
    if model_type != _MODEL_TYPE:
        return Dsv41Sm90Selection(off_reason=f"model_type {model_type!r}")
    if not sm90:
        return Dsv41Sm90Selection(off_reason="not an SM90 device")
    return Dsv41Sm90Selection(
        subsystems=frozenset(SUBSYSTEMS) - disabled,
        lossy=lossy_on,
        role=_role(disaggregation_mode=disaggregation_mode, encoder_only=encoder_only),
    )


def resolve_dsv41_sm90_from_env(
    *, model_config, disaggregation_mode: str, encoder_only: bool
) -> Dsv41Sm90Selection:
    """The selection the switch envs give this model and device. Server-arg
    resolution asks it before any worker has materialized the flags."""
    enabled = envs.SGLANG_OPT_DSV41_SM90.get()
    if not enabled:
        return Dsv41Sm90Selection()
    return resolve_dsv41_sm90(
        enabled=enabled,
        model_type=model_config.hf_config.model_type,
        sm90=is_sm90_supported(),
        disable=envs.SGLANG_OPT_DSV41_SM90_DISABLE.get(),
        lossy=envs.SGLANG_OPT_DSV41_SM90_LOSSY.get(),
        disaggregation_mode=disaggregation_mode,
        encoder_only=encoder_only,
    )


def initialize_dsv41_sm90(*, model_config) -> None:
    """Materialize ``flags.dsv41_sm90``; runs once per worker at layer-runtime init."""
    disagg = get_disagg()
    selection = resolve_dsv41_sm90_from_env(
        model_config=model_config,
        disaggregation_mode=disagg.disaggregation_mode,
        encoder_only=disagg.encoder_only,
    )
    flags = get_flags().dsv41_sm90
    flags.subsystems = selection.subsystems
    flags.lossy = selection.lossy
    flags.role = selection.role
    if selection.off_reason:
        logger.warning(
            "SGLANG_OPT_DSV41_SM90 is set but stays off: %s", selection.off_reason
        )
    elif selection.subsystems:
        logger.info(
            "DeepSeek-V4.1 SM90 optimizations on: role=%s, disabled=%s, lossy=%s",
            selection.role,
            sorted(set(SUBSYSTEMS) - selection.subsystems),
            sorted(selection.lossy),
        )


def dsv41_sm90_on(subsystem: str) -> bool:
    assert subsystem in SUBSYSTEMS, subsystem
    return subsystem in get_flags().dsv41_sm90.subsystems


def dsv41_sm90_lossy(option: str) -> bool:
    assert option in LOSSY_OPTIONS, option
    return option in get_flags().dsv41_sm90.lossy


def dsv41_sm90_role() -> str:
    return get_flags().dsv41_sm90.role
