from typing import Optional

from sglang.srt.layers.dsv41_sm90 import dsv41_sm90_on

# Arbitrary within the range below. Measured on 8x H20-3e (TP8 + prefill CP8): a
# 4K/1K request decodes ~240 rounds, and at 64 running requests 4 slots free up
# every ~8-15 rounds.
_MAX_FIT_HOLD_PASSES = 32


def resolve_min_free_slots(
    user_value: Optional[int],
    max_running_requests: int,
    is_dflash_family: bool = False,
) -> Optional[int]:
    """Resolve the min-free-slots threshold (None = disabled).

    An explicit user value always wins, capped by max_running_requests
    (<= 1 disables). When unset, DFlash workloads fall back to the legacy
    formula (preserving the always-on behavior, disabled when
    max_running_requests < 8); other workloads stay disabled.
    """
    max_running_requests = max(0, int(max_running_requests))
    if user_value is not None:
        threshold = min(user_value, max_running_requests)
        return threshold if threshold > 1 else None
    if is_dflash_family and max_running_requests >= 8:
        return min(4, max(2, (max_running_requests + 5) // 6))
    return None


class MinFreeSlotsDelayer:
    """Delay fresh prefill admissions until at least ``min_free_slots`` running-
    request slots free up, batching them into one admission instead of one at a
    time. Useful when each admission is expensive (e.g. DFlash's draft prefill).

    Per-rank local: running-batch slots are private to each DP rank, so a rank
    with free slots does not wait for a congested peer.
    """

    def __init__(self, min_free_slots: int):
        self._min_free_slots = min_free_slots
        # DeepSeek-V4.1 SM90 runtime: a hold whose waiting requests all fit the free
        # slots ends after _MAX_FIT_HOLD_PASSES rounds. It grows only if requests
        # arrive meanwhile: on a saturated server they follow the finishing ones
        # (4 slots free up in ~8-15 rounds at 64 running requests), while the last 3
        # of a burst (3 free, threshold 4) waited a whole decode round (~240 rounds)
        # for them.
        self._bound_fit_holds = dsv41_sm90_on("runtime")
        self._fit_hold_passes = 0

    def should_delay(
        self,
        *,
        running_bs: int,
        num_allocatable_reqs: int,
        num_waiting_reqs: Optional[int] = None,
    ) -> bool:
        delay = running_bs > 0 and num_allocatable_reqs < self._min_free_slots
        if not self._bound_fit_holds:
            return delay
        assert num_waiting_reqs is not None
        if not delay or num_waiting_reqs > num_allocatable_reqs:
            self._fit_hold_passes = 0
            return delay
        # Rounds are counted, not timed: every TP rank's scheduler must decide alike.
        self._fit_hold_passes += 1
        if self._fit_hold_passes > _MAX_FIT_HOLD_PASSES:
            self._fit_hold_passes = 0
            return False
        return True
