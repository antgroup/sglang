#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
export PYTHONPATH="${repo_root}/python${PYTHONPATH:+:${PYTHONPATH}}"

python -m compileall -q \
  python/sglang/multimodal_gen/runtime/distributed/device_communicators/ulysses_a2a \
  python/sglang/multimodal_gen/runtime/distributed/device_communicators/ulysses_p2p_a2a.py \
  python/sglang/multimodal_gen/runtime/distributed/group_coordinator.py \
  python/sglang/multimodal_gen/runtime/distributed/parallel_groups.py \
  python/sglang/multimodal_gen/runtime/distributed/parallel_state.py \
  python/sglang/multimodal_gen/runtime/layers/usp.py \
  python/sglang/multimodal_gen/runtime/layers/attention/layer.py \
  python/sglang/multimodal_gen/runtime/models/dits/lingbot_world.py \
  python/sglang/multimodal_gen/runtime/managers/gpu_worker.py \
  python/sglang/multimodal_gen/runtime/server_args.py \
  test/registered/unit/test_diffusion_ulysses_a2a.py \
  scripts/ulysses_a2a \
  python/sglang/srt/environ.py \
  python/sglang/srt/distributed/device_communicators/cuda_wrapper.py \
  python/sglang/srt/distributed/device_communicators/custom_all_reduce.py

ruff check \
  python/sglang/multimodal_gen/runtime/distributed/device_communicators/ulysses_a2a \
  python/sglang/multimodal_gen/runtime/distributed/device_communicators/ulysses_p2p_a2a.py \
  python/sglang/multimodal_gen/runtime/distributed/group_coordinator.py \
  python/sglang/multimodal_gen/runtime/distributed/parallel_groups.py \
  python/sglang/multimodal_gen/runtime/distributed/parallel_state.py \
  python/sglang/multimodal_gen/runtime/layers/usp.py \
  python/sglang/multimodal_gen/runtime/layers/attention/layer.py \
  python/sglang/multimodal_gen/runtime/models/dits/lingbot_world.py \
  python/sglang/multimodal_gen/runtime/managers/gpu_worker.py \
  python/sglang/multimodal_gen/runtime/server_args.py \
  test/registered/unit/test_diffusion_ulysses_a2a.py \
  python/sglang/srt/environ.py \
  python/sglang/srt/distributed/device_communicators/cuda_wrapper.py \
  python/sglang/srt/distributed/device_communicators/custom_all_reduce.py \
  scripts/ulysses_a2a

python scripts/ulysses_a2a/static_smoke.py
python test/registered/unit/test_diffusion_ulysses_a2a.py
