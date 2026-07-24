#!/usr/bin/env bash
set -euo pipefail

SGLANG_DIR="${SGLANG_DIR:-/home/admin/sglang}"
MODEL_PATH="${LINGBOT_MODEL_PATH:-/home/admin/lingbot-world-fast-diffusers}"
NVSHMEM_HOME="${FAST_ULYSSES_NVSHMEM_HOME:-/home/admin/nvshmem-3.7.2-cu12}"
LOG_DIR="${LOG_DIR:-/home/admin/logs}"

test -d "${SGLANG_DIR}"
test -d "${MODEL_PATH}"
test -d "${NVSHMEM_HOME}/lib"
mkdir -p "${LOG_DIR}"

export PYTHONPATH="${SGLANG_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}"
export LD_PRELOAD="/usr/lib64/libcuda.so.1${LD_PRELOAD:+:${LD_PRELOAD}}"
export LD_LIBRARY_PATH="${NVSHMEM_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export NVSHMEM_DISABLE_NVLS="${NVSHMEM_DISABLE_NVLS:-1}"
export NVSHMEM_REMOTE_TRANSPORT="${NVSHMEM_REMOTE_TRANSPORT:-none}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export SGLANG_FAST_ULYSSES_TAG_POOL_SIZE="${SGLANG_FAST_ULYSSES_TAG_POOL_SIZE:-32}"

unset SGLANG_ENABLE_ULYSSES_P2P_A2A
unset LINGBOT_FORCE_P2P
unset LINGBOT_ULYSSES_JIT

export SGLANG_LINGBOT_ENABLE_RIFE=1
export SGLANG_LINGBOT_RIFE_MODEL_PATH="${SGLANG_LINGBOT_RIFE_MODEL_PATH:-/home/admin/RIFE-4.22.lite}"
export SGLANG_LINGBOT_RIFE_EXP=1
export SGLANG_LINGBOT_RIFE_SCALE=1.0
export SGLANG_LINGBOT_TRITON_QKNORM_ACROSS_HEADS=1
export SGLANG_REALESRGAN_TORCH_COMPILE=1
export SGLANG_LINGBOT_PINNED_D2H=1
export SGLANG_LINGBOT_SEND_MEMORYVIEW=1
export SGLANG_LINGBOT_STARTUP_WARMUP_CHUNKS=10
export SGLANG_LOG_FULL_PROMPT=1
export SGLANG_LINGBOT_STARTUP_WARMUP_SIZES=1664x960,960x1664,1248x720,720x1248

cd "${SGLANG_DIR}"
exec sglang serve \
  --model-path "${MODEL_PATH}" \
  --host 127.0.0.1 \
  --port 8001 \
  --strict-ports true \
  --num-gpus 8 \
  --dit-cpu-offload false \
  --text-encoder-cpu-offload false \
  --image-encoder-cpu-offload false \
  --vae-cpu-offload false \
  --pin-cpu-memory false \
  --dit-layerwise-offload false \
  --realesrgan-half-precision \
  --ulysses-a2a-backend fast_ulysses \
  --ulysses-a2a-transfer auto \
  --ssl-certfile /home/admin/genie3_deploy/server.crt \
  --ssl-keyfile /home/admin/genie3_deploy/server.key \
  >>"${LOG_DIR}/sglang_diffusion.log" 2>&1
