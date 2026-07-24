#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 <nccl|fast_ulysses|native> [client_duration_seconds] [repetitions]" >&2
  exit 2
fi

BACKEND="$1"
CLIENT_DURATION="${2:-120}"
REPETITIONS="${3:-1}"
if [[ "${BACKEND}" != "nccl" && "${BACKEND}" != "fast_ulysses" && "${BACKEND}" != "native" ]]; then
  echo "backend must be nccl, fast_ulysses, or native" >&2
  exit 2
fi
if ! [[ "${REPETITIONS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "repetitions must be a positive integer" >&2
  exit 2
fi

SGLANG_DIR="${SGLANG_DIR:-/home/admin/sglang}"
MODEL_PATH="${LINGBOT_MODEL_PATH:-/home/admin/lingbot-world-fast-diffusers}"
CLIENT_PATH="${LINGBOT_CLIENT_PATH:-/ossfs/workspace/test_fusion_ling.py}"
NVSHMEM_HOME="${FAST_ULYSSES_NVSHMEM_HOME:-/home/admin/nvshmem-3.7.2-cu12}"
LOG_DIR="${FAST_ULYSSES_LOG_DIR:-/home/admin/fast_ulysses_experiment/logs}"
RIFE_MODEL_PATH="${SGLANG_LINGBOT_RIFE_MODEL_PATH:-/home/admin/RIFE-4.22.lite}"
STAMP="$(date +%Y%m%d-%H%M%S)"
SERVER_LOG="${LOG_DIR}/lingbot-${BACKEND}-${STAMP}.server.log"
CLIENT_LOG="${LOG_DIR}/lingbot-${BACKEND}-${STAMP}.client.log"
SERVER_PID=""

mkdir -p "${LOG_DIR}"
test -d "${SGLANG_DIR}"
test -d "${MODEL_PATH}"
test -f "${CLIENT_PATH}"

if ss -ltn | grep -q ':8001 '; then
  echo "Port 8001 is already in use; refusing to disturb an existing service." >&2
  exit 1
fi

cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    kill -TERM -- "-${SERVER_PID}" >/dev/null 2>&1 || true
    for _ in $(seq 1 60); do
      if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
        break
      fi
      sleep 1
    done
    if kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
      kill -KILL -- "-${SERVER_PID}" >/dev/null 2>&1 || true
    fi
  fi
}
trap cleanup EXIT INT TERM

export PYTHONPATH="${SGLANG_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${NVSHMEM_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export LD_PRELOAD="/usr/lib64/libcuda.so.1${LD_PRELOAD:+:${LD_PRELOAD}}"
export NVSHMEM_DISABLE_NVLS="${NVSHMEM_DISABLE_NVLS:-1}"
export NVSHMEM_REMOTE_TRANSPORT="${NVSHMEM_REMOTE_TRANSPORT:-none}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export SGLANG_FAST_ULYSSES_TAG_POOL_SIZE="${SGLANG_FAST_ULYSSES_TAG_POOL_SIZE:-32}"

unset SGLANG_ENABLE_ULYSSES_P2P_A2A
unset LINGBOT_FORCE_P2P
unset LINGBOT_ULYSSES_JIT
unset SGLANG_ULYSSES_A2A_BACKEND
unset SGLANG_ULYSSES_A2A_TRANSFER

export SGLANG_LINGBOT_ENABLE_RIFE=1
export SGLANG_LINGBOT_RIFE_MODEL_PATH="${RIFE_MODEL_PATH}"
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
A2A_ARGS=()
if [[ "${BACKEND}" != "native" ]]; then
  A2A_ARGS=(
    --ulysses-a2a-backend "${BACKEND}"
    --ulysses-a2a-transfer auto
  )
fi
setsid sglang serve \
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
  "${A2A_ARGS[@]}" \
  --ssl-certfile /home/admin/genie3_deploy/server.crt \
  --ssl-keyfile /home/admin/genie3_deploy/server.key \
  >"${SERVER_LOG}" 2>&1 &
SERVER_PID="$!"

ready=0
for _ in $(seq 1 900); do
  if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    echo "Server exited before readiness. Tail:" >&2
    tail -120 "${SERVER_LOG}" >&2
    exit 1
  fi
  if grep -q "Application startup complete" "${SERVER_LOG}"; then
    ready=1
    break
  fi
  sleep 1
done
if [[ "${ready}" != "1" ]]; then
  echo "Server did not become ready within 900 seconds." >&2
  exit 1
fi

for repetition in $(seq 1 "${REPETITIONS}"); do
  echo "=== repetition ${repetition}/${REPETITIONS} ===" >>"${CLIENT_LOG}"
  python "${CLIENT_PATH}" \
    --ws-host 127.0.0.1 \
    --ws-port 8001 \
    --duration "${CLIENT_DURATION}" \
    >>"${CLIENT_LOG}" 2>&1
  if [[ "${repetition}" -lt "${REPETITIONS}" ]]; then
    sleep 5
  fi
done

if [[ "${BACKEND}" == "fast_ulysses" ]]; then
  grep -q "Initialized genuine fast_ulysses backend" "${SERVER_LOG}"
  grep -q "genuine fast_ulysses collective active" "${SERVER_LOG}"
fi

echo "LINGBOT_BACKEND=${BACKEND}"
echo "LINGBOT_SERVER_LOG=${SERVER_LOG}"
echo "LINGBOT_CLIENT_LOG=${CLIENT_LOG}"
