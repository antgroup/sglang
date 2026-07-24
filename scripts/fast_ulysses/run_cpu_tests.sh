#!/usr/bin/env bash
set -euo pipefail

SGLANG_DIR="${SGLANG_DIR:-/home/admin/sglang}"
LOG_DIR="${FAST_ULYSSES_LOG_DIR:-/home/admin/fast_ulysses_experiment/logs}"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="${LOG_DIR}/cpu-tests-${STAMP}.log"

mkdir -p "${LOG_DIR}"
cd "${SGLANG_DIR}"
export PYTHONPATH="${SGLANG_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}"
export LD_PRELOAD="/usr/lib64/libcuda.so.1${LD_PRELOAD:+:${LD_PRELOAD}}"

python test/registered/unit/test_diffusion_ulysses_a2a.py -v 2>&1 | tee "${LOG_FILE}"
echo "FAST_ULYSSES_CPU_TEST_LOG=${LOG_FILE}"
