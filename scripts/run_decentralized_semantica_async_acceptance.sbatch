#!/usr/bin/env bash
#SBATCH --job-name=semantica-v3-active
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --mem=0
#SBATCH --output=/var/scratch/pmneague/Low_Coord_Semantic_DHT/logs/%x-%j.out
#SBATCH --error=/var/scratch/pmneague/Low_Coord_Semantic_DHT/logs/%x-%j.err

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: sbatch scripts/run_decentralized_semantica_async_acceptance.sbatch RUN_ROOT" >&2
  exit 2
fi

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
PYTHON_BIN="${PYTHON_BIN:-/home/pmneague/.pyenv/versions/3.11.7/bin/python}"
RUN_ROOT="$1"
BASE_PREPARED="${RUN_ROOT}/prepared-base-100"
LATE_PREPARED="${RUN_ROOT}/prepared-with-late-peer"
RUNTIME_DIR="${RUN_ROOT}/run"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${RUN_ROOT}/matplotlib-cache"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${RUN_ROOT}" "${MPLCONFIGDIR}" "${RUNTIME_DIR}"
cd "${PROJECT_DIR}"

"${PYTHON_BIN}" -m decentralized_semantica prepare \
  --output-dir "${BASE_PREPARED}" \
  --peer-count 100 \
  --seed 42

"${PYTHON_BIN}" -m decentralized_semantica prepare \
  --output-dir "${LATE_PREPARED}" \
  --peer-count 101 \
  --seed 42

PEER_PIDS=()

cleanup() {
  for pid in "${PEER_PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
  for pid in "${PEER_PIDS[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
}
trap cleanup EXIT

"${PYTHON_BIN}" -m decentralized_semantica start-root \
  --prepared-dir "${BASE_PREPARED}" \
  --peer-id 0 \
  --run-dir "${RUNTIME_DIR}" >"${RUN_ROOT}/peer-00000.out" 2>&1 &
PEER_PIDS+=("$!")

deadline=$((SECONDS + 300))
while [[ ! -f "${RUNTIME_DIR}/bootstrap.json" ]]; do
  if (( SECONDS >= deadline )); then
    echo "timed out waiting for the genesis bootstrap" >&2
    exit 1
  fi
  sleep 1
done

for peer_id in $(seq 1 99); do
  peer_label=$(printf '%05d' "${peer_id}")
  "${PYTHON_BIN}" -m decentralized_semantica join \
    --bootstrap "${RUNTIME_DIR}/bootstrap.json" \
    --peer-id "${peer_id}" \
    --shard "${BASE_PREPARED}/shards/peer_${peer_label}.npz" \
    --run-dir "${RUNTIME_DIR}" >"${RUN_ROOT}/peer-${peer_label}.out" 2>&1 &
  PEER_PIDS+=("$!")
done

deadline=$((SECONDS + 3600))
while true; do
  active_count=$(
    { grep -l '"status": "active"' "${RUNTIME_DIR}"/status/peer_*.json 2>/dev/null || true; } \
      | wc -l
  )
  if [[ "${active_count}" -eq 100 ]]; then
    break
  fi
  if (( SECONDS >= deadline )); then
    echo "timed out waiting for 100 active peers; observed ${active_count}" >&2
    exit 1
  fi
  sleep 2
done

mkdir -p "${RUN_ROOT}/prejoin-run"
cp -a "${RUNTIME_DIR}/state" "${RUN_ROOT}/prejoin-run/state"

"${PYTHON_BIN}" -m decentralized_semantica join \
  --bootstrap "${RUNTIME_DIR}/bootstrap.json" \
  --peer-id 100 \
  --shard "${LATE_PREPARED}/shards/peer_00100.npz" \
  --run-dir "${RUNTIME_DIR}" >"${RUN_ROOT}/peer-00100.out" 2>&1 &
PEER_PIDS+=("$!")

deadline=$((SECONDS + 900))
while ! grep -q '"status": "active"' "${RUNTIME_DIR}/status/peer_00100.json" 2>/dev/null; do
  if (( SECONDS >= deadline )); then
    echo "timed out waiting for late peer 100" >&2
    exit 1
  fi
  sleep 2
done

# Every peer is stopped locally; no peer can finalize or shut down the overlay.
cleanup
trap - EXIT

"${PYTHON_BIN}" -m decentralized_semantica validate \
  --prepared-dir "${BASE_PREPARED}" \
  --run-dir "${RUNTIME_DIR}" \
  --structural-only \
  --prejoin-run "${RUN_ROOT}/prejoin-run"
