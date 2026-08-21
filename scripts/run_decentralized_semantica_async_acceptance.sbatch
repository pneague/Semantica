#!/usr/bin/env bash
#SBATCH --job-name=semantica-async-join
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
REFERENCE_RUN="${REFERENCE_RUN:?REFERENCE_RUN must identify the 100-peer rollback run}"
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

mkdir -p "${RUN_ROOT}" "${MPLCONFIGDIR}"
cd "${PROJECT_DIR}"

"${PYTHON_BIN}" -m decentralized_semantica prepare \
  --output-dir "${BASE_PREPARED}" \
  --peer-count 100 \
  --seed 42

"${PYTHON_BIN}" -m decentralized_semantica prepare \
  --output-dir "${LATE_PREPARED}" \
  --peer-count 101 \
  --seed 42

"${PYTHON_BIN}" -m decentralized_semantica run \
  --prepared-dir "${BASE_PREPARED}" \
  --run-dir "${RUNTIME_DIR}" \
  --rpc-timeout-seconds 30 \
  --root-peer-id 0 \
  --run-timeout-seconds 10800 \
  --keep-active >"${RUN_ROOT}/runner.json" &
RUNNER_PID=$!

cleanup() {
  if kill -0 "${RUNNER_PID}" 2>/dev/null; then
    kill "${RUNNER_PID}" 2>/dev/null || true
  fi
  if [[ -n "${JOIN_PID:-}" ]] && kill -0 "${JOIN_PID}" 2>/dev/null; then
    kill "${JOIN_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

deadline=$((SECONDS + 900))
while [[ ! -f "${RUNTIME_DIR}/overlay_active.json" ]]; do
  if ! kill -0 "${RUNNER_PID}" 2>/dev/null; then
    wait "${RUNNER_PID}"
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    echo "timed out waiting for the initial overlay" >&2
    exit 1
  fi
  sleep 1
done

mkdir -p "${RUN_ROOT}/prejoin-run"
cp -a "${RUNTIME_DIR}/state" "${RUN_ROOT}/prejoin-run/state"

"${PYTHON_BIN}" -m decentralized_semantica join \
  --bootstrap "${RUNTIME_DIR}/bootstrap.json" \
  --peer-id 100 \
  --shard "${LATE_PREPARED}/shards/peer_00100.npz" \
  --run-dir "${RUNTIME_DIR}" >"${RUN_ROOT}/late-peer.json" &
JOIN_PID=$!

deadline=$((SECONDS + 900))
while ! grep -q '"event": "active".*"peer_id": 100' "${RUNTIME_DIR}/join_events.jsonl"; do
  if ! kill -0 "${RUNNER_PID}" 2>/dev/null || ! kill -0 "${JOIN_PID}" 2>/dev/null; then
    echo "runner or joining peer exited before late admission completed" >&2
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    echo "timed out waiting for late peer 100" >&2
    exit 1
  fi
  sleep 1
done

"${PYTHON_BIN}" -m decentralized_semantica finalize \
  --bootstrap "${RUNTIME_DIR}/bootstrap.json" >"${RUN_ROOT}/finalize.json"

wait "${RUNNER_PID}"
wait "${JOIN_PID}"
trap - EXIT

"${PYTHON_BIN}" -m decentralized_semantica validate \
  --prepared-dir "${BASE_PREPARED}" \
  --run-dir "${RUNTIME_DIR}" \
  --structural-only \
  --reference-run "${REFERENCE_RUN}" \
  --prejoin-run "${RUN_ROOT}/prejoin-run"
