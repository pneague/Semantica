#!/usr/bin/env bash
#SBATCH --job-name=semantica-decentralized-100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --mem=0
#SBATCH --output=/var/scratch/pmneague/Low_Coord_Semantic_DHT/logs/%x-%j.out
#SBATCH --error=/var/scratch/pmneague/Low_Coord_Semantic_DHT/logs/%x-%j.err

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: sbatch scripts/run_decentralized_semantica_100.sbatch RUN_ROOT" >&2
  exit 2
fi

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
PYTHON_BIN="${PYTHON_BIN:-/home/pmneague/.pyenv/versions/3.11.7/bin/python}"
RUN_ROOT="$1"
PREPARED_DIR="${RUN_ROOT}/prepared"
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
  --output-dir "${PREPARED_DIR}" \
  --peer-count 100 \
  --seed 42

"${PYTHON_BIN}" -m decentralized_semantica run \
  --prepared-dir "${PREPARED_DIR}" \
  --run-dir "${RUNTIME_DIR}" \
  --rpc-timeout-seconds 30 \
  --root-peer-id 0 \
  --run-timeout-seconds 10800

VALIDATE_ARGS=(
  --prepared-dir "${PREPARED_DIR}"
  --run-dir "${RUNTIME_DIR}"
  --float32-atol 1e-5
)
if [[ -n "${REFERENCE_RUN:-}" ]]; then
  VALIDATE_ARGS+=(--reference-run "${REFERENCE_RUN}")
fi
"${PYTHON_BIN}" -m decentralized_semantica validate "${VALIDATE_ARGS[@]}"
