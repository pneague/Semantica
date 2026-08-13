#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: scripts/submit_decentralized_semantica_100.sh RUN_ROOT" >&2
  exit 2
fi

RUN_ROOT="$1"
LOG_ROOT="/var/scratch/pmneague/Low_Coord_Semantic_DHT/logs"

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
scontrol show node >/dev/null
sbatch scripts/run_decentralized_semantica_100.sbatch "${RUN_ROOT}"
