#!/usr/bin/env bash
set -euo pipefail
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_dir"
: "${CHECKPOINT_BASE_DIR:?Set a persistent directory}"
: "${EXP_NAME:?Set a fresh 50 Hz experiment name}"
export GPU_COUNT=${GPU_COUNT:-8}
export WANDB_MODE=${WANDB_MODE:-offline}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
python_bin=${PYTHON_BIN:-"$project_dir/.venv/bin/python"}
"$python_bin" scripts/franka_50hz_preflight.py
if [[ "${PREFLIGHT_ONLY:-0}" == 1 ]]; then exit 0; fi
smoke_report="$CHECKPOINT_BASE_DIR/50hz_validation/${EXP_NAME}_${SLURM_JOB_ID:-local}_smoke.json"
# Every allocation must pass a full-model, full-batch smoke before the long run.
"$python_bin" scripts/smoke_franka_50hz.py --config "${OPENPI_FRANKA_CONFIG:-pi05_franka_pnp_overfit1_fix3_50hz}" --output "$smoke_report"
if [[ "${SMOKE_ONLY:-0}" == 1 ]]; then exit 0; fi
args=("${OPENPI_FRANKA_CONFIG:-pi05_franka_pnp_overfit1_fix3_50hz}" "--exp-name=$EXP_NAME" "--checkpoint-base-dir=$CHECKPOINT_BASE_DIR" "--fsdp-devices=$GPU_COUNT")
if [[ "${RESUME:-0}" == 1 ]]; then args+=(--resume); fi
exec "$python_bin" scripts/train.py "${args[@]}"
