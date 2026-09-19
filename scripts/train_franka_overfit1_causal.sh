#!/usr/bin/env bash
set -euo pipefail
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_dir"
export CHECKPOINT_BASE_DIR=${CHECKPOINT_BASE_DIR:-"$project_dir/checkpoints"}
export EXP_NAME=${EXP_NAME:-overfit1_causal_8g}
export WANDB_MODE=${WANDB_MODE:-offline}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python_bin=${PYTHON_BIN:-"$project_dir/.venv/bin/python"}
"$python_bin" scripts/franka_causal_preflight.py
if [[ "${PREFLIGHT_ONLY:-0}" == 1 ]]; then exit 0; fi
args=(pi05_franka_pnp_overfit1_causal "--exp-name=$EXP_NAME" "--checkpoint-base-dir=$CHECKPOINT_BASE_DIR")
if [[ "${SMOKE_TEST:-0}" == 1 ]]; then
    args+=(--num-train-steps=2 --save-interval=1 --inference-save-steps 1 --no-wandb-enabled)
fi
if [[ "${RESUME:-0}" == 1 ]]; then args+=(--resume); fi
exec "$python_bin" scripts/train.py "${args[@]}"
