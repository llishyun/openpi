#!/bin/bash
#SBATCH --partition=A6000,RTX6000ADA,L40S
#SBATCH --gres=gpu:4                      # 48 GB x4 is enough for pi05 full FT @ batch 32 (fsdp over all GPUs)
#SBATCH --cpus-per-task=16                # config num_workers=16
#SBATCH --mem=160G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=pi05_franka_pnp
#SBATCH --output=slurm_outputs/%x_%j.log
#SBATCH --signal=B:SIGUSR1@600            # 10 min before the time limit -> graceful stop + requeue
#SBATCH --requeue
#
# Usage (from the openpi checkout, after scripts/slurm/prepare.sh):
#   sbatch scripts/slurm/train.sbatch [EXP_NAME] [extra train.py args...]
#   sbatch scripts/slurm/train.sbatch base100_full
#   CONFIG=pi05_franka_pnp_lora sbatch --gres=gpu:1 scripts/slurm/train.sbatch base100_lora
#
# Re-submitting (or an automatic requeue) with the same EXP_NAME resumes from the last saved step.
# Env knobs: CONFIG (train config name), CKPT_BASE (checkpoint root, put it on scratch), WANDB=1 to log to W&B.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
mkdir -p slurm_outputs

CONFIG=${CONFIG:-pi05_franka_pnp}
EXP_NAME=${1:-base100_full}; shift || true
EXTRA_ARGS=("$@")
CKPT_BASE=${CKPT_BASE:-$PWD/checkpoints}
CKPT_DIR="${CKPT_BASE}/${CONFIG}/${EXP_NAME}"

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export HF_LEROBOT_HOME=${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}
export HF_HUB_OFFLINE=1                   # dataset + pi05_base were fetched by prepare.sh
export PATH="$PATH:$HOME/.local/bin"      # uv installed by prepare.sh

GPU_COUNT=$(nvidia-smi -L | wc -l)
if (( 32 % GPU_COUNT != 0 )); then echo "batch 32 not divisible by GPU_COUNT=${GPU_COUNT}"; exit 1; fi

# Resume iff a completed step directory exists (tmp dirs from an interrupted save are ignored by orbax).
MODE_FLAG=""
if [ -d "${CKPT_DIR}" ]; then
    if ls "${CKPT_DIR}" 2>/dev/null | grep -qE '^[0-9]+$'; then MODE_FLAG="--resume"; else MODE_FLAG="--overwrite"; fi
fi
WANDB_FLAG="--no-wandb-enabled"; [ "${WANDB:-0}" = "1" ] && WANDB_FLAG="--wandb-enabled"

echo "
Node: $(hostname -s)   JOB: ${SLURM_JOBID}   $(TZ=Asia/Seoul date)
CONFIG: ${CONFIG}   EXP: ${EXP_NAME}   GPUs: ${GPU_COUNT}   mode: ${MODE_FLAG:-fresh}
CKPT_DIR: ${CKPT_DIR}
EXTRA: ${EXTRA_ARGS[*]:-}
"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# Graceful stop on the pre-timeout signal, then requeue; the requeued job resumes from the last checkpoint.
requeue() {
    echo "[$(date +%T)] SIGUSR1: stopping trainer and requeueing"
    kill -TERM "${TRAIN_PID}" 2>/dev/null; wait "${TRAIN_PID}" 2>/dev/null
    scontrol requeue "${SLURM_JOBID}"
    exit 0
}
trap requeue SIGUSR1

uv run scripts/train.py "${CONFIG}" \
    --exp-name="${EXP_NAME}" \
    --checkpoint-base-dir="${CKPT_BASE}" \
    --fsdp-devices="${GPU_COUNT}" \
    ${MODE_FLAG} ${WANDB_FLAG} "${EXTRA_ARGS[@]}" &
TRAIN_PID=$!
wait "${TRAIN_PID}"
STATUS=$?
echo "[$(date +%T)] train.py exited with ${STATUS}"
exit ${STATUS}
