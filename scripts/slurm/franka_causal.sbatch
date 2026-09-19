#!/usr/bin/env bash
#SBATCH --job-name=pi05_causal
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the openpi checkout}"
: "${CHECKPOINT_BASE_DIR:?Set a persistent checkpoint directory before sbatch}"
: "${EXP_NAME:?Set an experiment name before sbatch}"
export HF_LEROBOT_HOME=${HF_LEROBOT_HOME:-"$HOME/.cache/huggingface/lerobot"}
export HF_HUB_OFFLINE=1
export WANDB_MODE=${WANDB_MODE:-offline}
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
# Slurm sets CUDA_VISIBLE_DEVICES. Do not replace it with physical device indices.
srun --ntasks=1 bash scripts/train_franka_overfit1_causal.sh
