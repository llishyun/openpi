#!/bin/bash
#SBATCH --job-name=pi05_franka_pnp
#SBATCH --partition=A100-80GB                # n[51-59,76-80]: 8x A100-80GB SXM4 (NVLink), 3.8 TB /local-data, 2 TB RAM
#SBATCH --qos=hpgpu                          # A100-80GB allows only hpgpu/add_hpgpu/test (not normal)
# 48 GB alternative (normal QoS):  sbatch -p A6000,RTX6000ADA,L40S -q normal --gres=gpu:4 scripts/slurm/train.sbatch <exp>
#SBATCH --gres=gpu:2                         # 2x80 GB is enough for pi05 full FT @ batch 32 (fsdp over all GPUs); gpu:4 is faster
#SBATCH --cpus-per-task=16                 # config num_workers=16
#SBATCH --mem=160G
#SBATCH --time=3-00:00:00                  # set to the partition max; the job requeues itself and resumes
#SBATCH --signal=B:SIGUSR1@600             # 10 min before the limit: graceful stop, sync, requeue
#SBATCH --requeue
#SBATCH --output=/local-data/user-data/%u/job_%j/slurm/%x_%j.out
#SBATCH --error=/local-data/user-data/%u/job_%j/slurm/%x_%j.err
#
# Usage (from the openpi checkout, after `bash scripts/slurm/prepare.sh` on the login node):
#   sbatch scripts/slurm/train.sbatch [EXP_NAME] [extra train.py args...]
#   sbatch scripts/slurm/train.sbatch base100_full
#   CONFIG=pi05_franka_pnp_lora sbatch --gres=gpu:1 scripts/slurm/train.sbatch base100_lora
#
# Layout
#   local SSD  $LOCAL_JOB_BASE/checkpoints/<CONFIG>/<EXP>/<step>/   orbax writes here (fast, per-job, wiped after)
#   home       $PERSIST_ROOT/<CONFIG>/<EXP>/<step>/                 synced every SYNC_MIN minutes and at exit:
#              params+assets for every completed step, train_state only for the latest (needed to resume)
#   Re-submitting with the same EXP_NAME copies the latest step back to local SSD and resumes.
# Env knobs: CONFIG, PERSIST_ROOT (default ~/openpi_runs), SYNC_MIN (default 20), WANDB=1.

set -uo pipefail

########################################
# 1. per-job local SSD from the prolog
########################################
ENV_FILE="/run/slurm/job_env_${SLURM_JOB_ID}"
[ -f "$ENV_FILE" ] && source "$ENV_FILE"
if [ -z "${LOCAL_JOB_BASE:-}" ]; then echo "LOCAL_JOB_BASE not set"; exit 1; fi

########################################
# 2. paths / args
########################################
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
CONFIG=${CONFIG:-pi05_franka_pnp}
EXP_NAME=${1:-base100_full}; shift || true
EXTRA_ARGS=("$@")
PERSIST_ROOT=${PERSIST_ROOT:-$HOME/openpi_runs}
SYNC_MIN=${SYNC_MIN:-20}

LOCAL_CKPT_BASE="$LOCAL_JOB_BASE/checkpoints"
LOCAL_DIR="$LOCAL_CKPT_BASE/$CONFIG/$EXP_NAME"
PERSIST_DIR="$PERSIST_ROOT/$CONFIG/$EXP_NAME"
LOG_DIR="$LOCAL_JOB_BASE/logs"
SLURM_DIR="$LOCAL_JOB_BASE/slurm"
mkdir -p "$LOCAL_DIR" "$PERSIST_DIR" "$LOG_DIR" "$SLURM_DIR"

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export HF_LEROBOT_HOME=${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}
export HF_HUB_OFFLINE=1                    # dataset + pi05_base were fetched by prepare.sh
PYTHON="$PWD/.venv/bin/python"             # created by prepare.sh (uv sync); no conda/docker needed

GPU_COUNT=$(nvidia-smi -L | wc -l)
if (( 32 % GPU_COUNT != 0 )); then echo "batch 32 not divisible by GPU_COUNT=${GPU_COUNT}"; exit 1; fi

completed_steps() {  # completed orbax step dirs (all digits; interrupted saves have a .orbax-checkpoint-tmp suffix)
    ls "$1" 2>/dev/null | grep -E '^[0-9]+$' | sort -n
}

########################################
# 3. resume: pull the latest persisted step back to local SSD
########################################
MODE_FLAG=""
LATEST=$(completed_steps "$PERSIST_DIR" | tail -1)
if [ -n "$LATEST" ] && [ -d "$PERSIST_DIR/$LATEST/train_state" ]; then
    echo "[$(date +%T)] resuming from $PERSIST_DIR/$LATEST"
    rsync -a "$PERSIST_DIR/$LATEST" "$LOCAL_DIR/"
    cp -n "$PERSIST_DIR"/wandb_id.txt "$LOCAL_DIR/" 2>/dev/null || true
    MODE_FLAG="--resume"
fi
WANDB_FLAG="--no-wandb-enabled"; [ "${WANDB:-0}" = "1" ] && WANDB_FLAG="--wandb-enabled"

########################################
# 4. sync local -> home (params/assets for all steps, train_state for the latest only)
########################################
sync_ckpt() {
    local steps latest s
    steps=$(completed_steps "$LOCAL_DIR"); [ -z "$steps" ] && return 0
    latest=$(echo "$steps" | tail -1)
    for s in $steps; do
        mkdir -p "$PERSIST_DIR/$s"
        rsync -a --exclude 'train_state' "$LOCAL_DIR/$s/" "$PERSIST_DIR/$s/"
    done
    rsync -a "$LOCAL_DIR/$latest/train_state" "$PERSIST_DIR/$latest/"
    for s in $(completed_steps "$PERSIST_DIR"); do          # drop stale optimizer states
        [ "$s" != "$latest" ] && rm -rf "$PERSIST_DIR/$s/train_state"
    done
    cp -f "$LOCAL_DIR"/wandb_id.txt "$PERSIST_DIR/" 2>/dev/null || true
    echo "[$(date +%T)] synced steps: $(echo $steps | tr '\n' ' ') (latest $latest) -> $PERSIST_DIR"
}
sync_loop() { while true; do sleep "$((SYNC_MIN * 60))"; sync_ckpt; done; }

finish() {  # copy logs home like the cluster template, plus a final checkpoint sync
    pkill -P "$SYNC_PID" 2>/dev/null; kill "$SYNC_PID" 2>/dev/null; wait "$SYNC_PID" 2>/dev/null   # loop + its sleep
    sync_ckpt
    FINAL_DIR="$HOME/cluster_log/${SLURM_JOB_USER}_job_${SLURM_JOB_ID}"
    mkdir -p "$FINAL_DIR"; cp -r "$SLURM_DIR" "$FINAL_DIR/slurm" 2>/dev/null; cp -r "$LOG_DIR" "$FINAL_DIR/logs" 2>/dev/null
    echo "[$(date +%T)] logs -> $FINAL_DIR ; checkpoints -> $PERSIST_DIR"
}
requeue() {
    echo "[$(date +%T)] SIGUSR1: stopping trainer, syncing, requeueing"
    kill -TERM "$TRAIN_PID" 2>/dev/null; wait "$TRAIN_PID" 2>/dev/null
    finish
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
}
trap requeue SIGUSR1

########################################
# 5. train
########################################
echo "
Node: $(hostname -s)   JOB: ${SLURM_JOB_ID}   $(TZ=Asia/Seoul date)
CONFIG: ${CONFIG}   EXP: ${EXP_NAME}   GPUs: ${GPU_COUNT}   mode: ${MODE_FLAG:-fresh}
LOCAL_DIR:   ${LOCAL_DIR}
PERSIST_DIR: ${PERSIST_DIR}
EXTRA: ${EXTRA_ARGS[*]:-}
"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
df -h "$LOCAL_JOB_BASE" | tail -1

sync_loop & SYNC_PID=$!
"$PYTHON" scripts/train.py "$CONFIG" \
    --exp-name="$EXP_NAME" \
    --checkpoint-base-dir="$LOCAL_CKPT_BASE" \
    --fsdp-devices="$GPU_COUNT" \
    ${MODE_FLAG} ${WANDB_FLAG} "${EXTRA_ARGS[@]}" > >(tee "$LOG_DIR/train_${SLURM_JOB_ID}.log") 2>&1 &
TRAIN_PID=$!                               # python's pid (not tee's), so SIGUSR1 handling can stop it
wait "$TRAIN_PID"; STATUS=$?
echo "[$(date +%T)] train.py exited with ${STATUS}"
finish
exit ${STATUS}
