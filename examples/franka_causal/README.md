# One-demo causal-state full fine-tuning on GSAI

This experiment trains **one demo**, not all 450 demos. Use branch `franka_pnp`
of `https://github.com/llishyun/openpi`. The simulator changes are separate;
Isaac Sim is not needed on the training cluster.

## Inputs and recipe

- Config: `pi05_franka_pnp_overfit1_causal`.
- Local dataset ID: `lithyeon/franka_pnp_overfit1_causal`; one episode, 323 frames,
  10 Hz; scene `base_big_r101_c00`, source `demo_c003_idx116_succ.hdf5`.
- State is reconstructed **before** subsampling as initial state + previous
  post-step states. Original actions are unchanged. All three first camera
  frames are repaired; later decoded frames are unchanged.
- Corrected 450-demo normalization is committed under
  `assets/pi05_franka_pnp_cells450_causal/`; do not recompute it from one demo.
- Full pi0.5 from `pi05_base`; action horizon 10; no image augmentation;
  batch 32; FSDP 8; AdamW; EMA .999; 20,000 steps; warmup 1,000;
  peak LR 2.5e-5, final LR 2.5e-6. Do not resume the old non-causal experiment.

## Prepare on the login node

```bash
git clone --branch franka_pnp https://github.com/llishyun/openpi.git
cd openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen
```

For an existing checkout, use `git pull --ff-only` on `franka_pnp` instead.
The single-demo archive is included in Git at
`data/franka_pnp_overfit1_causal.tar.gz` (about 29 MB); no separate transfer is needed.
The installer verifies every file against `dataset_sha256.json`, rejects
unexpected members, and refuses to overwrite an existing dataset.

```bash
.venv/bin/python scripts/install_franka_causal_data.py data/franka_pnp_overfit1_causal.tar.gz
.venv/bin/python -c "from openpi.shared import download; print(download.maybe_download('gs://openpi-assets/checkpoints/pi05_base/params'))"
```

Set `HF_LEROBOT_HOME` before installing and submitting if the dataset belongs
outside the default `~/.cache/huggingface/lerobot`. Keep the base model cache
on a filesystem visible from compute nodes. Installation/download needs network
access; the submitted training job uses the prepared local data.

## Allocation and smoke test

### Queue alternatives; train only the first eligible job

To try four/eight GPUs across eligible partitions with an eight-hour limit:

```bash
export CHECKPOINT_BASE_DIR="$HOME/openpi_runs"
export EXP_NAME=overfit1_causal_race
.venv/bin/python scripts/slurm/causal_race.py submit --replace-pending-job 969068
```

Omit `--replace-pending-job` if there is no old job to replace. Use a fresh
experiment name per submission group. Defaults are four candidates:

- A100-80GB / hpgpu / 8 GPUs
- A100-80GB / hpgpu / 4 GPUs
- A6000,RTX6000ADA,L40S / normal / 8 GPUs (one eligible partition, not multiple nodes)
- A6000,RTX6000ADA,L40S / normal / 4 GPUs

Rejected submissions (e.g. unavailable QoS) are reported and skipped. You can
override candidates with repeated `--candidate partition/qos/4-or-8` options.
All accepted jobs are held until their exact IDs have been recorded on shared
storage. If replacing an old job, it is cancelled only if still pending; if
it has started, the new candidates are cancelled instead. Then holds are released.
The first job to pass preflight atomically creates a permanent winner marker
and cancels only other IDs from this group. Simultaneous losers exit without
training even if cancellation is delayed. This elects the first preflight-ready
job, not the first to complete a training step; an eventual OOM/failure will
not automatically restart a cancelled alternative.

The shared directory `$CHECKPOINT_BASE_DIR/.causal_races/$EXP_NAME` contains
`jobs.json` and `winner.json`. It must support atomic exclusive file creation.
Do not delete the winner marker to restart a race. Resume the winning experiment
as a single job with its recorded GPU count, e.g. for a four-GPU A100 winner:

```bash
GPU_COUNT=4 RESUME=1 EXP_NAME=overfit1_causal_race \
  sbatch --gres=gpu:4 --time=08:00:00 scripts/slurm/franka_causal.sbatch
```

GPU_COUNT controls FSDP and preflight together; global batch remains 32.
The race submission client runs on the login node, not through sbatch.
See [Slurm held jobs](https://slurm.schedmd.com/sbatch.html#OPT_hold) and
[job release](https://slurm.schedmd.com/scontrol.html).

### Submit a single allocation

The default is **one node, eight A100 80GB GPUs, one Python process**, 16 CPUs,
160 GiB RAM. The existing GSAI scripts use `hpgpu` QoS for `A100-80GB`; verify
your association, partition time limits and storage quota before submission.
The supplied `sinfo` listing is a snapshot: `mix` does not guarantee eight free GPUs.

Choose a **persistent/shared** checkpoint directory with at least **152 GiB
available and quota headroom** (base model cache/environment are additional).
No per-job local SSD sync or automatic requeue is implemented by this script.
At a time limit, only the last completed checkpoint can be resumed.

```bash
export CHECKPOINT_BASE_DIR="$HOME/openpi_runs"
SMOKE_TEST=1 EXP_NAME=causal_smoke sbatch --time=01:00:00 scripts/slurm/franka_causal.sbatch
```

Check `slurm-pi05_causal-<jobid>.out/.err`. The smoke job uses the real model,
batch and dataset for two steps and writes checkpoint `1` plus an inference
export. It is not yet validated on GSAI. After it succeeds, check resume:

```bash
SMOKE_TEST=1 RESUME=1 EXP_NAME=causal_smoke sbatch --time=01:00:00 scripts/slurm/franka_causal.sbatch
```

This restores the completed two-step state and exits without extra updates.
Then launch the full run with a different experiment name:

```bash
EXP_NAME=overfit1_causal_8g sbatch scripts/slurm/franka_causal.sbatch
# If interrupted after a completed checkpoint:
RESUME=1 EXP_NAME=overfit1_causal_8g sbatch scripts/slurm/franka_causal.sbatch
```

Use `--partition=A100-40GB --qos=<allowed-qos>` (the 8-GPU SXM partition, not the 4-GPU PCIe partition) or another eligible partition
only after checking access and running the smoke test. The fixed recipe still
requires eight GPUs on one node. Do not override Slurm's `CUDA_VISIBLE_DEVICES`.
Slurm references: [sbatch](https://slurm.schedmd.com/sbatch.html),
[GPU allocation](https://slurm.schedmd.com/gres.html).

## Checkpoint policy and evaluation

Only the latest full checkpoint is retained (`keep_period=None`):
`$CHECKPOINT_BASE_DIR/pi05_franka_pnp_overfit1_causal/<EXP_NAME>/<step>`.
Separate params/assets exports at steps 5000, 10000 and 19999 are retained in
the sibling `<EXP_NAME>_inference/<step>` directory. These exports **cannot
resume training**. They can be passed directly to `serve_policy.py`:

```bash
.venv/bin/python scripts/serve_policy.py --port 8004 policy:checkpoint \
  --policy.config=pi05_franka_pnp_overfit1_causal \
  --policy.dir=/path/to/overfit1_causal_8g_inference/19999
```

Storage estimate: ~50 GiB per full checkpoint, ~12.5 GiB per inference export;
~87.5 GiB retained after the run, ~151.2 GiB free-space reserve for replacement
and copying overhead. Older runs keep their original retention settings.
W&B defaults to offline mode; logs are under `wandb/`.

Transfer inference exports back to the simulator machine. Evaluate with the
corrected exact-initial-state path, zero reset noise, control stride 5 and
replan interval 5. Prior validation: all 323 data frames and normalization
checked, two real loader batches passed, expert replay succeeded 2/2. These
are not evidence that the newly trained policy succeeds; repeated closed-loop
evaluation remains required.
