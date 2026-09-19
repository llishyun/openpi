"""Preflight for the four/eight-GPU causal experiment; does not train."""

import dataclasses
import json
import os
from pathlib import Path
import shutil

import jax
from lerobot.common.constants import HF_LEROBOT_HOME

from openpi.training import config

# Uncompressed arrays from eval_shape of this full-finetuning model + AdamW + EMA.
FULL_CHECKPOINT_BYTES = 53_654_941_964
INFERENCE_CHECKPOINT_BYTES = 13_413_735_488


def main():
    cfg = config.get_config("pi05_franka_pnp_overfit1_causal")
    cfg = dataclasses.replace(
        cfg,
        exp_name=os.environ["EXP_NAME"],
        checkpoint_base_dir=os.environ["CHECKPOINT_BASE_DIR"],
        fsdp_devices=int(os.environ.get("GPU_COUNT", "8")),
    )
    if cfg.fsdp_devices not in (4, 8):
        raise SystemExit("GPU_COUNT must be 4 or 8.")
    if os.environ.get("SMOKE_TEST") == "1":
        cfg = dataclasses.replace(cfg, num_train_steps=2, save_interval=1, inference_save_steps=(1,))
    if cfg.keep_period is not None:
        raise SystemExit("This storage estimate requires keep_period=None (latest full checkpoint only).")
    run = cfg.checkpoint_dir
    if not cfg.exp_name or Path(cfg.exp_name).name != cfg.exp_name:
        raise SystemExit("EXP_NAME must be a single directory name.")
    exports = run.parent / f"{run.name}_inference"
    if os.environ.get("RESUME") == "1":
        if not run.is_dir():
            raise SystemExit(f"No run to resume: {run}")
    elif run.exists() or exports.exists():
        raise SystemExit(f"Run or exports already exist: {run}. Use another EXP_NAME or RESUME=1.")
    # Conservative reserve, even on resume: two full states during replacement,
    # all selected inference exports, and 10% overhead. Check filesystem quotas too.
    required = int((2 * FULL_CHECKPOINT_BYTES + len(cfg.inference_save_steps) * INFERENCE_CHECKPOINT_BYTES) * 1.1)
    parent = run.parent
    while not parent.exists():
        parent = parent.parent
    free = shutil.disk_usage(parent).free
    if free < required:
        raise SystemExit(
            f"Insufficient space: {free / 2**30:.1f} GiB free; {required / 2**30:.1f} GiB reserve required."
        )
    dataset = HF_LEROBOT_HOME / "lithyeon/franka_pnp_overfit1_causal"
    info = json.loads((dataset / "meta/info.json").read_text())
    if info["total_episodes"] != 1 or info["total_frames"] != 323:
        raise SystemExit(f"Unexpected dataset metadata: {dataset}")
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    if dc.norm_stats is None:
        raise SystemExit("Missing causal normalization statistics.")
    # Use CUDA-visible JAX devices, not physical nvidia-smi IDs that Slurm may remap.
    devices = jax.local_devices()
    if len(devices) != cfg.fsdp_devices or any(d.platform != "gpu" for d in devices):
        raise SystemExit(f"Expected {cfg.fsdp_devices} allocated GPUs; got {devices}")
    print(f"Preflight passed: {run}; free {free / 2**30:.1f} GiB; reserve {required / 2**30:.1f} GiB.")
    print(f"GPUs: {[d.device_kind for d in devices]}; dataset: {dataset}")


if __name__ == "__main__":
    main()
