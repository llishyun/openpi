"""Verify the immutable 50 Hz experiment inputs before allocating training work."""

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import shutil

from lerobot.common.constants import HF_LEROBOT_HOME

from openpi.shared import download
from openpi.training import config

CONFIG = os.environ.get("OPENPI_FRANKA_CONFIG", "pi05_franka_pnp_overfit1_fix3_50hz")
EXPERIMENTS = {
    "pi05_franka_pnp_overfit1_fix3_50hz": "franka_50hz",
    "pi05_franka_pnp_center5_v2_50hz": "franka_center5_50hz",
    "pi05_franka_pnp_mid10x15_50hz": "franka_mid10x15_50hz",
}
PROJECT = Path(__file__).resolve().parents[1]


def verify_inputs():
    cfg = config.get_config(CONFIG)
    experiment_dir = PROJECT / "examples" / EXPERIMENTS[CONFIG]
    expected = json.loads((experiment_dir / "experiment.json").read_text())
    manifest = json.loads((experiment_dir / "dataset_sha256.json").read_text())
    assert len(manifest) >= 15, "Incomplete dataset manifest"
    assert any(name.endswith(".hdf5") for name in manifest), "Missing replay reference"
    root = HF_LEROBOT_HOME / cfg.data.repo_id
    for name, expected_hash in manifest.items():
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"Dataset file missing or changed: {path}")
    info = json.loads((root / "meta/info.json").read_text())
    assert (info["fps"], info["total_frames"], info["total_episodes"]) == (50, expected["frames"], expected["episodes"])
    norm = cfg.assets_dirs / cfg.data.repo_id / "norm_stats.json"
    assert hashlib.sha256(norm.read_bytes()).hexdigest() == expected["norm_sha256"], "Changed norm stats"
    assert cfg.model.action_horizon == 50
    assert cfg.model.augment_images is expected["image_augmentation"]
    assert cfg.policy_metadata["action_fps"] == 50
    assert cfg.policy_metadata["action_horizon"] == 50
    assert cfg.keep_period is None
    assert cfg.inference_save_steps == (5000, 10000, 19999)
    assert cfg.batch_size == 32
    assert cfg.num_train_steps == 20000
    return cfg, root


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-only", action="store_true")
    args = p.parse_args()
    cfg, root = verify_inputs()
    if args.data_only:
        print(f"Data and normalization verified: {root}")
        return
    import jax

    count = int(os.environ.get("GPU_COUNT", "8"))
    assert count in (4, 8), "GPU_COUNT must be 4 or 8"
    devices = jax.local_devices()
    assert len(devices) == count, (count, devices)
    assert all(d.platform == "gpu" for d in devices), devices
    assert cfg.batch_size % count == 0
    cfg = dataclasses.replace(
        cfg, exp_name=os.environ["EXP_NAME"], checkpoint_base_dir=os.environ["CHECKPOINT_BASE_DIR"]
    )
    assert cfg.exp_name, "EXP_NAME is empty"
    assert Path(cfg.exp_name).name == cfg.exp_name, "EXP_NAME must be a directory name"
    run = cfg.checkpoint_dir
    exports = run.parent / (run.name + "_inference")
    if os.environ.get("RESUME") == "1":
        assert run.is_dir(), f"No existing 50 Hz run: {run}"
    else:
        assert not run.exists(), f"Run exists: {run}; choose a fresh EXP_NAME"
        assert not exports.exists(), f"Exports exist: {exports}; choose a fresh EXP_NAME"
    parent = run.parent
    while not parent.exists():
        parent = parent.parent
    required = int((2 * 53_654_941_964 + 3 * 13_413_735_488) * 1.1)
    assert shutil.disk_usage(parent).free >= required, (
        f"Need at least {required / 2**30:.1f} GiB free (also check quota)"
    )
    cache = download.get_cache_dir()
    base = cache / "openpi-assets/checkpoints/pi05_base/params"
    tokenizer = cache / "big_vision/paligemma_tokenizer.model"
    assert (base / "_METADATA").is_file(), f"Prepare base model on login node: {base}"
    assert (base / "manifest.ocdbt").is_file(), f"Missing base model arrays: {base}"
    assert tokenizer.is_file(), f"Prepare tokenizer on login node: {tokenizer}"
    cfg.data.create(cfg.assets_dirs, cfg.model)
    print(f"Preflight PASS: {count} GPUs; {CONFIG} @ 50 Hz; run={run}; reserve={required / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
