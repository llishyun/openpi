"""Verify full pretrained pi0.5 loading and 50-action serving; not a task-success test."""

import json
from pathlib import Path
import time

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np

from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config


def main():
    cfg = config.get_config("pi05_franka_pnp_overfit1_fix3_50hz")
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    base = download.get_cache_dir() / "openpi-assets/checkpoints/pi05_base"
    assert (base / "params/_METADATA").is_file()
    started = time.monotonic()
    policy = policy_config.create_trained_policy(cfg, base, norm_stats=dc.norm_stats)
    ds = LeRobotDataset(dc.repo_id)
    rows = []
    for i in [0, 213, 1232]:
        d = ds[i]

        def image(k, item=d):
            return np.ascontiguousarray((item[k].numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8))

        obs = {
            "observation/exterior_image_1_left": image("observation.images.exterior_image_1_left"),
            "observation/wrist_image_left": image("observation.images.wrist_image_left"),
            "observation/state": d["observation.state"].numpy(),
            "prompt": d["task"],
        }
        out = policy.infer(obs)["actions"]
        assert out.shape == (50, 8)
        assert np.isfinite(out).all()
        rows.append({"frame": i, "shape": list(out.shape), "finite": True})
    result = {
        "status": "PASS",
        "full_pretrained_model": True,
        "fine_tuned": False,
        "samples": rows,
        "metadata": policy.metadata,
        "wall_seconds": time.monotonic() - started,
    }
    Path("diagnostics/overfit1_fix3_50hz/base_model_probe.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
