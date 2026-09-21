"""Compute full-dataset delta-action stats, including the final partial batch."""

import argparse
import json

from lerobot.common.constants import HF_LEROBOT_HOME
import numpy as np
import pyarrow.parquet as pq

from openpi.shared import normalize
from openpi.training import config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pi05_franka_pnp_overfit1_fix3_50hz")
    cfg = config.get_config(parser.parse_args().config)
    root = HF_LEROBOT_HOME / cfg.data.repo_id
    info = json.loads((root / "meta/info.json").read_text())
    assert info["fps"] == 50
    assert cfg.model.action_horizon == 50
    stats = {key: normalize.RunningStats() for key in ["state", "actions"]}
    total = 0
    for ep in range(info["total_episodes"]):
        path = root / info["data_path"].format(episode_chunk=ep // info["chunks_size"], episode_index=ep)
        data = pq.read_table(path).to_pydict()
        state = np.array(data["observation.state"], np.float32)
        action = np.array(data["action"], np.float32)
        for start in range(0, len(state), cfg.batch_size):
            rows = np.arange(start, min(start + cfg.batch_size, len(state)))
            chunks = action[np.minimum(rows[:, None] + np.arange(cfg.model.action_horizon), len(action) - 1)].copy()
            chunks[..., :7] -= state[rows, None, :7]
            stats["state"].update(state[rows])
            stats["actions"].update(chunks)
        total += len(state)
    assert total == info["total_frames"]
    output = cfg.assets_dirs / cfg.data.repo_id
    normalize.save(output, {k: v.get_statistics() for k, v in stats.items()})
    print(f"Computed from all {total} frames and {cfg.model.action_horizon}-action chunks: {output}")


if __name__ == "__main__":
    main()
