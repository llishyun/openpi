"""Audit every 50 Hz sample, camera timestamp, normalization and training batch."""

import argparse
import copy
import json
from pathlib import Path

import h5py
import imageio.v2 as imageio
from lerobot.common.constants import HF_LEROBOT_HOME
import numpy as np
import pyarrow.parquet as pq
import torch

from openpi import transforms
from openpi.policies.franka_pnp_policy import _parse_image
from openpi.shared import normalize
from openpi.training import config
from openpi.training import data_loader

CONFIG = "pi05_franka_pnp_overfit1_fix3_50hz"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("diagnostics/overfit1_fix3_50hz/data_audit.json"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(1)
    cfg = config.get_config(CONFIG)
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    root = HF_LEROBOT_HOME / dc.repo_id
    info = json.loads((root / "meta/info.json").read_text())
    assert info["fps"] == 50
    assert info["total_frames"] == 1291
    assert info["total_episodes"] == 1
    assert cfg.model.action_horizon == 50
    assert cfg.model.augment_images is False
    assert cfg.policy_metadata["action_fps"] == 50
    table = pq.read_table(root / "data/chunk-000/episode_000000.parquet").to_pydict()
    states, actions = [np.asarray(table[k], np.float32) for k in ["observation.state", "action"]]
    with h5py.File(root / "reference.hdf5") as f:
        g = f["data/demo_0"]
        pre = np.concatenate(
            [
                g["initial_state/articulation/robot/joint_position"][:],
                g["states/articulation/robot/joint_position"][:-1],
            ],
            axis=0,
        )[:, :8]
        np.testing.assert_array_equal(states, pre)
        np.testing.assert_array_equal(actions, g["actions"][:])
    n = len(states)
    np.testing.assert_array_equal(table["frame_index"], np.arange(n))
    np.testing.assert_array_equal(table["index"], np.arange(n))
    np.testing.assert_allclose(table["timestamp"], np.arange(n) / 50, atol=1e-6, rtol=0)
    ds = data_loader.create_torch_dataset(dc, 50, cfg.model)
    assert len(ds) == n
    videos = {}
    for name in ["exterior_image_1_left", "exterior_image_2_left", "wrist_image_left"]:
        with imageio.get_reader(root / f"videos/chunk-000/observation.images.{name}/episode_000000.mp4") as reader:
            assert reader.get_meta_data()["fps"] == 50
            videos[name] = np.stack(list(reader.iter_data()))
        assert videos[name].shape == (n, 320, 320, 3)
    before = transforms.compose([*dc.repack_transforms.inputs, *dc.data_transforms.inputs])
    normalizer = transforms.Normalize(dc.norm_stats, use_quantiles=dc.use_quantile_norm)
    after = transforms.compose(
        [transforms.Unnormalize(dc.norm_stats, use_quantiles=dc.use_quantile_norm), *dc.data_transforms.outputs]
    )
    max_pixel_error = max_roundtrip = max_tokens = 0
    raw_loader = torch.utils.data.DataLoader(
        ds, batch_size=None, num_workers=args.workers, **({"multiprocessing_context": "spawn"} if args.workers else {})
    )
    for i, raw in enumerate(raw_loader):
        if i % 100 == 0:
            print(f"Audit {i}/{n}", flush=True)
        np.testing.assert_array_equal(raw["observation.state"].numpy(), states[i])
        gt = actions[np.minimum(i + np.arange(50), n - 1)]
        np.testing.assert_array_equal(raw["action"].numpy(), gt)
        assert raw["prompt"] == "Pick up the coke can and place it in the terracotta dish."
        for name, frames in videos.items():
            error = float(np.abs(_parse_image(raw["observation.images." + name]).astype(float) - frames[i]).max())
            assert error <= 1, (i, name, error)
            max_pixel_error = max(max_pixel_error, error)
        converted = before(copy.deepcopy(raw))
        expected = gt.copy()
        expected[:, :7] -= states[i, None, :7]
        np.testing.assert_allclose(converted["actions"], expected, atol=1e-6, rtol=0)
        restored = after(normalizer(copy.deepcopy(converted)))["actions"]
        error = float(np.abs(restored - gt).max())
        assert error < 1e-5, (i, error)
        max_roundtrip = max(max_roundtrip, error)
        sample = data_loader.transform_dataset([copy.deepcopy(raw)], dc)[0]
        assert sample["actions"].shape == (50, 32)
        assert sample["state"].shape == (32,)
        assert np.isfinite(sample["actions"]).all()
        assert np.isfinite(sample["state"]).all()
        assert not sample["image_mask"]["right_wrist_0_rgb"]
        nt = int(sample["tokenized_prompt_mask"].sum())
        max_tokens = max(max_tokens, nt)
        assert nt < cfg.model.max_token_len, "Prompt truncation"
    assert i == n - 1
    running = {k: normalize.RunningStats() for k in ["state", "actions"]}
    for start in range(0, n, cfg.batch_size):
        rows = np.arange(start, min(start + cfg.batch_size, n))
        chunks = actions[np.minimum(rows[:, None] + np.arange(50), n - 1)].copy()
        chunks[:, :, :7] -= states[rows, None, :7]
        running["state"].update(states[rows])
        running["actions"].update(chunks)
    for key, accumulator in running.items():
        expected = accumulator.get_statistics()
        for field in ["mean", "std", "q01", "q99"]:
            np.testing.assert_array_equal(getattr(expected, field), getattr(dc.norm_stats[key], field))
        assert np.all(dc.norm_stats[key].q99 - dc.norm_stats[key].q01 > 1e-6), "Degenerate normalization"
    for obs, acts in data_loader.create_data_loader(cfg, shuffle=True, num_batches=2):
        assert obs.state.shape == (32, 32)
        assert acts.shape == (32, 50, 32)
        assert np.isfinite(np.asarray(acts)).all()
    result = {
        "status": "PASS",
        "frames_checked": n,
        "future_actions_per_frame": 50,
        "camera_frames_checked": n * 3,
        "source_action_state_match": "exact",
        "norm_recomputation": "exact",
        "max_loader_pixel_error": max_pixel_error,
        "max_action_roundtrip_error": max_roundtrip,
        "max_prompt_tokens": max_tokens,
        "max_token_len": cfg.model.max_token_len,
        "training_batch_shape": [32, 50, 32],
        "training_batches_checked": 2,
        "configured_workers": cfg.num_workers,
        "gripper_transition_frames": (np.flatnonzero(np.diff(actions[:, 7])) + 1).tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
