"""Audit all 50 episodes and videos, plus loader samples at episode boundaries."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import json
from pathlib import Path

import av
import h5py
from lerobot.common.constants import HF_LEROBOT_HOME
import numpy as np
import pyarrow.parquet as pq
import torch

from openpi import transforms
from openpi.shared import normalize
from openpi.training import config
from openpi.training import data_loader


def check_video(job):
    path, length = job
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        assert float(stream.average_rate) == 50
        count = 0
        for i, frame in enumerate(container.decode(stream)):
            assert (frame.width, frame.height) == (320, 320)
            assert abs(float(frame.pts * frame.time_base) - i / 50) < 1e-6
            count += 1
        assert count == length, (path, count, length)
    return count


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("diagnostics/center5_v2_50hz/audit.json"))
    args = p.parse_args()
    torch.set_num_threads(1)
    cfg = config.get_config("pi05_franka_pnp_center5_v2_50hz")
    root = HF_LEROBOT_HOME / cfg.data.repo_id
    info = json.loads((root / "meta/info.json").read_text())
    assert (info["fps"], info["total_episodes"], info["total_frames"]) == (50, 50, 69898)
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    ds = data_loader.create_torch_dataset(dc, 50, cfg.model)
    assert len(ds) == info["total_frames"]
    before = transforms.compose([*dc.repack_transforms.inputs, *dc.data_transforms.inputs])
    normalizer = transforms.Normalize(dc.norm_stats, use_quantiles=dc.use_quantile_norm)
    after = transforms.compose(
        [transforms.Unnormalize(dc.norm_stats, use_quantiles=dc.use_quantile_norm), *dc.data_transforms.outputs]
    )
    running = {k: normalize.RunningStats() for k in ["state", "actions"]}
    offset = checked = 0
    max_roundtrip = 0.0
    video_jobs = []
    lengths = []
    for ep in range(50):
        table = pq.read_table(root / f"data/chunk-000/episode_{ep:06d}.parquet").to_pydict()
        state, action = [np.asarray(table[k], np.float32) for k in ["observation.state", "action"]]
        n = len(state)
        lengths.append(n)
        with h5py.File(root / f"references/episode_{ep:06d}.hdf5") as f:
            g = f["data/demo_0"]
            assert g.attrs["success"]
            post = g["states/articulation/robot/joint_position"][:]
            initial = g["initial_state/articulation/robot/joint_position"][:].reshape(1, -1)
            np.testing.assert_array_equal(state, np.concatenate([initial, post[:-1]])[:, :8])
            np.testing.assert_array_equal(action, g["actions"][:])
        assert np.isfinite(state).all()
        assert np.isfinite(action).all()
        np.testing.assert_array_equal(table["index"], np.arange(offset, offset + n))
        np.testing.assert_array_equal(table["frame_index"], np.arange(n))
        np.testing.assert_array_equal(table["episode_index"], np.full(n, ep))
        np.testing.assert_allclose(table["timestamp"], np.arange(n) / 50, atol=2e-6, rtol=0)
        transitions = (np.flatnonzero(np.diff(action[:, 7])) + 1).tolist()
        samples = sorted({0, 1, n // 2, n - 51, n - 50, n - 2, n - 1, *transitions, *(i - 1 for i in transitions)})
        for i in samples:
            raw = ds[offset + i]
            np.testing.assert_array_equal(np.asarray(raw["observation.state"]), state[i])
            expected = action[np.minimum(i + np.arange(50), n - 1)]
            np.testing.assert_array_equal(np.asarray(raw["action"]), expected)
            converted = before(copy.deepcopy(raw))
            delta = expected.copy()
            delta[:, :7] -= state[i, None, :7]
            np.testing.assert_allclose(converted["actions"], delta, atol=1e-6, rtol=0)
            restored = after(normalizer(copy.deepcopy(converted)))["actions"]
            error = float(np.abs(restored - expected).max())
            assert error < 1e-5
            max_roundtrip = max(max_roundtrip, error)
            transformed = data_loader.transform_dataset([copy.deepcopy(raw)], dc)[0]
            assert transformed["actions"].shape == (50, 32)
            assert np.isfinite(transformed["actions"]).all()
            assert int(transformed["tokenized_prompt_mask"].sum()) < cfg.model.max_token_len
            checked += 1
        for start in range(0, n, cfg.batch_size):
            rows = np.arange(start, min(start + cfg.batch_size, n))
            chunk = action[np.minimum(rows[:, None] + np.arange(50), n - 1)].copy()
            chunk[..., :7] -= state[rows, None, :7]
            running["state"].update(state[rows])
            running["actions"].update(chunk)
        for key, feature in info["features"].items():
            if feature["dtype"] == "video":
                video_jobs.append((root / f"videos/chunk-000/{key}/episode_{ep:06d}.mp4", n))
        offset += n
        print(f"Audit episode {ep + 1}/50: all {n} state/action rows exact", flush=True)
    assert offset == 69898
    for key, accumulator in running.items():
        expected = accumulator.get_statistics()
        for field in ["mean", "std", "q01", "q99"]:
            np.testing.assert_array_equal(getattr(expected, field), getattr(dc.norm_stats[key], field))
        assert np.all(dc.norm_stats[key].q99 - dc.norm_stats[key].q01 > 1e-6)
    with ProcessPoolExecutor(max_workers=4) as pool:
        video_frames = sum(pool.map(check_video, video_jobs))
    assert video_frames == 3 * offset
    for obs, acts in data_loader.create_data_loader(cfg, shuffle=True, num_batches=2):
        assert obs.state.shape == (32, 32)
        assert acts.shape == (32, 50, 32)
        assert np.isfinite(np.asarray(acts)).all()
    result = {
        "status": "PASS",
        "episodes": 50,
        "frames": offset,
        "fps": 50,
        "state_action_reference_match": "exact",
        "norm_recomputation": "exact",
        "videos_decoded": len(video_jobs),
        "video_frames_and_timestamps_checked": video_frames,
        "loader_samples_checked": checked,
        "episode_boundary_padding": "PASS",
        "max_action_roundtrip_error": max_roundtrip,
        "training_batch_shape": [32, 50, 32],
        "training_batches_checked": 2,
        "length_range": [min(lengths), max(lengths)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
