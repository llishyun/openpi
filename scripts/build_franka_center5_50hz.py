"""Convert the 50 center5_v2 successes at full rate, using Isaac Sim Python."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def convert_episode(job):
    from isaaclab_arena_pnp_franka.Tools.datagen.big100_to_lerobot import CAMERA_MAP
    from isaaclab_arena_pnp_franka.Tools.datagen.big100_to_lerobot import _img_stats
    from isaaclab_arena_pnp_franka.Tools.datagen.big100_to_lerobot import _stats
    from isaaclab_arena_pnp_franka.Tools.datagen.big100_to_lerobot import _write_video
    from isaaclab_arena_pnp_franka.Tools.datagen.hdf5_alignment import pre_action_joint_positions

    row, source, output, offset = job
    ep = row["episode_index"]
    out = Path(output)
    files = sorted((Path(source) / row["scene"]).glob("demo_c*_succ.hdf5"))
    assert len(files) == 1
    src = files[0]
    with h5py.File(src) as f:
        assert len(f["data"]) == 1
        g = f["data"][next(iter(f["data"]))]
        assert g.attrs["success"]
        action = np.asarray(g["actions"], np.float32)
        state = pre_action_joint_positions(g)[:, :8].astype(np.float32)
        n = len(action)
        assert n == row["frames_raw_50hz"]
        assert state.shape == action.shape == (n, 8)
        assert np.isfinite(state).all()
        assert np.isfinite(action).all()
        assert set(np.unique(action[:, 7])) == {-1.0, 1.0}
        stats = {"observation.state": _stats(state), "action": _stats(action)}
        for cam, key in CAMERA_MAP.items():
            frames = np.asarray(g["camera_obs/" + cam])
            assert frames.shape == (n, 320, 320, 3)
            assert frames.dtype == np.uint8
            _write_video(out / f"videos/chunk-000/{key}/episode_{ep:06d}.mp4", frames, 50)
            stats[key] = _img_stats(frames)
            del frames
        refpath = out / f"references/episode_{ep:06d}.hdf5"
        refpath.parent.mkdir(exist_ok=True)
        with h5py.File(refpath, "w") as ref:
            target = ref.create_group("data/demo_0")
            target.attrs["success"] = True
            target.attrs["num_samples"] = n
            for key in ["initial_state", "actions", "states", "success"]:
                g.copy(key, target)
    columns = {
        "observation.state": pa.array(list(state), type=pa.list_(pa.float32(), 8)),
        "action": pa.array(list(action), type=pa.list_(pa.float32(), 8)),
    }
    for key, values in {
        "timestamp": (np.arange(n) / 50).astype(np.float32),
        "frame_index": np.arange(n, dtype=np.int64),
        "episode_index": np.full(n, ep, dtype=np.int64),
        "index": np.arange(offset, offset + n, dtype=np.int64),
        "task_index": np.zeros(n, dtype=np.int64),
    }.items():
        columns[key] = pa.array(values)
        stats[key] = _stats(values[:, None])
    pq.write_table(pa.table(columns), out / f"data/chunk-000/episode_{ep:06d}.parquet")
    row = dict(row, suite="center5_v2_50hz", frames=n)
    row["files"] = dict(row["files"], reference=f"references/episode_{ep:06d}.hdf5")
    with src.open("rb") as source_file:
        source_hash = hashlib.file_digest(source_file, "sha256").hexdigest()
    source_record = {
        "episode_index": ep,
        "scene": row["scene"],
        "file": src.name,
        "sha256": source_hash,
        "frames": n,
    }
    print(f"Converted {ep + 1}/50: {n} frames", flush=True)
    return row, stats, source_record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--template", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    assert not args.output.exists(), "Refusing to overwrite an existing dataset"
    rows = [json.loads(s) for s in (args.template / "meta/scenes.jsonl").read_text().splitlines()]
    assert len(rows) == 50
    assert [r["episode_index"] for r in rows] == list(range(50))
    out = args.output
    (out / "meta").mkdir(parents=True)
    (out / "data/chunk-000").mkdir(parents=True)
    shutil.copytree(args.template / "scenes", out / "scenes")
    for name in ["tasks.jsonl", "modality.json"]:
        shutil.copy2(args.template / "meta" / name, out / "meta" / name)
    offset = 0
    jobs = []
    for row in rows:
        jobs.append((row, str(args.source), str(out), offset))
        offset += row["frames_raw_50hz"]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(convert_episode, jobs))
    for name, records in {
        "scenes.jsonl": [r[0] for r in results],
        "episodes.jsonl": [
            {"episode_index": r[0]["episode_index"], "tasks": [r[0]["instruction"]], "length": r[0]["frames"]}
            for r in results
        ],
        "episodes_stats.jsonl": [{"episode_index": r[0]["episode_index"], "stats": r[1]} for r in results],
        "sources.jsonl": [r[2] for r in results],
    }.items():
        (out / "meta" / name).write_text("".join(json.dumps(r) + "\n" for r in records))
    info = json.loads((args.template / "meta/info.json").read_text())
    info.update(fps=50, total_frames=offset, total_episodes=50, total_videos=150)
    for feature in info["features"].values():
        if feature["dtype"] == "video":
            feature["info"]["video.fps"] = 50
    (out / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")
    from isaaclab_arena_pnp_franka.Tools.datagen.big100_to_lerobot import _stats

    full = pq.read_table(out / "data/chunk-000").to_pydict()
    (out / "meta/stats.json").write_text(
        json.dumps({k: _stats(np.asarray(full[k])) for k in ["observation.state", "action"]}, indent=2) + "\n"
    )
    collection = json.loads((args.template / "meta/collection.json").read_text())
    collection.update(suite="center5_v2_50hz", subsample=1, dataset_fps=50, final_frame_retained=True)
    (out / "meta/collection.json").write_text(json.dumps(collection, indent=2) + "\n")
    print(f"COMPLETE: 50 episodes, {offset} frames, 50 Hz", flush=True)


if __name__ == "__main__":
    main()
