"""Build all frames of the audited fix3 demo. Run with Isaac Sim's Python in Docker."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--template", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    from isaaclab_arena_pnp_franka.Tools.datagen.big100_to_lerobot import CAMERA_MAP
    from isaaclab_arena_pnp_franka.Tools.datagen.big100_to_lerobot import _img_stats
    from isaaclab_arena_pnp_franka.Tools.datagen.big100_to_lerobot import _stats
    from isaaclab_arena_pnp_franka.Tools.datagen.big100_to_lerobot import _write_video
    from isaaclab_arena_pnp_franka.Tools.datagen.hdf5_alignment import pre_action_joint_positions

    out = args.output
    (out / "meta").mkdir(parents=True)
    shutil.copytree(args.template / "scenes", out / "scenes")
    for name in ["tasks.jsonl", "modality.json"]:
        shutil.copy2(args.template / "meta" / name, out / "meta" / name)
    with h5py.File(args.source, "r") as f:
        assert len(f["data"]) == 1
        g = f["data"][next(iter(f["data"]))]
        assert bool(g.attrs["success"])
        action = np.asarray(g["actions"], np.float32)
        state = pre_action_joint_positions(g)[:, :8].astype(np.float32)
        n = len(action)
        assert state.shape == action.shape == (1291, 8)
        assert np.isfinite(state).all()
        assert np.isfinite(action).all()
        assert set(np.unique(action[:, 7])) == {-1.0, 1.0}
        stats = {"observation.state": _stats(state), "action": _stats(action)}
        for camera, key in CAMERA_MAP.items():
            frames = np.asarray(g["camera_obs/" + camera])
            assert frames.shape == (n, 320, 320, 3)
            assert frames.dtype == np.uint8
            _write_video(out / f"videos/chunk-000/{key}/episode_000000.mp4", frames, 50)
            stats[key] = _img_stats(frames)
        with h5py.File(out / "reference.hdf5", "w") as ref:
            target = ref.create_group("data/demo_0")
            target.attrs["success"] = True
            target.attrs["num_samples"] = n
            for key in ["initial_state", "actions", "states", "success"]:
                g.copy(key, target)
    columns = {
        "observation.state": pa.array(list(state), type=pa.list_(pa.float32(), 8)),
        "action": pa.array(list(action), type=pa.list_(pa.float32(), 8)),
        "timestamp": pa.array((np.arange(n) / 50).astype(np.float32)),
        "frame_index": pa.array(np.arange(n, dtype=np.int64)),
        "episode_index": pa.array(np.zeros(n, dtype=np.int64)),
        "index": pa.array(np.arange(n, dtype=np.int64)),
        "task_index": pa.array(np.zeros(n, dtype=np.int64)),
    }
    table = pa.table(columns)
    (out / "data/chunk-000").mkdir(parents=True)
    pq.write_table(table, out / "data/chunk-000/episode_000000.parquet")
    for key in ["timestamp", "frame_index", "episode_index", "index", "task_index"]:
        stats[key] = _stats(table[key].to_numpy()[:, None])
    info = json.loads((args.template / "meta/info.json").read_text())
    info.update(total_frames=n, fps=50)
    for key in CAMERA_MAP.values():
        info["features"][key]["info"]["video.fps"] = 50

    def write(name, value):
        (out / "meta" / name).write_text(json.dumps(value, indent=2) + "\n")

    write("info.json", info)
    write("stats.json", {k: stats[k] for k in ["observation.state", "action"]})
    task = json.loads((out / "meta/tasks.jsonl").read_text())["task"]
    (out / "meta/episodes.jsonl").write_text(json.dumps({"episode_index": 0, "tasks": [task], "length": n}) + "\n")
    (out / "meta/episodes_stats.jsonl").write_text(json.dumps({"episode_index": 0, "stats": stats}) + "\n")
    scene = json.loads((args.template / "meta/scenes.jsonl").read_text())
    scene.update(suite="fix3_50hz", frames=n, frames_raw_50hz=n)
    (out / "meta/scenes.jsonl").write_text(json.dumps(scene) + "\n")
    collection = json.loads((args.template / "meta/collection.json").read_text())
    collection.update(suite="fix3_50hz", subsample=1, dataset_fps=50, recording_hz=50, final_frame_retained=True)
    write("collection.json", collection)
    source = {
        "source_hdf5": args.source.name,
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "frames": n,
        "fps": 50,
        "gripper_transition_frames": (np.flatnonzero(np.diff(action[:, 7])) + 1).tolist(),
        "state_timing": "initial_state then previous post-step state; all frames retained",
    }
    write("source.json", source)
    print(json.dumps(source, indent=2))


if __name__ == "__main__":
    main()
