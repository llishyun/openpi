#!/usr/bin/env python
"""Closed-loop evaluation of a fine-tuned openpi pi05 checkpoint on one pnp layout.

Runs INSIDE the Isaac Sim container (same env code path as Tools/datagen/datagen_runner.py,
which produced the training data), talks to an openpi ``serve_policy.py`` server over
websocket, and scores episodes with the pick_place task's termination terms.

Observation wire format = what openpi/policies/franka_pnp_policy.py expects (the server
applies the training-time transforms: 224 resize, normalisation, delta -> absolute actions):
    observation/exterior_image_1_left  (320, 320, 3) uint8   obs["camera_obs"]["scene_cam_rgb"]
    observation/wrist_image_left       (320, 320, 3) uint8   obs["camera_obs"]["wrist_cam_rgb"]
    observation/state                  (8,) float32          panda_joint1..7 + panda_finger_joint1, absolute
    prompt                             str                   layout.instruction
Server reply ``actions``: (10, 8) absolute joint targets (7) + gripper command (+1 open / -1 close),
one row per 0.1 s (the dataset is 50 Hz recording subsampled x5 -> 10 fps).

Timing: one policy action is held for ``--control_stride`` sim steps (5 = 10 Hz at the 50 Hz
env rate); after ``--replan_every`` actions the remainder of the chunk is discarded and a new
chunk is fetched from the current observation.

Usage (inside the container, repo mounted at /workspaces/isaaclab_arena):
    /isaac-sim/python.sh isaaclab_arena_pnp_franka/Tools/eval/pi05_eval.py --headless --enable_cameras \
        --num_episodes 5 --out_dir /tmp/pnp_eval/base --run_name eval_big100_s000 --video \
        --remote_host 127.0.0.1 --remote_port 8000 \
        --external_environment_class_path isaaclab_arena_pnp_franka.environments.environment:PnPObstaclesEnvironment \
        isaaclab_arena_pnp_obstacles --task pick_place \
        --layout isaaclab_arena_pnp_franka/demo/lerobot/franka_pnp_big100_base/eval/scenes/eval_big100_s000/layout.yaml
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

sys.path.insert(0, "/workspaces/isaaclab_arena/isaaclab_arena_pnp_franka/Tools/eval")

# panda_finger_joint1: 0.04 m fully open, ~0.033 m when pinching the 6.6 cm can.
_FINGER_OPEN_M = 0.04


@dataclass
class EpisodeResult:
    episode_index: int
    success: bool
    reason: str
    ever_grasped: bool
    placed: bool
    dropped: bool
    tipped: bool
    timed_out: bool
    grasp_missed: bool
    steps: int
    fetches: int
    wall_s: float
    video: str | None = None


def _classify(ever_grasped: bool, placed: bool, dropped: bool, tipped: bool, timed_out: bool,
              grasp_missed: bool = False) -> tuple[bool, str]:
    if placed:
        return True, "success"
    if tipped:
        return False, "target_tipped"
    if grasp_missed and not ever_grasped:
        return False, "grasp_missed"   # task term: target displaced while not held (Panda only)
    if dropped:
        return False, "grasped_then_dropped" if ever_grasped else "knocked_off_desk"
    if ever_grasped:
        return False, "grasped_not_placed"
    if timed_out:
        return False, "never_grasped_timeout"
    return False, "never_grasped"


def _summary(results: list[EpisodeResult]) -> dict[str, Any]:
    n = len(results)
    ns = sum(r.success for r in results)
    reasons: dict[str, int] = {}
    for r in results:
        reasons[r.reason] = reasons.get(r.reason, 0) + 1
    return {"num_episodes": n, "num_success": ns, "success_rate": (ns / n if n else 0.0),
            "reason_counts": dict(sorted(reasons.items(), key=lambda kv: -kv[1]))}


def main() -> int:
    ap = get_isaaclab_arena_cli_parser()
    ap.add_argument("--num_episodes", type=int, default=5)
    ap.add_argument("--out_dir", type=str, default="/tmp/pnp_eval")
    ap.add_argument("--run_name", type=str, default="eval")
    ap.add_argument("--remote_host", type=str, default="127.0.0.1")
    ap.add_argument("--remote_port", type=int, default=8000)
    ap.add_argument("--prompt", type=str, default=None, help="override the layout instruction")
    ap.add_argument("--control_stride", type=int, default=5, help="sim steps per policy action (5 = 10 Hz)")
    ap.add_argument("--replan_every", type=int, default=5, help="actions executed per fetched chunk (<= horizon 10)")
    ap.add_argument("--max_joint_delta", type=float, default=0.0,
                    help="if > 0, clamp each arm target to +-this (rad) around the CURRENT joint position")
    ap.add_argument("--interp", action="store_true",
                    help="linearly interpolate arm targets between consecutive 10 fps actions over the control_stride "
                         "sim steps (mimics the expert's smooth 50 Hz ramp instead of stair-step targets)")
    ap.add_argument("--teacher_hdf5", type=str, default=None,
                    help="diagnostic: demo *_succ.hdf5 whose recorded 50 Hz actions are replayed for the first "
                         "--teacher_steps sim steps of every episode before handing control to the policy")
    ap.add_argument("--teacher_steps", type=int, default=0, help="sim steps driven by the recorded demo (0 = off)")
    ap.add_argument("--gripper_debounce", type=int, default=1,
                    help="consecutive actions the gripper sign must agree before the command flips (1 = off)")
    ap.add_argument("--max_steps", type=int, default=3000, help="per-episode sim-step cap (3000 = 60 s); counts as time_out")
    ap.add_argument("--target_pos_noise", type=float, default=0.0, help="uniform +-m xy perturbation of the target at reset")
    ap.add_argument("--init_arm_q", type=str, default="dataset",
                    help="episode-start arm pose: 'dataset' (the demos' start pose), 'layout' (env default), or 7 comma floats")
    ap.add_argument("--initial_state_hdf5", type=str, default=None,
                    help="restore a single demo's full initial state and render cameras without physics warmup")
    ap.add_argument("--gripper_closed_threshold", type=float, default=0.038, help="finger joint (m) below this = closed")
    ap.add_argument("--grasp_proximity", type=float, default=0.12, help="target-to-hand distance (m) for 'held'")
    ap.add_argument("--grasp_lift_margin", type=float, default=0.03, help="target lift above desk (m) for 'held'")
    ap.add_argument("--friction", type=float, default=0.9,
                    help="global sim static friction (dynamic = 0.8x). 0.9 = the value the demos were collected with "
                         "(evalenv.DEMO_FRICTION); 0 = keep the IsaacLab default 0.5")
    ap.add_argument("--video", action="store_true", help="write one mp4 per episode (scene + wrist side by side)")
    ap.add_argument("--video_stride", type=int, default=2, help="record every Nth sim step (2 -> 25 fps real time)")
    args_cli, _ = ap.parse_known_args()

    with SimulationAppContext(args_cli):
        import evalenv
        evalenv.install_lightwheel_fallback("eval")
        import numpy as np
        import torch
        from isaaclab_arena_environments.cli import get_isaaclab_arena_environments_cli_parser
        from openpi_client import websocket_client_policy

        from isaaclab_arena_pnp_franka.layout import Layout

        args_cli = get_isaaclab_arena_environments_cli_parser(ap).parse_args()
        os.makedirs(args_cli.out_dir, exist_ok=True)
        assert 1 <= args_cli.replan_every <= 10, "--replan_every must be in 1..10 (chunk horizon is 10)"

        lay = Layout.load(args_cli.layout)
        prompt = args_cli.prompt or lay.instruction
        print(f"[eval] layout={args_cli.layout}\n[eval] prompt={prompt!r}", flush=True)

        env = evalenv.make_env(args_cli, args_cli.out_dir, args_cli.run_name,
                               init_arm_q=evalenv.parse_init_arm_q(args_cli.init_arm_q))
        base_env = env.unwrapped
        assert base_env.num_envs == 1, "pi05_eval.py drives one env (one websocket request per step)"
        dev = base_env.device
        rig = evalenv.PandaRig(base_env)
        initial_state = None
        if args_cli.initial_state_hdf5:
            assert args_cli.target_pos_noise == 0, "Exact demo start requires zero target noise"
            initial_state = evalenv.load_demo_initial_state(args_cli.initial_state_hdf5, dev)

        def _pack(obs) -> dict[str, Any]:
            ext, wri = evalenv.cam_pair(obs)
            return {"observation/exterior_image_1_left": ext, "observation/wrist_image_left": wri,
                    "observation/state": rig.state8(), "prompt": prompt}

        teacher = None
        if args_cli.teacher_hdf5 and args_cli.teacher_steps > 0:
            import h5py
            with h5py.File(args_cli.teacher_hdf5, "r") as f:
                g = f["data"][list(f["data"].keys())[0]]
                reference_actions = np.asarray(g["actions"], dtype=np.float32)
            import pyarrow.parquet as pq
            from pathlib import Path
            dataset_root = Path(args_cli.teacher_hdf5).parent
            info = json.loads((dataset_root / "meta/info.json").read_text())
            assert info["fps"] == 50 and info["total_frames"] == 1291
            table = pq.read_table(dataset_root / "data/chunk-000/episode_000000.parquet")
            teacher = np.array(table["action"].to_pylist(), dtype=np.float32)
            np.testing.assert_array_equal(teacher, reference_actions)
            assert args_cli.teacher_steps % args_cli.control_stride == 0, "--teacher_steps must be a multiple of --control_stride"
            print(f"[eval] teacher forcing: first {args_cli.teacher_steps} steps ({args_cli.teacher_steps/50:.1f} s) replay {args_cli.teacher_hdf5}", flush=True)

        # --- policy client ---
        print(f"[eval] connecting to openpi server {args_cli.remote_host}:{args_cli.remote_port} ...", flush=True)
        class ReplayClient:
            def get_server_metadata(self): return {"mode": "dataset_actions", "action_fps": 50}
            def reset(self): pass
        client = ReplayClient()
        print(f"[eval] server metadata: {client.get_server_metadata()}", flush=True)

        if args_cli.video:
            import imageio.v2 as imageio

        results: list[EpisodeResult] = []
        def _episode_start(obs):
            """Restore an exact demo start when requested; otherwise use the legacy warmup."""
            if initial_state is not None:
                return evalenv.restore_demo_start(base_env, rig, initial_state)
            rig.perturb_target(args_cli.target_pos_noise)
            rig.pin_wrist_cam()
            for _ in range(2):
                obs, *_ = env.step(rig.hold_action())
                rig.pin_wrist_cam()
            return obs

        obs, _ = env.reset()
        obs = _episode_start(obs)

        while len(results) < args_cli.num_episodes:
            ep = len(results)
            mode = ["dataset50_run1", "dataset50_run2"][ep]
            args_cli.replan_every = 1 if mode == "policy1" else 5
            args_cli.teacher_steps = 100000
            trace_state, trace_action, trace_target, trace_prediction, trace_query_step = [], [], [], [], []
            assert abs(base_env.step_dt-0.02)<1e-8
            print("[audit] mode="+mode, flush=True)
            t0 = time.time()
            chunk: list[np.ndarray] = []
            fetches = steps = 0
            ever_grasped = False
            grip_cmd, grip_pending, grip_count = 1.0, 1.0, 0
            frames: list[np.ndarray] = []
            action = None
            prev_arm = None      # last commanded arm target (for --interp)
            done = False
            placed = dropped = tipped = timed_out = grasp_missed = False
            with torch.inference_mode():
                while not done:
                    if teacher is not None and steps < args_cli.teacher_steps:
                        # teacher-forced phase: recorded expert action for this sim step, no policy query
                        teacher_index = min((steps // 5) * 5 if mode == "expert10" else steps, len(teacher) - 1)
                        action = torch.tensor(teacher[teacher_index], device=dev).unsqueeze(0)
                        prev_arm = teacher[min(steps, len(teacher) - 1)][:7].copy()
                    elif steps % args_cli.control_stride == 0:
                        if not chunk:
                            out = client.infer(_pack(obs))
                            acts = np.asarray(out["actions"], dtype=np.float32)
                            chunk = [a for a in acts[: args_cli.replan_every]]
                            fetches += 1
                            trace_prediction.append(acts.copy())
                            trace_query_step.append(steps)
                            if steps == 0:
                                np.savez_compressed(os.path.join(args_cli.out_dir, mode+"_initial.npz"), state=rig.state8(), exterior=evalenv.cam_pair(obs)[0], wrist=evalenv.cam_pair(obs)[1])
                        a = chunk.pop(0)
                        arm = a[:7].copy()
                        if args_cli.max_joint_delta > 0:
                            cur = rig.state8()[:7]
                            arm = np.clip(arm, cur - args_cli.max_joint_delta, cur + args_cli.max_joint_delta)
                        g = 1.0 if a[7] >= 0 else -1.0
                        if args_cli.gripper_debounce > 1:
                            if g == grip_pending:
                                grip_count += 1
                            else:
                                grip_pending, grip_count = g, 1
                            if grip_count >= args_cli.gripper_debounce:
                                grip_cmd = grip_pending
                        else:
                            grip_cmd = g
                        if args_cli.interp:
                            interp_from = prev_arm if prev_arm is not None else rig.state8()[:7]
                            interp_to, prev_arm = arm.copy(), arm.copy()
                        action = torch.tensor(np.concatenate([arm, [grip_cmd]]), device=dev, dtype=torch.float32).unsqueeze(0)
                    if args_cli.interp and not (teacher is not None and steps < args_cli.teacher_steps):
                        frac = ((steps % args_cli.control_stride) + 1) / args_cli.control_stride
                        arm_i = interp_from + frac * (interp_to - interp_from)
                        action = torch.tensor(np.concatenate([arm_i, [grip_cmd]]), device=dev, dtype=torch.float32).unsqueeze(0)
                    trace_state.append(rig.state8().copy())
                    trace_action.append(action[0].detach().cpu().numpy().copy())
                    trace_target.append(rig._as_torch(rig.target.data.root_state_w)[0].detach().cpu().numpy().copy())
                    obs, _, terminated, truncated, _ = env.step(action)
                    rig.pin_wrist_cam()
                    steps += 1
                    ever_grasped |= rig.grasp_active(args_cli.gripper_closed_threshold, args_cli.grasp_proximity, args_cli.grasp_lift_margin)
                    if args_cli.video and steps % args_cli.video_stride == 0:
                        frames.append(np.concatenate(evalenv.cam_pair(obs), axis=1))
                    if bool(terminated[0]) or bool(truncated[0]):
                        tm = base_env.termination_manager
                        placed, dropped = evalenv.term(tm, "success"), evalenv.term(tm, "object_dropped")
                        tipped, timed_out = evalenv.term(tm, "object_tipped"), evalenv.term(tm, "time_out")
                        grasp_missed = evalenv.term(tm, "grasp_missed")
                        done = True
                    elif steps >= args_cli.max_steps:
                        timed_out, done = True, True
                        obs, _ = env.reset()
            np.savez_compressed(os.path.join(args_cli.out_dir, mode+"_trace.npz"),state=trace_state,action=trace_action,target=trace_target,prediction=trace_prediction,query_step=trace_query_step)
            success, reason = _classify(ever_grasped, placed, dropped, tipped, timed_out, grasp_missed)
            video_path = None
            if args_cli.video and frames:
                video_path = os.path.join(args_cli.out_dir, f"{args_cli.run_name}_ep{ep:02d}_{'succ' if success else 'fail'}.mp4")
                w = imageio.get_writer(video_path, format="ffmpeg", fps=50.0 / args_cli.video_stride, codec="libx264",
                                       quality=6, pixelformat="yuv420p", macro_block_size=None, ffmpeg_log_level="error")
                for f in frames:
                    w.append_data(f)
                w.close()
            results.append(EpisodeResult(ep, success, reason, ever_grasped, placed, dropped, tipped, timed_out,
                                         grasp_missed, steps, fetches, round(time.time() - t0, 1), video_path))
            print(f"[eval] ep {ep + 1}/{args_cli.num_episodes} {'SUCCESS' if success else 'FAIL'}:{reason} "
                  f"steps={steps} fetches={fetches} wall={results[-1].wall_s}s", flush=True)
            # env auto-resets on termination; make the new episode start like the first one
            obs = _episode_start(obs)
            client.reset()

        summary = _summary(results)
        out = {"layout": args_cli.layout, "prompt": prompt, "run_name": args_cli.run_name,
               "config": {k: getattr(args_cli, k) for k in ("control_stride", "replan_every", "max_joint_delta",
                          "gripper_debounce", "interp", "teacher_hdf5", "teacher_steps", "max_steps", "target_pos_noise", "init_arm_q", "initial_state_hdf5", "gripper_closed_threshold",
                          "grasp_proximity", "grasp_lift_margin", "remote_host", "remote_port")},
               **summary, "episodes": [asdict(r) for r in results]}
        path = os.path.join(args_cli.out_dir, f"{args_cli.run_name}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
        print(f"[eval] DONE {args_cli.run_name}: {summary['num_success']}/{summary['num_episodes']} "
              f"{summary['reason_counts']} -> {path}", flush=True)
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
