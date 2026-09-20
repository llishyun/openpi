#!/usr/bin/env bash
set -euo pipefail
# Start the 50 Hz checkpoint server first; simulator lives in the existing Docker project.
pkg=isaaclab_arena_pnp_franka
dataset="$pkg/demo/lerobot/franka_pnp_overfit1_fix3_50hz"
sim_gpu=${SIM_GPU:-6}
port=${POLICY_PORT:-8000}
run_tag=${RUN_TAG:-overfit1_fix3_50hz}
replan=${REPLAN_EVERY:-25}
docker exec -w /workspaces/isaaclab_arena isaaclab_arena-latest \
  env CUDA_VISIBLE_DEVICES="$sim_gpu" /isaac-sim/python.sh "$pkg/Tools/eval/pi05_eval.py" \
  --headless --enable_cameras --num_episodes "${EPISODES:-5}" --video \
  --out_dir "$pkg/Tools/_work/eval/$run_tag" --run_name base_big_r101_c00 \
  --remote_host 127.0.0.1 --remote_port "$port" \
  --expected_action_fps 50 --control_stride 1 --replan_every "$replan" --max_steps 3000 \
  --target_pos_noise 0 --initial_state_hdf5 "$dataset/reference.hdf5" --friction 0.9 \
  --external_environment_class_path "$pkg.environments.environment:PnPObstaclesEnvironment" \
  isaaclab_arena_pnp_obstacles --task pick_place \
  --layout "$dataset/scenes/base_big_r101_c00/layout.yaml"
