#!/bin/bash
# One-time setup on the cluster LOGIN node (needs internet). Run from the openpi checkout:
#   bash scripts/slurm/prepare.sh
# Afterwards the compute-node job (train.sbatch) works fully offline.
set -euo pipefail
cd "$(dirname "$0")/../.."

REPO_ID=${REPO_ID:-lithyeon/franka_pnp_big100_base}
export HF_LEROBOT_HOME=${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}

echo "== 1/4 python env (uv sync)"
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
GIT_LFS_SKIP_SMUDGE=1 uv sync

echo "== 2/4 pi05_base weights -> ~/.cache/openpi (≈11 GB)"
uv run python -c "import openpi.shared.download as d; print(d.maybe_download('gs://openpi-assets/checkpoints/pi05_base/params'))"

echo "== 3/4 dataset ${REPO_ID} -> ${HF_LEROBOT_HOME}/${REPO_ID} (≈540 MB)"
uv run huggingface-cli download --repo-type dataset "${REPO_ID}" --local-dir "${HF_LEROBOT_HOME}/${REPO_ID}"

echo "== 4/4 sanity check (config + norm stats + one batch on CPU)"
JAX_PLATFORMS=cpu uv run python - <<'PY'
import dataclasses, openpi.training.config as c, openpi.training.data_loader as dl
cfg = dataclasses.replace(c.get_config("pi05_franka_pnp"), num_workers=0)
loader = dl.create_data_loader(cfg, num_batches=1, shuffle=False)
obs, act = next(iter(loader))
print("OK  state", obs.state.shape, "actions", act.shape, "images", {k: v.shape for k, v in obs.images.items()})
PY
echo "prepare done"
