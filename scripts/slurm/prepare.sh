#!/bin/bash
# One-time setup on the cluster LOGIN node (needs internet). Run from the openpi checkout:
#   bash scripts/slurm/prepare.sh
# Afterwards the compute-node job (train.sbatch) works fully offline.
set -euo pipefail
cd "$(dirname "$0")/../.."

# Datasets to have locally (Hub download if not linked from LOCAL_DATA_DIR). Default: the cells450 set used by
# pi05_franka_pnp_cells450; add lithyeon/franka_pnp_big100_base for the older pi05_franka_pnp config.
REPO_IDS=${REPO_IDS:-lithyeon/franka_pnp_cells450_base}
CONFIG=${CONFIG:-pi05_franka_pnp_cells450}          # train config checked in step 4
export HF_LEROBOT_HOME=${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}

echo "== 1/4 python env (uv sync)"
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
# Use uv's own CPython build (ships Python.h): the cluster's system python has no dev headers and
# evdev (pulled in via lerobot -> pynput) is built from source.
export UV_PYTHON_PREFERENCE=only-managed
uv python install 3.11
GIT_LFS_SKIP_SMUDGE=1 uv sync

echo "== 2/4 pi05_base weights -> ~/.cache/openpi (≈11 GB)"
uv run python -c "import openpi.shared.download as d; print(d.maybe_download('gs://openpi-assets/checkpoints/pi05_base/params'))"

echo "== 3/4 datasets -> ${HF_LEROBOT_HOME}/lithyeon/<name>"
# Datasets already copied to the cluster (e.g. franka_pnp_big100_distract, which is not on the Hub):
#   LOCAL_DATA_DIR=/path/to/data bash scripts/slurm/prepare.sh
# every <LOCAL_DATA_DIR>/<name>/meta/info.json gets a symlink at ${HF_LEROBOT_HOME}/lithyeon/<name>.
mkdir -p "${HF_LEROBOT_HOME}/lithyeon"
if [ -n "${LOCAL_DATA_DIR:-}" ]; then
    for d in "${LOCAL_DATA_DIR}"/*/; do
        d=${d%/}; name=$(basename "$d")
        [ -f "$d/meta/info.json" ] || { echo "skip $d (no meta/info.json)"; continue; }
        ln -sfn "$(readlink -f "$d")" "${HF_LEROBOT_HOME}/lithyeon/${name}"
        echo "linked lithyeon/${name} -> $d"
    done
fi
for REPO_ID in ${REPO_IDS//,/ }; do
    if [ -f "${HF_LEROBOT_HOME}/${REPO_ID}/meta/info.json" ]; then
        echo "${REPO_ID} already present, skipping download"
    else
        echo "downloading ${REPO_ID} from the Hub"
        uv run huggingface-cli download --repo-type dataset "${REPO_ID}" --local-dir "${HF_LEROBOT_HOME}/${REPO_ID}"
    fi
done

echo "== 4/4 sanity check (${CONFIG}: config + norm stats + one batch on CPU)"
CONFIG=$CONFIG JAX_PLATFORMS=cpu uv run python - <<'PY'
import dataclasses, os, openpi.training.config as c, openpi.training.data_loader as dl
cfg = dataclasses.replace(c.get_config(os.environ.get("CONFIG", "pi05_franka_pnp_cells450")), num_workers=0)
loader = dl.create_data_loader(cfg, num_batches=1, shuffle=False)
obs, act = next(iter(loader))
print("OK  state", obs.state.shape, "actions", act.shape, "images", {k: v.shape for k, v in obs.images.items()})
PY
echo "prepare done"
