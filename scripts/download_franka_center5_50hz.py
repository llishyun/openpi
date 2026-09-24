"""Download and verify a 50 Hz Franka dataset (experiment picked by OPENPI_FRANKA_CONFIG) using the logged-in HF account."""

import hashlib
import json
import os
from pathlib import Path
import tempfile

from huggingface_hub import snapshot_download
from lerobot.common.constants import HF_LEROBOT_HOME

EXPERIMENTS = {
    "pi05_franka_pnp_center5_v2_50hz": "franka_center5_50hz",
    "pi05_franka_pnp_mid10x15_50hz": "franka_mid10x15_50hz",
}


def main():
    project = Path(__file__).resolve().parents[1]
    experiment_dir = project / "examples" / EXPERIMENTS[os.environ.get("OPENPI_FRANKA_CONFIG", "pi05_franka_pnp_center5_v2_50hz")]
    experiment = json.loads((experiment_dir / "experiment.json").read_text())
    manifest = json.loads((experiment_dir / "dataset_sha256.json").read_text())
    root = HF_LEROBOT_HOME / experiment["dataset"]

    def verify(directory):
        for name, expected in manifest.items():
            path = directory / name
            assert path.is_file(), f"Missing dataset file: {path}"
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, f"Changed dataset file: {path}"

    if root.exists():
        verify(root)
    else:
        root.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".center5-50hz-", dir=root.parent) as temporary:
            staging = Path(temporary) / "dataset"
            snapshot_download(
                repo_id=experiment["dataset"], repo_type="dataset", local_dir=staging, allow_patterns=list(manifest)
            )
            verify(staging)
            staging.rename(root)
    print(f"Verified {experiment['episodes']} episodes at 50 Hz: {root}")


if __name__ == "__main__":
    main()
