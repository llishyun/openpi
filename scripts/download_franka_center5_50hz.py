"""Download and verify the private 50-demo dataset using the logged-in HF account."""

import hashlib
import json
from pathlib import Path
import tempfile

from huggingface_hub import snapshot_download
from lerobot.common.constants import HF_LEROBOT_HOME


def main():
    project = Path(__file__).resolve().parents[1]
    experiment = json.loads((project / "examples/franka_center5_50hz/experiment.json").read_text())
    manifest = json.loads((project / "examples/franka_center5_50hz/dataset_sha256.json").read_text())
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
    print(f"Verified 50 episodes at 50 Hz: {root}")


if __name__ == "__main__":
    main()
