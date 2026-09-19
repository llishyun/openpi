"""Install the bundled one-demo archive, verifying every file."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    manifest = json.loads((project / "examples/franka_causal/dataset_sha256.json").read_text())
    name = "franka_pnp_overfit1_causal"
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    root = Path(os.environ.get("HF_LEROBOT_HOME", hf_home / "lerobot")) / "lithyeon"
    destination = root / name
    if destination.exists():
        raise FileExistsError(f"Dataset already exists; not overwriting: {destination}")
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".causal-install-", dir=root) as temp:
        staging = Path(temp) / name
        staging.mkdir()
        with tarfile.open(args.archive, "r:gz") as archive:
            expected = {f"{name}/{key}": key for key in manifest}
            members = archive.getmembers()
            if len(members) != len(expected) or {m.name for m in members} != set(expected):
                raise ValueError("Archive members differ from the checked-in dataset manifest.")
            for member in members:
                if not member.isfile():
                    raise ValueError(f"Expected regular file: {member.name}")
                key = expected[member.name]
                path = staging / key
                path.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, path.open("wb") as target:
                    shutil.copyfileobj(source, target)
                if hashlib.sha256(path.read_bytes()).hexdigest() != manifest[key]:
                    raise ValueError(f"Checksum mismatch: {key}")
        staging.rename(destination)
    print(f"Installed and verified {len(manifest)} files: {destination}")


if __name__ == "__main__":
    main()
