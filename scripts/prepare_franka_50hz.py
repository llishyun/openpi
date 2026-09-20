"""Run on a login node to verify data and cache model/tokenizer before sbatch."""

import importlib
import json

from franka_50hz_preflight import verify_inputs

from openpi.shared import download


def main():
    for module in ["h5py", "pyarrow", "imageio_ffmpeg", "av"]:
        importlib.import_module(module)
    cfg, root = verify_inputs()
    base = download.maybe_download("gs://openpi-assets/checkpoints/pi05_base/params")
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    assert dc.norm_stats is not None
    assert (base / "_METADATA").exists()
    assert (base / "manifest.ocdbt").exists()
    result = {"dataset": str(root), "base_model": str(base), "action_fps": 50, "action_horizon": 50}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
