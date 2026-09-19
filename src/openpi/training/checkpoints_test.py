import json

import numpy as np

from openpi.training import checkpoints


def test_inference_export_survives_retention_and_resume(tmp_path):
    run = tmp_path / "run"
    manager, resumed = checkpoints.initialize_checkpoint_dir(run, keep_period=None, overwrite=False, resume=False)
    assert not resumed

    def save_assets(directory):
        (directory / "norm_stats.json").write_text(json.dumps({"state": "test statistics"}))

    for step in (1, 2):
        manager.save(
            step,
            {
                "params": {"params": {"weights": np.arange(4, dtype=np.float32) + step}},
                "train_state": {"step": np.array(step), "optimizer": np.ones(4)},
                "assets": save_assets,
            },
        )
        exported = checkpoints.export_inference_checkpoint(manager, step)
        assert (exported / "params").is_dir()
        assert json.loads((exported / "assets/norm_stats.json").read_text()) == {"state": "test statistics"}
        assert not (exported / "train_state").exists()
    manager.close()
    assert not (run / "1").exists()
    assert (tmp_path / "run_inference/1/params").is_dir()

    manager, resumed = checkpoints.initialize_checkpoint_dir(run, keep_period=None, overwrite=False, resume=True)
    assert resumed
    restored = manager.restore(
        2,
        items={
            "params": {"params": {"weights": np.zeros(4, dtype=np.float32)}},
            "train_state": {"step": np.array(0), "optimizer": np.zeros(4)},
        },
    )
    np.testing.assert_array_equal(restored["params"]["params"]["weights"], np.arange(4) + 2)
    assert restored["train_state"]["step"] == 2
    manager.close()
