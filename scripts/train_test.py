import dataclasses
import os
import pathlib

import pytest

os.environ["JAX_PLATFORMS"] = "cpu"

from openpi.training import config as _config

from . import train


@pytest.mark.parametrize("config_name", ["debug"])
def test_train(tmp_path: pathlib.Path, config_name: str):
    config = dataclasses.replace(
        _config._CONFIGS_DICT[config_name],  # noqa: SLF001
        batch_size=2,
        checkpoint_base_dir=str(tmp_path / "checkpoint"),
        exp_name="test",
        overwrite=False,
        resume=False,
        num_train_steps=2,
        log_interval=1,
        save_interval=1,
        keep_period=None,
        inference_save_steps=(1,),
    )
    train.main(config)

    # test resuming
    config = dataclasses.replace(config, resume=True, num_train_steps=4, inference_save_steps=(1, 3))
    train.main(config)
    exports = config.checkpoint_dir.parent / "test_inference"
    for step in (1, 3):
        assert (exports / str(step) / "params").is_dir()
        assert (exports / str(step) / "assets").is_dir()
        assert not (exports / str(step) / "train_state").exists()
    assert (config.checkpoint_dir / "3").is_dir()
    assert not (config.checkpoint_dir / "1").exists()
