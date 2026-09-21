"""Two real gradient updates before a long cluster run; no checkpoint writes."""

import argparse
import dataclasses
import functools
import json
import os
from pathlib import Path
import tempfile
import time

import jax
import numpy as np
import torch
import train

from openpi.shared import download
from openpi.training import config
from openpi.training import data_loader
from openpi.training import sharding
from openpi.training import weight_loaders


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="Local pipeline test with small text/action transformers, not full model validation",
    )
    parser.add_argument(
        "--check-checkpoint", action="store_true", help="Also test save/export/restore with the local tiny model"
    )
    parser.add_argument("--config", default="pi05_franka_pnp_overfit1_fix3_50hz")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    cfg = config.get_config(args.config)
    cfg = dataclasses.replace(cfg, fsdp_devices=int(os.environ.get("GPU_COUNT", "8")))
    if args.tiny:
        cfg = dataclasses.replace(
            cfg,
            batch_size=2,
            num_workers=0,
            fsdp_devices=1,
            model=dataclasses.replace(cfg.model, paligemma_variant="dummy", action_expert_variant="dummy"),
            weight_loader=weight_loaders.NoOpWeightLoader(),
        )
    else:
        base = download.get_cache_dir() / "openpi-assets/checkpoints/pi05_base/params"
        assert base.is_dir(), "Prepare the base model before submitting"
        cfg = dataclasses.replace(cfg, weight_loader=weight_loaders.CheckpointWeightLoader(str(base)))
    mesh = sharding.make_mesh(cfg.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    loader = data_loader.create_data_loader(cfg, sharding=data_sharding, shuffle=True, num_batches=2)
    started = time.monotonic()
    state, state_sharding = train.init_train_state(cfg, jax.random.key(123), mesh, resume=False)
    jax.block_until_ready(state)
    step = jax.jit(
        functools.partial(train.train_step, cfg),
        in_shardings=(replicated, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated),
        donate_argnums=(1,),
    )
    rows = []
    for batch in loader:
        assert batch[1].shape == (cfg.batch_size, 50, 32)
        with sharding.set_mesh(mesh):
            state, metrics = step(jax.random.key(456), state, batch)
        row = {key: float(value) for key, value in jax.device_get(metrics).items()}
        assert all(np.isfinite(value) for value in row.values()), row
        assert row["grad_norm"] > 0, row
        rows.append(row)
        print(row, flush=True)
    assert len(rows) == 2
    assert int(state.step) == 2
    checkpoint_result = None
    if args.check_checkpoint:
        assert args.tiny, "Checkpoint smoke uses the small local model only"
        from openpi.training import checkpoints

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="checkpoint-smoke-", dir=args.output.parent) as temporary:
            manager, _ = checkpoints.initialize_checkpoint_dir(
                Path(temporary) / "run", keep_period=None, overwrite=False, resume=False
            )
            checkpoints.save_state(manager, state, loader, 1)
            manager.wait_until_finished()
            export = checkpoints.export_inference_checkpoint(manager, 1)
            assert (export / "params").is_dir()
            saved_stats = checkpoints.load_norm_stats(export / "assets", loader.data_config().asset_id)
            for key, value in loader.data_config().norm_stats.items():
                for field in ["mean", "std", "q01", "q99"]:
                    np.testing.assert_array_equal(getattr(saved_stats[key], field), getattr(value, field))
            restored = checkpoints.restore_state(manager, state, loader, step=1)
            assert int(restored.step) == 2
            for original, reloaded in zip(jax.tree.leaves(state.params), jax.tree.leaves(restored.params), strict=True):
                np.testing.assert_array_equal(np.asarray(original), np.asarray(reloaded))
            manager.close()
            checkpoint_result = {"save": "PASS", "export_assets": "PASS", "restore_step": 2, "restored_params": "exact"}
    result = {
        "status": "PASS",
        "checkpoint_test": checkpoint_result,
        "tiny": args.tiny,
        "steps": 2,
        "batch_size": cfg.batch_size,
        "action_horizon": 50,
        "devices": [d.device_kind for d in jax.local_devices()],
        "metrics": rows,
        "wall_seconds": time.monotonic() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
