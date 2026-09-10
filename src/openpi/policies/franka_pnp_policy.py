"""Transforms for the CVLAB Franka Panda pick-and-place LeRobot datasets (franka_pnp_big100_*).

Dataset conventions (see meta/collection.json in the dataset):
  observation.state  : 8-D absolute joint positions (panda_joint1-7 rad, panda_finger_joint1 m)
  action             : 8-D absolute joint position targets (7) + gripper command (+1 open / -1 close)
  images             : exterior_image_1_left, exterior_image_2_left, wrist_image_left (320x320 RGB)

Camera slots follow the DROID convention used by pi05: exterior_image_1_left -> base_0_rgb,
wrist_image_left -> left_wrist_0_rgb; the third slot is masked out.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_franka_pnp_example() -> dict:
    return {
        "observation/exterior_image_1_left": np.random.randint(256, size=(320, 320, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(320, 320, 3), dtype=np.uint8),
        "observation/state": np.random.rand(8),
        "prompt": "Pick up the coke can and place it in the terracotta dish.",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class FrankaPnPInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["observation/state"], dtype=np.float32)
        base_image = _parse_image(data["observation/exterior_image_1_left"])
        wrist_image = _parse_image(data["observation/wrist_image_left"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image, np.zeros_like(base_image))
                image_masks = (np.True_, np.True_, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (base_image, np.zeros_like(base_image), wrist_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt
        return inputs


@dataclasses.dataclass(frozen=True)
class FrankaPnPOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # 7 joint targets (delta -> absolute is restored by the AbsoluteActions transform) + gripper command.
        return {"actions": np.asarray(data["actions"][..., :8])}
