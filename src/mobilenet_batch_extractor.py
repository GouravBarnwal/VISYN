from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    mobilenet_v3_small,
)

IMAGE_SIZE = 224

MEAN = torch.tensor(
    [0.485, 0.456, 0.406],
    dtype=torch.float32,
).view(1, 3, 1, 1)

STD = torch.tensor(
    [0.229, 0.224, 0.225],
    dtype=torch.float32,
).view(1, 3, 1, 1)


def load_model():
    weights = MobileNet_V3_Small_Weights.DEFAULT

    model = mobilenet_v3_small(
        weights=weights,
    )

    model.eval()

    return model


def preprocess_image(image_path):
    image_path = Path(image_path)

    image = Image.open(
        image_path,
    ).convert("RGB")

    image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR,
    )

    array = (
        np.asarray(
            image,
            dtype=np.float32,
        )
        / 255.0
    )

    tensor = torch.from_numpy(
        array,
    )

    tensor = tensor.permute(
        2,
        0,
        1,
    )

    tensor = tensor.unsqueeze(0)

    tensor = (tensor - MEAN) / STD

    return tensor


def load_batch(
    image_paths,
):
    tensors = [preprocess_image(path) for path in image_paths]

    if not tensors:
        raise ValueError("image_paths cannot be empty.")

    return torch.cat(
        tensors,
        dim=0,
    )


@torch.no_grad()
def extract_batch(
    model,
    image_paths,
):
    batch = load_batch(
        image_paths,
    )

    x = batch

    l4 = None
    l8 = None

    for index, layer in enumerate(
        model.features,
    ):
        x = layer(x)

        if index == 4:
            l4 = x

        elif index == 8:
            l8 = x

    if l4 is None or l8 is None:
        raise RuntimeError("Failed to capture L4/L8 features.")

    batch_size = l4.shape[0]

    l4 = l4.permute(
        0,
        2,
        3,
        1,
    ).reshape(
        batch_size,
        -1,
        l4.shape[1],
    )

    l8 = l8.permute(
        0,
        2,
        3,
        1,
    ).reshape(
        batch_size,
        -1,
        l8.shape[1],
    )

    l4 = F.normalize(
        l4,
        p=2,
        dim=2,
    )

    l8 = F.normalize(
        l8,
        p=2,
        dim=2,
    )

    return l4, l8
