import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
)


# ============================================================
# CONFIG
# ============================================================

SPLITS_PATH = Path(
    "artifacts/splits/normal_splits.json"
)

OUTPUT_DIR = Path(
    "artifacts/patch_banks/mobilenet_l8_spatial"
)

DATA_ROOT = Path("data")

DEVICE = torch.device("cpu")

FEATURE_H = 14
FEATURE_W = 14
FEATURE_C = 48

CATEGORIES = [
    "bottle",
    "hazelnut",
    "cable",
    "capsule",
    "screw",
    "metal_nut",
]


# ============================================================
# PREPROCESSING
# ============================================================

weights = MobileNet_V3_Small_Weights.DEFAULT
preprocess = weights.transforms()


# ============================================================
# MODEL
# ============================================================

def build_model():

    model = mobilenet_v3_small(
        weights=weights
    )

    feature_extractor = nn.Sequential(
        *list(model.features.children())[:9]
    )

    feature_extractor.eval()
    feature_extractor.to(DEVICE)

    return feature_extractor


# ============================================================
# PATH
# ============================================================

def resolve_image_path(image_path):

    image_path = Path(image_path)

    if image_path.is_absolute():
        return image_path

    return DATA_ROOT / image_path


# ============================================================
# FEATURE EXTRACTION
# ============================================================

@torch.no_grad()
def extract_feature_map(
    model,
    image_path
):

    image_path = resolve_image_path(
        image_path
    )

    image = Image.open(
        image_path
    ).convert("RGB")

    tensor = preprocess(
        image
    ).unsqueeze(
        0
    ).to(
        DEVICE
    )

    feature_map = model(
        tensor
    ).squeeze(0)

    if tuple(feature_map.shape) != (
        FEATURE_C,
        FEATURE_H,
        FEATURE_W
    ):

        raise RuntimeError(
            "Unexpected feature map shape: "
            f"{tuple(feature_map.shape)}"
        )

    return feature_map.cpu().numpy()


# ============================================================
# CONVERT FEATURE MAP TO SPATIAL PATCHES
# ============================================================

def feature_map_to_spatial_patches(
    feature_map
):

    # Input:
    # [48, 14, 14]
    #
    # Output:
    # [14, 14, 48]

    patches = np.transpose(
        feature_map,
        (1, 2, 0)
    )

    # Normalize every spatial descriptor.

    norms = np.linalg.norm(
        patches,
        axis=2,
        keepdims=True
    )

    patches = patches / np.maximum(
        norms,
        1e-12
    )

    return patches.astype(
        np.float32
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print(
        "VISIONFORGE — SPATIAL MOBILENET L8 PATCH BANKS"
    )
    print("=" * 72)

    print(
        "Representation: MobileNetV3-Small L8"
    )

    print(
        "Feature map: 48 x 14 x 14"
    )

    print(
        "Spatial reference matching: ENABLED"
    )

    # --------------------------------------------------------
    # Load canonical normal split
    # --------------------------------------------------------

    split_data = json.loads(
        SPLITS_PATH.read_text(
            encoding="utf-8"
        )
    )

    model = build_model()

    print(
        "Model loaded."
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # CATEGORY LOOP
    # ========================================================

    for category in CATEGORIES:

        print()
        print("-" * 72)
        print(category.upper())
        print("-" * 72)

        reference_images = (
            split_data[
                "categories"
            ][
                category
            ][
                "reference"
            ]
        )

        print(
            f"Reference images: "
            f"{len(reference_images)}"
        )

        # ----------------------------------------------------
        # Allocate:
        #
        # [N, 14, 14, 48]
        # ----------------------------------------------------

        bank = np.empty(
            (
                len(reference_images),
                FEATURE_H,
                FEATURE_W,
                FEATURE_C,
            ),
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Extract each normal reference image
        # ----------------------------------------------------

        for index, image_path in enumerate(
            reference_images
        ):

            feature_map = (
                extract_feature_map(
                    model,
                    image_path
                )
            )

            spatial_patches = (
                feature_map_to_spatial_patches(
                    feature_map
                )
            )

            bank[index] = (
                spatial_patches
            )

            if (
                (index + 1) % 25 == 0
                or index + 1 == len(reference_images)
            ):

                print(
                    f"Processed "
                    f"{index + 1}/"
                    f"{len(reference_images)}"
                )

        # ----------------------------------------------------
        # Verify normalization
        # ----------------------------------------------------

        norms = np.linalg.norm(
            bank,
            axis=3
        )

        print(
            "Descriptor norm:"
        )

        print(
            f"  min  = {norms.min():.6f}"
        )

        print(
            f"  mean = {norms.mean():.6f}"
        )

        print(
            f"  max  = {norms.max():.6f}"
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        output_path = (
            OUTPUT_DIR
            / f"{category}.npy"
        )

        np.save(
            output_path,
            bank
        )

        print(
            f"Saved: {output_path}"
        )

        print(
            f"Shape: {bank.shape}"
        )

    print()
    print("=" * 72)
    print(
        "SPATIAL PATCH BANK BUILD COMPLETE"
    )
    print("=" * 72)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()