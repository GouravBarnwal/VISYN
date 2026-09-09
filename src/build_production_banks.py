from pathlib import Path
import json

import numpy as np
import torch
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    mobilenet_v3_small,
)

from src.mobilenet_batch_extractor import extract_batch
from src.production_config import (
    BULK_BATCH_SIZE,
    CATEGORIES,
    DATA_ROOT,
    SPLITS_PATH,
)


OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent
    / "artifacts"
    / "production"
    / "reference_banks"
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model():
    weights = MobileNet_V3_Small_Weights.DEFAULT

    model = mobilenet_v3_small(
        weights=weights,
    )

    model.eval()

    return model


def build_category_bank(
    model,
    image_paths,
):
    l4_parts = []
    l8_parts = []

    for start in range(
        0,
        len(image_paths),
        BULK_BATCH_SIZE,
    ):
        batch_paths = image_paths[
            start:start + BULK_BATCH_SIZE
        ]

        l4, l8 = extract_batch(
            model,
            batch_paths,
        )

        # extract_batch returns:
        # [B, 196, channels]
        #
        # Production reference banks require:
        # [B * 196, channels]

        batch_size = l4.shape[0]

        l4 = l4.reshape(
            batch_size * l4.shape[1],
            l4.shape[2],
        )

        l8 = l8.reshape(
            batch_size * l8.shape[1],
            l8.shape[2],
        )

        l4_parts.append(l4)
        l8_parts.append(l8)

        processed = min(
            start + BULK_BATCH_SIZE,
            len(image_paths),
        )

        print(
            f"  Processed "
            f"{processed}/{len(image_paths)}"
        )

    l4_bank = torch.cat(
        l4_parts,
        dim=0,
    )

    l8_bank = torch.cat(
        l8_parts,
        dim=0,
    )

    return (
        l4_bank.cpu().numpy().astype(
            np.float32
        ),
        l8_bank.cpu().numpy().astype(
            np.float32
        ),
    )


def validate_bank(
    category,
    l4_bank,
    l8_bank,
    reference_count,
):
    expected_patches = (
        reference_count * 196
    )

    expected_l4_shape = (
        expected_patches,
        40,
    )

    expected_l8_shape = (
        expected_patches,
        48,
    )

    if l4_bank.shape != expected_l4_shape:
        raise RuntimeError(
            f"{category}: unexpected L4 shape "
            f"{l4_bank.shape}; expected "
            f"{expected_l4_shape}"
        )

    if l8_bank.shape != expected_l8_shape:
        raise RuntimeError(
            f"{category}: unexpected L8 shape "
            f"{l8_bank.shape}; expected "
            f"{expected_l8_shape}"
        )

    if not np.isfinite(l4_bank).all():
        raise RuntimeError(
            f"{category}: L4 bank contains "
            "non-finite values."
        )

    if not np.isfinite(l8_bank).all():
        raise RuntimeError(
            f"{category}: L8 bank contains "
            "non-finite values."
        )

    l4_norms = np.linalg.norm(
        l4_bank,
        axis=1,
    )

    l8_norms = np.linalg.norm(
        l8_bank,
        axis=1,
    )

    if not np.allclose(
        l4_norms,
        1.0,
        atol=1e-5,
    ):
        raise RuntimeError(
            f"{category}: L4 bank is not "
            "properly L2 normalized."
        )

    if not np.allclose(
        l8_norms,
        1.0,
        atol=1e-5,
    ):
        raise RuntimeError(
            f"{category}: L8 bank is not "
            "properly L2 normalized."
        )


def main():
    print("=" * 72)
    print("VISIONFORGE — PRODUCTION REFERENCE BANK BUILDER")
    print("=" * 72)

    split = load_json(
        SPLITS_PATH
    )

    model = load_model()

    print()
    print("Model: MobileNetV3-Small")
    print("Features: L4 + L8")
    print(
        "Preprocessing: "
        "canonical 224x224 direct resize"
    )
    print(
        f"Batch size: {BULK_BATCH_SIZE}"
    )
    print()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = {
        "model": "MobileNetV3-Small",
        "feature_layers": {
            "L4": {
                "layer_index": 4,
                "channels": 40,
                "patches_per_image": 196,
            },
            "L8": {
                "layer_index": 8,
                "channels": 48,
                "patches_per_image": 196,
            },
        },
        "preprocessing": {
            "resize": [224, 224],
            "resize_mode": "direct",
            "normalization": "ImageNet",
            "center_crop": False,
        },
        "reference_split": {
            "path": str(SPLITS_PATH),
            "seed": 42,
            "reference_ratio": 0.80,
        },
        "dtype": "float32",
        "categories": {},
    }

    for category in CATEGORIES:
        print("=" * 72)
        print(f"CATEGORY: {category}")
        print("=" * 72)

        relative_paths = split[
            "categories"
        ][category]["reference"]

        image_paths = [
            DATA_ROOT / path
            for path in relative_paths
        ]

        print(
            f"Reference images: "
            f"{len(image_paths)}"
        )

        l4_bank, l8_bank = (
            build_category_bank(
                model,
                image_paths,
            )
        )

        validate_bank(
            category,
            l4_bank,
            l8_bank,
            len(image_paths),
        )

        category_dir = (
            OUTPUT_DIR / category
        )

        category_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        l4_path = (
            category_dir / "L4.npy"
        )

        l8_path = (
            category_dir / "L8.npy"
        )

        np.save(
            l4_path,
            l4_bank,
        )

        np.save(
            l8_path,
            l8_bank,
        )

        manifest["categories"][category] = {
            "reference_images": len(
                image_paths
            ),
            "l4_shape": list(
                l4_bank.shape
            ),
            "l8_shape": list(
                l8_bank.shape
            ),
            "l4_path": str(
                l4_path.relative_to(
                    OUTPUT_DIR.parent.parent
                )
            ),
            "l8_path": str(
                l8_path.relative_to(
                    OUTPUT_DIR.parent.parent
                )
            ),
        }

        print()
        print(
            f"L4 bank shape: {l4_bank.shape}"
        )
        print(
            f"L8 bank shape: {l8_bank.shape}"
        )
        print(
            f"Saved: {l4_path}"
        )
        print(
            f"Saved: {l8_path}"
        )

    manifest_path = (
        OUTPUT_DIR.parent
        / "reference_bank_manifest.json"
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )

    print()
    print("=" * 72)
    print(
        "PRODUCTION REFERENCE BANK BUILD: PASS"
    )
    print("=" * 72)
    print(
        f"Manifest: {manifest_path}"
    )


if __name__ == "__main__":
    main()