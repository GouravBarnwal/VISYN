import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from src.production_config import (
    AGGREGATION,
    ARTIFACTS_ROOT,
    BULK_BATCH_SIZE,
    CHUNK_SIZE,
    CATEGORIES,
    DEFAULT_BATCH_SIZE,
    DISTANCE,
    FEATURE_LAYERS,
    FUSION_WEIGHT_L4,
    FUSION_WEIGHT_L8,
    IMAGE_SIZE,
    IMAGE_MEAN,
    IMAGE_STD,
    LOCALIZATION_ALPHA,
    LOCALIZATION_METHOD,
    MODEL_NAME,
    PATCHES_PER_LAYER,
    RANDOM_SEED,
    REFERENCE_RATIO,
    REVIEW_MULTIPLIER,
    THRESHOLD_PERCENTILE,
    TOP_K,
)


PRODUCTION_ROOT = ARTIFACTS_ROOT / "production"
REFERENCE_MANIFEST_PATH = PRODUCTION_ROOT / "reference_bank_manifest.json"
THRESHOLDS_PATH = ARTIFACTS_ROOT / "evaluation" / "production_thresholds.json"
OUTPUT_PATH = PRODUCTION_ROOT / "production_artifact_manifest.json"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        while True:
            chunk = file.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def file_metadata(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Required artifact not found: {path}"
        )

    return {
        "path": str(
            path.relative_to(ARTIFACTS_ROOT)
        ),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Required JSON artifact not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def resolve_bank_path(relative_path: str) -> Path:
    """
    Resolve a reference-bank path stored in the existing
    reference_bank_manifest.json.

    The existing manifest stores paths such as:

        production\\reference_banks\\bottle\\L4.npy

    relative to the artifacts directory.
    """

    path = ARTIFACTS_ROOT / Path(relative_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Reference-bank artifact not found: {path}"
        )

    return path


def build_reference_bank_metadata(reference_manifest: dict) -> dict:
    categories = {}

    manifest_categories = reference_manifest.get(
        "categories",
        {},
    )

    for category in CATEGORIES:
        if category not in manifest_categories:
            raise ValueError(
                "Category "
                f"'{category}' missing from "
                "reference_bank_manifest.json."
            )

        category_manifest = manifest_categories[category]

        l4_path = resolve_bank_path(
            category_manifest["l4_path"]
        )

        l8_path = resolve_bank_path(
            category_manifest["l8_path"]
        )

        categories[category] = {
            "reference_images": category_manifest[
                "reference_images"
            ],
            "L4": {
                "shape": category_manifest["l4_shape"],
                "artifact": file_metadata(l4_path),
            },
            "L8": {
                "shape": category_manifest["l8_shape"],
                "artifact": file_metadata(l8_path),
            },
        }

    return categories


def build_manifest() -> dict:
    reference_manifest = load_json(
        REFERENCE_MANIFEST_PATH
    )

    thresholds = load_json(
        THRESHOLDS_PATH
    )

    if reference_manifest.get("model") != MODEL_NAME:
        raise ValueError(
            "Reference-bank manifest model does not "
            "match production_config.py."
        )

    missing_thresholds = [
        category
        for category in CATEGORIES
        if category
        not in thresholds.get("categories", {})
    ]

    if missing_thresholds:
        raise ValueError(
            "Missing production thresholds for "
            "categories: "
            + ", ".join(missing_thresholds)
        )

    return {
        "artifact_type": (
            "VisionForge production inference bundle"
        ),
        "schema_version": "1.0",
        "generated_at_utc": (
            datetime.now(timezone.utc).isoformat()
        ),
        "model": {
            "name": MODEL_NAME,
            "feature_layers": FEATURE_LAYERS,
            "patches_per_layer": PATCHES_PER_LAYER,
        },
        "preprocessing": {
            "resize": [
                IMAGE_SIZE,
                IMAGE_SIZE,
            ],
            "resize_mode": "direct",
            "center_crop": False,
            "mean": list(IMAGE_MEAN),
            "std": list(IMAGE_STD),
        },
        "scoring": {
            "distance": DISTANCE,
            "aggregation": AGGREGATION,
            "top_k": TOP_K,
            "chunk_size": CHUNK_SIZE,
            "fusion": {
                "L4_weight": FUSION_WEIGHT_L4,
                "L8_weight": FUSION_WEIGHT_L8,
            },
        },
        "decision_policy": {
            "threshold_percentile": (
                THRESHOLD_PERCENTILE
            ),
            "review_multiplier": REVIEW_MULTIPLIER,
        },
        "localization": {
            "method": LOCALIZATION_METHOD,
            "alpha": LOCALIZATION_ALPHA,
        },
        "reference_split": {
            "seed": RANDOM_SEED,
            "reference_ratio": REFERENCE_RATIO,
            "manifest": file_metadata(
                REFERENCE_MANIFEST_PATH
            ),
        },
        "thresholds": {
            "artifact": file_metadata(
                THRESHOLDS_PATH
            ),
            "categories": {
                category: thresholds["categories"][
                    category
                ]
                for category in CATEGORIES
            },
        },
        "runtime": {
            "default_batch_size": DEFAULT_BATCH_SIZE,
            "bulk_batch_size": BULK_BATCH_SIZE,
        },
        "reference_banks": (
            build_reference_bank_metadata(
                reference_manifest
            )
        ),
    }


def main():
    manifest = build_manifest()

    PRODUCTION_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
        )

    print("=" * 72)
    print(
        "VISIONFORGE — PRODUCTION ARTIFACT MANIFEST"
    )
    print("=" * 72)
    print(f"Output: {OUTPUT_PATH}")
    print(
        f"Model: {manifest['model']['name']}"
    )
    print(
        f"Categories: "
        f"{len(manifest['reference_banks'])}"
    )

    total_files = 2

    for category, banks in (
        manifest["reference_banks"].items()
    ):
        print(
            f"{category:12s} "
            f"L4={banks['L4']['shape']} "
            f"L8={banks['L8']['shape']}"
        )

        total_files += 2

    print(
        f"Hashed production files: {total_files}"
    )
    print()
    print(
        "PRODUCTION ARTIFACT MANIFEST: PASS"
    )


if __name__ == "__main__":
    main()