from pathlib import Path
import json

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score
from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
)


# ============================================================
# CONFIG
# ============================================================

DATA_ROOT = Path("data")

NORMAL_SPLIT_PATH = Path(
    "artifacts/splits/normal_splits.json"
)

DEFECT_SCORES_PATH = Path(
    "artifacts/evaluation/mobilenet_l8_dev_scores.json"
)

OUTPUT_PATH = Path(
    "artifacts/evaluation/day11_depth_ablation.json"
)

CATEGORIES = [
    "bottle",
    "hazelnut",
    "cable",
    "capsule",
    "screw",
    "metal_nut",
]

LAYERS = {
    "L4": 4,
    "L6": 6,
    "L8": 8,
}

TOP_K = 5

DEVICE = torch.device("cpu")


# ============================================================
# MODEL
# ============================================================

def build_model():
    weights = MobileNet_V3_Small_Weights.DEFAULT

    model = mobilenet_v3_small(
        weights=weights
    )

    model.eval()
    model.to(DEVICE)

    return model


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_image(image_path):
    image = Image.open(
        image_path
    ).convert("RGB")

    image = image.resize(
        (224, 224)
    )

    image = np.asarray(
        image,
        dtype=np.float32,
    ) / 255.0

    image = torch.from_numpy(
        image
    ).permute(
        2,
        0,
        1,
    )

    mean = torch.tensor(
        [0.485, 0.456, 0.406]
    ).view(3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(3, 1, 1)

    image = (
        image - mean
    ) / std

    return image.unsqueeze(0).to(DEVICE)


# ============================================================
# PATH RESOLUTION
# ============================================================

def resolve_path(relative_path):
    path = Path(relative_path)

    if path.is_absolute():
        return path

    return DATA_ROOT / path


# ============================================================
# FEATURE EXTRACTION
# ============================================================

@torch.no_grad()
def extract_feature_map(
    model,
    image_path,
    layer_index,
):
    x = preprocess_image(
        image_path
    )

    for index, layer in enumerate(
        model.features
    ):
        x = layer(x)

        if index == layer_index:
            break

    return x.squeeze(0)


# ============================================================
# PATCH EXTRACTION
# ============================================================

def feature_map_to_patches(
    feature_map,
):
    channels, height, width = (
        feature_map.shape
    )

    patches = feature_map.permute(
        1,
        2,
        0,
    ).reshape(
        height * width,
        channels,
    )

    patches = torch.nn.functional.normalize(
        patches,
        p=2,
        dim=1,
    )

    return patches


# ============================================================
# PATCH BANK
# ============================================================

def build_patch_bank(
    model,
    image_paths,
    layer_index,
):
    banks = []

    total = len(image_paths)

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):
        feature_map = extract_feature_map(
            model,
            image_path,
            layer_index,
        )

        patches = feature_map_to_patches(
            feature_map
        )

        banks.append(
            patches.cpu()
        )

        if (
            index % 50 == 0
            or index == total
        ):
            print(
                f"    reference: "
                f"{index}/{total}"
            )

    return torch.cat(
        banks,
        dim=0,
    )


# ============================================================
# IMAGE SCORE
# ============================================================

@torch.no_grad()
def calculate_top5_score(
    model,
    image_path,
    layer_index,
    reference_bank,
):
    feature_map = extract_feature_map(
        model,
        image_path,
        layer_index,
    )

    patches = feature_map_to_patches(
        feature_map
    )

    distances = torch.cdist(
        patches,
        reference_bank,
        p=2,
    )

    nearest_distances = distances.min(
        dim=1
    ).values

    k = min(
        TOP_K,
        len(nearest_distances),
    )

    topk_values = torch.topk(
        nearest_distances,
        k=k,
    ).values

    return float(
        topk_values.mean().item()
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    normal_scores,
    defect_scores,
):
    y_true = np.concatenate(
        [
            np.zeros(
                len(normal_scores),
                dtype=np.int32,
            ),
            np.ones(
                len(defect_scores),
                dtype=np.int32,
            ),
        ]
    )

    y_score = np.concatenate(
        [
            normal_scores,
            defect_scores,
        ]
    )

    return {
        "normal_count": len(
            normal_scores
        ),
        "defect_count": len(
            defect_scores
        ),
        "auroc": float(
            roc_auc_score(
                y_true,
                y_score,
            )
        ),
        "average_precision": float(
            average_precision_score(
                y_true,
                y_score,
            )
        ),
        "normal_median": float(
            np.median(
                normal_scores
            )
        ),
        "defect_median": float(
            np.median(
                defect_scores
            )
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 72)
    print(
        "VISIONFORGE — DAY 11 DEPTH ABLATION"
    )
    print("=" * 72)

    print()
    print(
        "Layers: L4, L6, L8"
    )
    print(
        "L4: 40 × 14 × 14"
    )
    print(
        "L6: 40 × 14 × 14"
    )
    print(
        "L8: 48 × 14 × 14"
    )
    print(
        "Distance: Euclidean (torch.cdist)"
    )
    print(
        "Aggregation: TOP5"
    )
    print(
        "Split: canonical development"
    )
    print(
        "Final locked defect set: NOT USED"
    )
    print()

    # --------------------------------------------------------
    # Load split
    # --------------------------------------------------------

    with open(
        NORMAL_SPLIT_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        normal_split = json.load(f)

    # --------------------------------------------------------
    # Load defect development data
    # --------------------------------------------------------

    with open(
        DEFECT_SCORES_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        defect_data = json.load(f)

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = build_model()

    results = {
        "model": "MobileNetV3-Small",
        "distance_metric": "euclidean_cdist_p2",
        "aggregation": "TOP5",
        "split": "canonical_development",
        "final_set_used": False,
        "layers": {},
    }

    # ========================================================
    # LAYER LOOP
    # ========================================================

    for layer_name, layer_index in LAYERS.items():
        print()
        print("=" * 72)
        print(
            f"EVALUATING {layer_name}"
        )
        print("=" * 72)

        results["layers"][layer_name] = {
            "layer_index": layer_index,
            "categories": {},
        }

        macro_aurocs = []
        macro_aps = []

        # ----------------------------------------------------
        # Category loop
        # ----------------------------------------------------

        for category in CATEGORIES:
            print()
            print(
                f"Category: {category}"
            )

            reference_paths = [
                resolve_path(path)
                for path in normal_split[
                    "categories"
                ][category]["reference"]
            ]

            development_paths = [
                resolve_path(path)
                for path in normal_split[
                    "categories"
                ][category]["development"]
            ]

            print(
                f"Reference images: "
                f"{len(reference_paths)}"
            )

            print(
                "Building patch bank..."
            )

            reference_bank = build_patch_bank(
                model,
                reference_paths,
                layer_index,
            )

            print(
                f"Patch bank shape: "
                f"{tuple(reference_bank.shape)}"
            )

            # ------------------------------------------------
            # Normal development
            # ------------------------------------------------

            normal_scores = []

            print(
                "Scoring normal development..."
            )

            for index, image_path in enumerate(
                development_paths,
                start=1,
            ):
                score = calculate_top5_score(
                    model,
                    image_path,
                    layer_index,
                    reference_bank,
                )

                normal_scores.append(
                    score
                )

                if (
                    index % 25 == 0
                    or index == len(
                        development_paths
                    )
                ):
                    print(
                        f"    normal: "
                        f"{index}/"
                        f"{len(development_paths)}"
                    )

            # ------------------------------------------------
            # Defect development
            # ------------------------------------------------

            defect_samples = (
                defect_data[
                    category
                ]["defect_dev"]
            )

            defect_scores = []

            print(
                "Scoring defect development..."
            )

            for index, sample in enumerate(
                defect_samples,
                start=1,
            ):
                image_path = resolve_path(
                    sample["path"]
                )

                score = calculate_top5_score(
                    model,
                    image_path,
                    layer_index,
                    reference_bank,
                )

                defect_scores.append(
                    score
                )

                if (
                    index % 25 == 0
                    or index == len(
                        defect_samples
                    )
                ):
                    print(
                        f"    defect: "
                        f"{index}/"
                        f"{len(defect_samples)}"
                    )

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            metrics = calculate_metrics(
                normal_scores,
                defect_scores,
            )

            results["layers"][layer_name][
                "categories"
            ][category] = metrics

            macro_aurocs.append(
                metrics["auroc"]
            )

            macro_aps.append(
                metrics["average_precision"]
            )

            print(
                f"  AUROC: "
                f"{metrics['auroc']:.4f}"
            )

            print(
                f"  AP: "
                f"{metrics['average_precision']:.4f}"
            )

        # ----------------------------------------------------
        # Macro
        # ----------------------------------------------------

        results["layers"][layer_name][
            "macro"
        ] = {
            "auroc": float(
                np.mean(
                    macro_aurocs
                )
            ),
            "average_precision": float(
                np.mean(
                    macro_aps
                )
            ),
        }

        print()
        print(
            f"{layer_name} macro AUROC: "
            f"{np.mean(macro_aurocs):.4f}"
        )

        print(
            f"{layer_name} macro AP: "
            f"{np.mean(macro_aps):.4f}"
        )

    # ========================================================
    # DIRECT COMPARISON
    # ========================================================

    l4 = results["layers"]["L4"]
    l6 = results["layers"]["L6"]
    l8 = results["layers"]["L8"]

    results["comparison"] = {
        "macro": {
            "L4_minus_L8": (
                l4["macro"]["auroc"]
                - l8["macro"]["auroc"]
            ),
            "L6_minus_L8": (
                l6["macro"]["auroc"]
                - l8["macro"]["auroc"]
            ),
        },
        "categories": {},
    }

    for category in CATEGORIES:
        results["comparison"][
            "categories"
        ][category] = {
            "L4_minus_L8": (
                l4["categories"][category]["auroc"]
                - l8["categories"][category]["auroc"]
            ),
            "L6_minus_L8": (
                l6["categories"][category]["auroc"]
                - l8["categories"][category]["auroc"]
            ),
        }

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 72)
    print(
        "DAY 11 DEPTH ABLATION SUMMARY"
    )
    print("=" * 72)

    print()
    print(
        f"{'Layer':<10}"
        f"{'Macro AUROC':>15}"
        f"{'Macro AP':>15}"
    )

    print("-" * 42)

    for layer_name in [
        "L4",
        "L6",
        "L8",
    ]:
        metrics = results[
            "layers"
        ][layer_name]["macro"]

        print(
            f"{layer_name:<10}"
            f"{metrics['auroc']:>15.4f}"
            f"{metrics['average_precision']:>15.4f}"
        )

    print("-" * 42)

    print()
    print(
        f"L4 − L8 macro AUROC: "
        f"{results['comparison']['macro']['L4_minus_L8']:+.4f}"
    )

    print(
        f"L6 − L8 macro AUROC: "
        f"{results['comparison']['macro']['L6_minus_L8']:+.4f}"
    )

    print()
    print(
        f"Saved to: {OUTPUT_PATH}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()