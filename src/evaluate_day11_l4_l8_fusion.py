from pathlib import Path
import json

import numpy as np
import torch
from PIL import Image
from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
)

# ============================================================
# CONFIG
# ============================================================

DATA_ROOT = Path("data")

NORMAL_SPLIT_PATH = Path("artifacts/splits/normal_splits.json")

DEFECT_SCORES_PATH = Path("artifacts/evaluation/mobilenet_l8_dev_scores.json")

OUTPUT_PATH = Path("artifacts/evaluation/day11_l4_l8_fusion.json")

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
    "L8": 8,
}

WEIGHTS = [
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
]

TOP_K = 5

DEVICE = torch.device("cpu")


# ============================================================
# MODEL
# ============================================================


def build_model():
    weights = MobileNet_V3_Small_Weights.DEFAULT

    model = mobilenet_v3_small(weights=weights)

    model.eval()
    model.to(DEVICE)

    return model


# ============================================================
# PREPROCESSING
# ============================================================


def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")

    image = image.resize((224, 224))

    image = (
        np.asarray(
            image,
            dtype=np.float32,
        )
        / 255.0
    )

    image = torch.from_numpy(image).permute(
        2,
        0,
        1,
    )

    mean = torch.tensor([0.485, 0.456, 0.406]).view(
        3,
        1,
        1,
    )

    std = torch.tensor([0.229, 0.224, 0.225]).view(
        3,
        1,
        1,
    )

    image = (image - mean) / std

    return image.unsqueeze(0).to(DEVICE)


# ============================================================
# PATH
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
    x = preprocess_image(image_path)

    for index, layer in enumerate(model.features):
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
    channels, height, width = feature_map.shape

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

        patches = feature_map_to_patches(feature_map)

        banks.append(patches.cpu())

        if index % 50 == 0 or index == total:
            print(f"    reference: " f"{index}/{total}")

    return torch.cat(
        banks,
        dim=0,
    )


# ============================================================
# SCORE
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

    patches = feature_map_to_patches(feature_map)

    distances = torch.cdist(
        patches,
        reference_bank,
        p=2,
    )

    nearest_distances = distances.min(dim=1).values

    k = min(
        TOP_K,
        len(nearest_distances),
    )

    topk_values = torch.topk(
        nearest_distances,
        k=k,
    ).values

    return float(topk_values.mean().item())


# ============================================================
# NORMALIZATION
# ============================================================


def fit_normalization(scores):
    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    median = np.median(scores)

    mad = np.median(np.abs(scores - median))

    if mad < 1e-12:
        mad = np.std(scores)

    if mad < 1e-12:
        mad = 1.0

    return {
        "median": float(median),
        "mad": float(mad),
    }


def normalize_scores(
    scores,
    statistics,
):
    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    return (scores - statistics["median"]) / statistics["mad"]


# ============================================================
# METRICS
# ============================================================


def calculate_auroc(
    normal_scores,
    defect_scores,
):
    normal_scores = np.asarray(
        normal_scores,
        dtype=np.float64,
    )

    defect_scores = np.asarray(
        defect_scores,
        dtype=np.float64,
    )

    comparisons = defect_scores[:, None] - normal_scores[None, :]

    greater = np.sum(comparisons > 0)

    ties = np.sum(comparisons == 0)

    return float((greater + 0.5 * ties) / (len(defect_scores) * len(normal_scores)))


def calculate_average_precision(
    normal_scores,
    defect_scores,
):
    normal_scores = np.asarray(
        normal_scores,
        dtype=np.float64,
    )

    defect_scores = np.asarray(
        defect_scores,
        dtype=np.float64,
    )

    scores = np.concatenate(
        [
            normal_scores,
            defect_scores,
        ]
    )

    labels = np.concatenate(
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

    order = np.argsort(
        -scores,
        kind="mergesort",
    )

    sorted_labels = labels[order]

    cumulative_tp = np.cumsum(sorted_labels)

    positions = np.arange(
        1,
        len(sorted_labels) + 1,
    )

    precision = cumulative_tp / positions

    total_positives = np.sum(labels)

    if total_positives == 0:
        return 0.0

    return float(np.sum(precision * sorted_labels) / total_positives)


def calculate_metrics(
    normal_scores,
    defect_scores,
):
    return {
        "auroc": calculate_auroc(
            normal_scores,
            defect_scores,
        ),
        "average_precision": (
            calculate_average_precision(
                normal_scores,
                defect_scores,
            )
        ),
    }


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 72)
    print("VISYN — DAY 11 L4 + L8 FUSION")
    print("=" * 72)

    print()
    print("L4: 40 × 14 × 14")
    print("L8: 48 × 14 × 14")
    print("Distance: Euclidean (torch.cdist)")
    print("Aggregation: TOP5")
    print("Normalization: median/MAD")
    print("Normalization fitted on normal development only")
    print("Split: canonical development")
    print("Final locked defect set: NOT USED")
    print()

    # --------------------------------------------------------
    # Load canonical split
    # --------------------------------------------------------

    with open(
        NORMAL_SPLIT_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        normal_split = json.load(f)

    # --------------------------------------------------------
    # Load defect development paths
    # --------------------------------------------------------

    with open(
        DEFECT_SCORES_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        defect_data = json.load(f)

    model = build_model()

    results = {
        "model": "MobileNetV3-Small",
        "layers": {
            "L4": {
                "index": 4,
                "shape": [
                    40,
                    14,
                    14,
                ],
            },
            "L8": {
                "index": 8,
                "shape": [
                    48,
                    14,
                    14,
                ],
            },
        },
        "distance_metric": ("euclidean_cdist_p2"),
        "aggregation": "TOP5",
        "normalization": "median_mad",
        "normalization_fit": ("normal_development_only"),
        "split": ("canonical_development"),
        "final_set_used": False,
        "categories": {},
    }

    # ========================================================
    # CATEGORY LOOP
    # ========================================================

    for category in CATEGORIES:

        print()
        print("=" * 72)
        print(f"CATEGORY: {category}")
        print("=" * 72)

        reference_paths = [
            resolve_path(path)
            for path in normal_split["categories"][category]["reference"]
        ]

        normal_dev_paths = [
            resolve_path(path)
            for path in normal_split["categories"][category]["development"]
        ]

        defect_samples = defect_data[category]["defect_dev"]

        defect_paths = [resolve_path(sample["path"]) for sample in defect_samples]

        # ----------------------------------------------------
        # Build banks
        # ----------------------------------------------------

        reference_banks = {}

        for (
            layer_name,
            layer_index,
        ) in LAYERS.items():

            print()
            print(f"Building {layer_name} " "reference bank...")

            reference_banks[layer_name] = build_patch_bank(
                model,
                reference_paths,
                layer_index,
            )

            print(f"{layer_name} bank: " f"{tuple(reference_banks[layer_name].shape)}")

        # ----------------------------------------------------
        # Score normal development
        # ----------------------------------------------------

        layer_normal_scores = {
            "L4": [],
            "L8": [],
        }

        print()
        print("Scoring normal development...")

        for index, image_path in enumerate(
            normal_dev_paths,
            start=1,
        ):

            for (
                layer_name,
                layer_index,
            ) in LAYERS.items():

                score = calculate_top5_score(
                    model,
                    image_path,
                    layer_index,
                    reference_banks[layer_name],
                )

                layer_normal_scores[layer_name].append(score)

            if index % 25 == 0 or index == len(normal_dev_paths):
                print(f"    normal: " f"{index}/" f"{len(normal_dev_paths)}")

        # ----------------------------------------------------
        # Score defect development
        # ----------------------------------------------------

        layer_defect_scores = {
            "L4": [],
            "L8": [],
        }

        print()
        print("Scoring defect development...")

        for index, image_path in enumerate(
            defect_paths,
            start=1,
        ):

            for (
                layer_name,
                layer_index,
            ) in LAYERS.items():

                score = calculate_top5_score(
                    model,
                    image_path,
                    layer_index,
                    reference_banks[layer_name],
                )

                layer_defect_scores[layer_name].append(score)

            if index % 25 == 0 or index == len(defect_paths):
                print(f"    defect: " f"{index}/" f"{len(defect_paths)}")

        # ----------------------------------------------------
        # Fit normalization on normal only
        # ----------------------------------------------------

        normalization = {}

        for layer_name in LAYERS:

            normalization[layer_name] = fit_normalization(
                layer_normal_scores[layer_name]
            )

        # ----------------------------------------------------
        # Normalize scores
        # ----------------------------------------------------

        normalized_normal = {}
        normalized_defect = {}

        for layer_name in LAYERS:

            normalized_normal[layer_name] = normalize_scores(
                layer_normal_scores[layer_name],
                normalization[layer_name],
            )

            normalized_defect[layer_name] = normalize_scores(
                layer_defect_scores[layer_name],
                normalization[layer_name],
            )

        # ----------------------------------------------------
        # Save sample-level scores
        # ----------------------------------------------------

        sample_scores = {
            "normal": {
                "paths": [str(path) for path in normal_dev_paths],
                "L4": [float(x) for x in layer_normal_scores["L4"]],
                "L8": [float(x) for x in layer_normal_scores["L8"]],
                "L4_normalized": [float(x) for x in normalized_normal["L4"]],
                "L8_normalized": [float(x) for x in normalized_normal["L8"]],
            },
            "defect": {
                "paths": [str(path) for path in defect_paths],
                "L4": [float(x) for x in layer_defect_scores["L4"]],
                "L8": [float(x) for x in layer_defect_scores["L8"]],
                "L4_normalized": [float(x) for x in normalized_defect["L4"]],
                "L8_normalized": [float(x) for x in normalized_defect["L8"]],
            },
        }

        # ----------------------------------------------------
        # Fusion weights
        # ----------------------------------------------------

        fusion_results = []

        for l4_weight in WEIGHTS:

            l8_weight = 1.0 - l4_weight

            fused_normal = (
                l4_weight * normalized_normal["L4"]
                + l8_weight * normalized_normal["L8"]
            )

            fused_defect = (
                l4_weight * normalized_defect["L4"]
                + l8_weight * normalized_defect["L8"]
            )

            metrics = calculate_metrics(
                fused_normal,
                fused_defect,
            )

            fusion_results.append(
                {
                    "l4_weight": float(l4_weight),
                    "l8_weight": float(l8_weight),
                    "auroc": metrics["auroc"],
                    "average_precision": (metrics["average_precision"]),
                }
            )

        best = max(
            fusion_results,
            key=lambda x: x["auroc"],
        )

        results["categories"][category] = {
            "normal_count": len(normal_dev_paths),
            "defect_count": len(defect_paths),
            "normalization": normalization,
            "sample_scores": sample_scores,
            "fusion_results": fusion_results,
            "best": best,
        }

        print()
        print(f"Best L4 weight: " f"{best['l4_weight']:.1f}")

        print(f"Best L8 weight: " f"{best['l8_weight']:.1f}")

        print(f"Best AUROC: " f"{best['auroc']:.4f}")

        print(f"Best AP: " f"{best['average_precision']:.4f}")

    # ========================================================
    # MACRO FUSION
    # ========================================================

    macro_results = []

    for l4_weight in WEIGHTS:

        l8_weight = 1.0 - l4_weight

        category_aurocs = []
        category_aps = []

        for category in CATEGORIES:

            category_results = results["categories"][category]["fusion_results"]

            candidate = next(
                item
                for item in category_results
                if abs(item["l4_weight"] - l4_weight) < 1e-9
            )

            category_aurocs.append(candidate["auroc"])

            category_aps.append(candidate["average_precision"])

        macro_results.append(
            {
                "l4_weight": float(l4_weight),
                "l8_weight": float(l8_weight),
                "macro_auroc": float(np.mean(category_aurocs)),
                "macro_average_precision": (float(np.mean(category_aps))),
            }
        )

    best_macro = max(
        macro_results,
        key=lambda x: x["macro_auroc"],
    )

    results["macro_fusion"] = {
        "weights": macro_results,
        "best": best_macro,
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
    print("MACRO FUSION SUMMARY")
    print("=" * 72)

    print()
    print(f"{'L4':>6}" f"{'L8':>6}" f"{'Macro AUROC':>15}" f"{'Macro AP':>15}")

    print("-" * 48)

    for item in macro_results:

        print(
            f"{item['l4_weight']:>6.1f}"
            f"{item['l8_weight']:>6.1f}"
            f"{item['macro_auroc']:>15.4f}"
            f"{item['macro_average_precision']:>15.4f}"
        )

    print("-" * 48)

    print()
    print("BEST MACRO FUSION")

    print(f"L4 weight: " f"{best_macro['l4_weight']:.1f}")

    print(f"L8 weight: " f"{best_macro['l8_weight']:.1f}")

    print(f"Macro AUROC: " f"{best_macro['macro_auroc']:.4f}")

    print(f"Macro AP: " f"{best_macro['macro_average_precision']:.4f}")

    print()
    print("Sample-level L4/L8 scores saved " "for paired bootstrap.")

    print(f"Saved to: {OUTPUT_PATH}")

    print("=" * 72)


if __name__ == "__main__":
    main()
