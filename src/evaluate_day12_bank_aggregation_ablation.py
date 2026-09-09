import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torchvision import models, transforms


# ============================================================
# Configuration
# ============================================================

SEED = 42

NORMAL_SPLIT_PATH = Path(
    "artifacts/splits/normal_splits.json"
)

DEFECT_SPLIT_PATH = Path(
    "artifacts/splits/defect_splits.json"
)

OUTPUT_PATH = Path(
    "artifacts/evaluation/day12_bank_aggregation_ablation.json"
)

CATEGORIES = [
    "bottle",
    "hazelnut",
    "cable",
    "capsule",
    "screw",
    "metal_nut",
]

REFERENCE_FRACTIONS = [
    1.0,
    0.50,
    0.25,
]

AGGREGATIONS = [
    "MAX",
    "TOP5",
    "TOP10",
]

L4_WEIGHT = 0.50
L8_WEIGHT = 0.50


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# Preprocessing
# ============================================================

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


# ============================================================
# Model
# ============================================================

device = torch.device("cpu")

model = models.mobilenet_v3_small(
    weights=models.MobileNet_V3_Small_Weights.DEFAULT
)

model = model.to(device)
model.eval()


# ============================================================
# Feature extraction
# ============================================================

@torch.no_grad()
def extract_features(image_path):
    image = Image.open(image_path).convert("RGB")

    tensor = transform(image).unsqueeze(0).to(device)

    x = tensor

    l4 = None
    l8 = None

    for idx, layer in enumerate(model.features):
        x = layer(x)

        if idx == 4:
            l4 = x.clone()

        if idx == 8:
            l8 = x.clone()

    if l4 is None or l8 is None:
        raise RuntimeError(
            "Failed to extract L4/L8 features."
        )

    l4 = (
        l4.squeeze(0)
        .permute(1, 2, 0)
        .reshape(-1, l4.shape[1])
    )

    l8 = (
        l8.squeeze(0)
        .permute(1, 2, 0)
        .reshape(-1, l8.shape[1])
    )

    l4 = torch.nn.functional.normalize(
        l4,
        dim=1,
    )

    l8 = torch.nn.functional.normalize(
        l8,
        dim=1,
    )

    return l4.cpu(), l8.cpu()


# ============================================================
# Paths
# ============================================================

def resolve_path(path_string):
    path = Path(path_string)

    if path.is_absolute():
        return path

    return Path("data") / path


# ============================================================
# Aggregation
# ============================================================

def aggregate_patch_distances(
    nearest_distances,
    aggregation,
):
    if aggregation == "MAX":
        return float(
            nearest_distances.max().item()
        )

    if aggregation == "TOP5":
        k = min(
            5,
            nearest_distances.numel(),
        )

    elif aggregation == "TOP10":
        k = min(
            10,
            nearest_distances.numel(),
        )

    else:
        raise ValueError(
            f"Unknown aggregation: {aggregation}"
        )

    return float(
        torch.topk(
            nearest_distances,
            k=k,
            largest=True,
        ).values.mean().item()
    )


# ============================================================
# Score against reference bank
# ============================================================

def score_against_bank(
    query_features,
    reference_features,
    aggregation,
):
    distances = torch.cdist(
        query_features,
        reference_features,
        p=2,
    )

    nearest = distances.min(dim=1).values

    return aggregate_patch_distances(
        nearest,
        aggregation,
    )


# ============================================================
# Robust normalization
# ============================================================

def fit_normalization(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    median = float(
        np.median(values)
    )

    mad = float(
        np.median(
            np.abs(
                values - median
            )
        )
    )

    if mad < 1e-8:
        mad = 1e-8

    return {
        "median": median,
        "mad": mad,
    }


def robust_normalize(
    value,
    normalization,
):
    return (
        value - normalization["median"]
    ) / (
        1.4826 * normalization["mad"]
    )


# ============================================================
# Load splits
# ============================================================

def load_splits():
    with open(
        NORMAL_SPLIT_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        normal_splits = json.load(f)

    with open(
        DEFECT_SPLIT_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        defect_splits = json.load(f)

    return normal_splits, defect_splits


# ============================================================
# Feature cache
# ============================================================

def build_feature_cache(
    normal_splits,
    defect_splits,
):
    cache = {}

    for category in CATEGORIES:
        print(
            f"Extracting features: {category}"
        )

        reference_paths = list(
            normal_splits[
                "categories"
            ][category]["reference"]
        )

        normal_dev_paths = list(
            normal_splits[
                "categories"
            ][category]["development"]
        )

        defect_dev_paths = list(
            defect_splits[
                "categories"
            ][category]["development"]
        )

        all_paths = (
            reference_paths
            + normal_dev_paths
            + defect_dev_paths
        )

        category_cache = {}

        for path_string in all_paths:
            if path_string in category_cache:
                continue

            path = resolve_path(
                path_string
            )

            if not path.exists():
                raise FileNotFoundError(
                    f"Image not found: {path}"
                )

            l4, l8 = extract_features(
                path
            )

            category_cache[path_string] = {
                "L4": l4,
                "L8": l8,
            }

        cache[category] = category_cache

    return cache


# ============================================================
# Evaluate one configuration
# ============================================================

def evaluate_configuration(
    category,
    reference_paths,
    normal_dev_paths,
    defect_dev_paths,
    cache,
    aggregation,
):
    l4_bank = torch.cat(
        [
            cache[path]["L4"]
            for path in reference_paths
        ],
        dim=0,
    )

    l8_bank = torch.cat(
        [
            cache[path]["L8"]
            for path in reference_paths
        ],
        dim=0,
    )

    normal_l4 = []
    normal_l8 = []

    defect_l4 = []
    defect_l8 = []

    # --------------------------------------------------------
    # Normal development
    # --------------------------------------------------------

    for path in normal_dev_paths:
        l4_score = score_against_bank(
            cache[path]["L4"],
            l4_bank,
            aggregation,
        )

        l8_score = score_against_bank(
            cache[path]["L8"],
            l8_bank,
            aggregation,
        )

        normal_l4.append(l4_score)
        normal_l8.append(l8_score)

    # --------------------------------------------------------
    # Defect development
    # --------------------------------------------------------

    for path in defect_dev_paths:
        l4_score = score_against_bank(
            cache[path]["L4"],
            l4_bank,
            aggregation,
        )

        l8_score = score_against_bank(
            cache[path]["L8"],
            l8_bank,
            aggregation,
        )

        defect_l4.append(l4_score)
        defect_l8.append(l8_score)

    # --------------------------------------------------------
    # Normalize using normal development only
    # --------------------------------------------------------

    l4_norm = fit_normalization(
        normal_l4
    )

    l8_norm = fit_normalization(
        normal_l8
    )

    normal_l4_norm = np.asarray(
        [
            robust_normalize(
                value,
                l4_norm,
            )
            for value in normal_l4
        ],
        dtype=np.float64,
    )

    normal_l8_norm = np.asarray(
        [
            robust_normalize(
                value,
                l8_norm,
            )
            for value in normal_l8
        ],
        dtype=np.float64,
    )

    defect_l4_norm = np.asarray(
        [
            robust_normalize(
                value,
                l4_norm,
            )
            for value in defect_l4
        ],
        dtype=np.float64,
    )

    defect_l8_norm = np.asarray(
        [
            robust_normalize(
                value,
                l8_norm,
            )
            for value in defect_l8
        ],
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Fixed 50/50 fusion
    # --------------------------------------------------------

    normal_fusion = (
        L4_WEIGHT * normal_l4_norm
        + L8_WEIGHT * normal_l8_norm
    )

    defect_fusion = (
        L4_WEIGHT * defect_l4_norm
        + L8_WEIGHT * defect_l8_norm
    )

    y_true = np.concatenate(
        [
            np.zeros(
                len(normal_dev_paths)
            ),
            np.ones(
                len(defect_dev_paths)
            ),
        ]
    )

    fusion_scores = np.concatenate(
        [
            normal_fusion,
            defect_fusion,
        ]
    )

    return {
        "AUROC": float(
            roc_auc_score(
                y_true,
                fusion_scores,
            )
        ),
        "AP": float(
            average_precision_score(
                y_true,
                fusion_scores,
            )
        ),
    }


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 78)
    print(
        "DAY 12 — REFERENCE BANK × AGGREGATION ABLATION"
    )
    print("=" * 78)

    normal_splits, defect_splits = load_splits()

    print("\nBuilding feature cache...")

    cache = build_feature_cache(
        normal_splits,
        defect_splits,
    )

    results = {
        "seed": SEED,
        "aggregations": AGGREGATIONS,
        "reference_fractions": REFERENCE_FRACTIONS,
        "fusion": {
            "L4_weight": L4_WEIGHT,
            "L8_weight": L8_WEIGHT,
        },
        "categories": {},
    }

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    for category in CATEGORIES:
        print("\n" + "=" * 78)
        print(
            f"CATEGORY: {category}"
        )
        print("=" * 78)

        reference_paths = list(
            normal_splits[
                "categories"
            ][category]["reference"]
        )

        normal_dev_paths = list(
            normal_splits[
                "categories"
            ][category]["development"]
        )

        defect_dev_paths = list(
            defect_splits[
                "categories"
            ][category]["development"]
        )

        results["categories"][
            category
        ] = {}

        for fraction in REFERENCE_FRACTIONS:
            if fraction == 1.0:
                selected_paths = list(
                    reference_paths
                )

            else:
                n_select = max(
                    1,
                    int(
                        round(
                            len(reference_paths)
                            * fraction
                        )
                    ),
                )

                rng = random.Random(
                    SEED
                    + int(
                        fraction * 1000
                    )
                )

                selected_paths = rng.sample(
                    reference_paths,
                    n_select,
                )

            fraction_key = str(
                fraction
            )

            results["categories"][
                category
            ][fraction_key] = {
                "reference_count": len(
                    selected_paths
                ),
                "methods": {},
            }

            print(
                f"\nReference fraction: "
                f"{fraction:.0%} "
                f"({len(selected_paths)} images)"
            )

            for aggregation in AGGREGATIONS:
                print(
                    f"  Evaluating {aggregation}..."
                )

                metrics = evaluate_configuration(
                    category=category,
                    reference_paths=selected_paths,
                    normal_dev_paths=normal_dev_paths,
                    defect_dev_paths=defect_dev_paths,
                    cache=cache[category],
                    aggregation=aggregation,
                )

                results["categories"][
                    category
                ][fraction_key][
                    "methods"
                ][aggregation] = metrics

                print(
                    f"    AUROC={metrics['AUROC']:.4f} "
                    f"AP={metrics['AP']:.4f}"
                )

    # --------------------------------------------------------
    # Macro summary
    # --------------------------------------------------------

    results["macro"] = {}

    for fraction in REFERENCE_FRACTIONS:
        fraction_key = str(
            fraction
        )

        results["macro"][
            fraction_key
        ] = {}

        for aggregation in AGGREGATIONS:
            aurocs = []
            aps = []

            for category in CATEGORIES:
                metrics = results[
                    "categories"
                ][category][fraction_key][
                    "methods"
                ][aggregation]

                aurocs.append(
                    metrics["AUROC"]
                )

                aps.append(
                    metrics["AP"]
                )

            results["macro"][
                fraction_key
            ][aggregation] = {
                "AUROC": float(
                    np.mean(aurocs)
                ),
                "AP": float(
                    np.mean(aps)
                ),
            }

    # --------------------------------------------------------
    # Degradation from 100% to 25%
    # --------------------------------------------------------

    results[
        "macro_degradation_100_to_25"
    ] = {}

    for aggregation in AGGREGATIONS:
        full = results["macro"]["1.0"][
            aggregation
        ]["AUROC"]

        quarter = results["macro"]["0.25"][
            aggregation
        ]["AUROC"]

        results[
            "macro_degradation_100_to_25"
        ][aggregation] = {
            "absolute": float(
                quarter - full
            ),
            "relative_percent": float(
                (
                    (quarter - full)
                    / full
                )
                * 100.0
            ),
        }

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print("\n" + "=" * 78)
    print(
        "MACRO FUSION RESULTS"
    )
    print("=" * 78)

    for fraction in REFERENCE_FRACTIONS:
        key = str(fraction)

        print(
            f"\nReference fraction: "
            f"{fraction:.0%}"
        )

        for aggregation in AGGREGATIONS:
            metrics = results[
                "macro"
            ][key][aggregation]

            print(
                f"  {aggregation:6s} "
                f"AUROC={metrics['AUROC']:.4f} "
                f"AP={metrics['AP']:.4f}"
            )

    print("\n" + "=" * 78)
    print(
        "100% → 25% AUROC DEGRADATION"
    )
    print("=" * 78)

    for aggregation in AGGREGATIONS:
        degradation = results[
            "macro_degradation_100_to_25"
        ][aggregation]

        print(
            f"  {aggregation:6s} "
            f"Δ={degradation['absolute']:+.4f} "
            f"({degradation['relative_percent']:+.2f}%)"
        )

    print("\nResults saved to:")
    print(OUTPUT_PATH)

    print(
        "\nDAY 12 EXPERIMENT 5 COMPLETE"
    )


if __name__ == "__main__":
    main()