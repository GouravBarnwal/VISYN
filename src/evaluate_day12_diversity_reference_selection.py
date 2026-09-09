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
    "artifacts/evaluation/day12_diversity_reference_selection.json"
)

CATEGORIES = [
    "bottle",
    "hazelnut",
    "cable",
    "capsule",
    "screw",
    "metal_nut",
]

REFERENCE_FRACTION = 0.25

TOP_K = 5

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
# Build image-level descriptors
# ============================================================

def build_image_descriptors(
    reference_paths,
    cache,
):
    """
    Build one descriptor per normal reference image.

    The descriptor is the mean normalized patch embedding.

    This descriptor is used ONLY to select representative
    reference images. Actual anomaly scoring still uses the
    complete patch bank from the selected images.
    """

    l4_descriptors = []
    l8_descriptors = []

    for path in reference_paths:
        l4 = cache[path]["L4"]
        l8 = cache[path]["L8"]

        l4_descriptor = l4.mean(dim=0)
        l8_descriptor = l8.mean(dim=0)

        l4_descriptor = torch.nn.functional.normalize(
            l4_descriptor.unsqueeze(0),
            dim=1,
        ).squeeze(0)

        l8_descriptor = torch.nn.functional.normalize(
            l8_descriptor.unsqueeze(0),
            dim=1,
        ).squeeze(0)

        l4_descriptors.append(
            l4_descriptor
        )

        l8_descriptors.append(
            l8_descriptor
        )

    l4_descriptors = torch.stack(
        l4_descriptors
    )

    l8_descriptors = torch.stack(
        l8_descriptors
    )

    # Concatenate normalized L4 and L8 descriptors.
    combined = torch.cat(
        [
            l4_descriptors,
            l8_descriptors,
        ],
        dim=1,
    )

    combined = torch.nn.functional.normalize(
        combined,
        dim=1,
    )

    return combined


# ============================================================
# Diversity selection
# ============================================================

def select_diverse_images(
    paths,
    descriptors,
    n_select,
    seed=42,
):
    """
    Greedy farthest-point selection.

    Start from a deterministic seed image, then repeatedly select
    the image farthest from the already-selected set.

    Selection happens ONLY among normal reference images.
    """

    if n_select >= len(paths):
        return list(paths)

    if n_select <= 0:
        raise ValueError(
            "n_select must be positive."
        )

    generator = torch.Generator()
    generator.manual_seed(seed)

    first_index = int(
        torch.randint(
            low=0,
            high=len(paths),
            size=(1,),
            generator=generator,
        ).item()
    )

    selected_indices = [
        first_index
    ]

    selected_descriptor = descriptors[
        first_index
    ].unsqueeze(0)

    min_distances = torch.cdist(
        descriptors,
        selected_descriptor,
        p=2,
    ).squeeze(1)

    min_distances[first_index] = -1.0

    while len(selected_indices) < n_select:
        next_index = int(
            torch.argmax(
                min_distances
            ).item()
        )

        selected_indices.append(
            next_index
        )

        new_descriptor = descriptors[
            next_index
        ].unsqueeze(0)

        new_distances = torch.cdist(
            descriptors,
            new_descriptor,
            p=2,
        ).squeeze(1)

        min_distances = torch.minimum(
            min_distances,
            new_distances,
        )

        min_distances[
            selected_indices
        ] = -1.0

    return [
        paths[index]
        for index in selected_indices
    ]


# ============================================================
# Scoring
# ============================================================

def score_against_bank(
    query_features,
    reference_features,
):
    distances = torch.cdist(
        query_features,
        reference_features,
        p=2,
    )

    nearest = distances.min(dim=1).values

    k = min(
        TOP_K,
        nearest.numel(),
    )

    top_values = torch.topk(
        nearest,
        k=k,
        largest=True,
    ).values

    return float(
        top_values.mean().item()
    )


def score_sample(
    l4_query,
    l8_query,
    l4_bank,
    l8_bank,
):
    l4_score = score_against_bank(
        l4_query,
        l4_bank,
    )

    l8_score = score_against_bank(
        l8_query,
        l8_bank,
    )

    return l4_score, l8_score


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
        value
        - normalization["median"]
    ) / (
        1.4826
        * normalization["mad"]
    )


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    category,
    selected_paths,
    normal_dev_paths,
    defect_dev_paths,
    cache,
):
    l4_bank = torch.cat(
        [
            cache[path]["L4"]
            for path in selected_paths
        ],
        dim=0,
    )

    l8_bank = torch.cat(
        [
            cache[path]["L8"]
            for path in selected_paths
        ],
        dim=0,
    )

    normal_l4 = []
    normal_l8 = []

    defect_l4 = []
    defect_l8 = []

    for path in normal_dev_paths:
        l4_score, l8_score = score_sample(
            cache[path]["L4"],
            cache[path]["L8"],
            l4_bank,
            l8_bank,
        )

        normal_l4.append(l4_score)
        normal_l8.append(l8_score)

    for path in defect_dev_paths:
        l4_score, l8_score = score_sample(
            cache[path]["L4"],
            cache[path]["L8"],
            l4_bank,
            l8_bank,
        )

        defect_l4.append(l4_score)
        defect_l8.append(l8_score)

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
        ]
    )

    normal_l8_norm = np.asarray(
        [
            robust_normalize(
                value,
                l8_norm,
            )
            for value in normal_l8
        ]
    )

    defect_l4_norm = np.asarray(
        [
            robust_normalize(
                value,
                l4_norm,
            )
            for value in defect_l4
        ]
    )

    defect_l8_norm = np.asarray(
        [
            robust_normalize(
                value,
                l8_norm,
            )
            for value in defect_l8
        ]
    )

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

    scores = {
        "L4": np.concatenate(
            [
                normal_l4,
                defect_l4,
            ]
        ),
        "L8": np.concatenate(
            [
                normal_l8,
                defect_l8,
            ]
        ),
        "fusion_50_50": np.concatenate(
            [
                normal_fusion,
                defect_fusion,
            ]
        ),
    }

    metrics = {}

    for method, values in scores.items():
        metrics[method] = {
            "AUROC": float(
                roc_auc_score(
                    y_true,
                    values,
                )
            ),
            "AP": float(
                average_precision_score(
                    y_true,
                    values,
                )
            ),
        }

    return metrics


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 78)
    print(
        "DAY 12 — DIVERSITY-PRESERVING "
        "REFERENCE SELECTION"
    )
    print("=" * 78)

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

    results = {
        "seed": SEED,
        "reference_fraction": REFERENCE_FRACTION,
        "top_k": TOP_K,
        "fusion": {
            "L4_weight": L4_WEIGHT,
            "L8_weight": L8_WEIGHT,
        },
        "categories": {},
    }

    # --------------------------------------------------------
    # Feature cache
    # --------------------------------------------------------

    feature_cache = {}

    for category in CATEGORIES:
        print(
            f"\nExtracting features: {category}"
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

        feature_cache[category] = (
            category_cache
        )

    # --------------------------------------------------------
    # Category experiments
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

        cache = feature_cache[
            category
        ]

        n_select = max(
            1,
            int(
                round(
                    len(reference_paths)
                    * REFERENCE_FRACTION
                )
            ),
        )

        # ----------------------------------------------------
        # Random baseline
        # ----------------------------------------------------

        rng = random.Random(
            SEED
        )

        random_paths = rng.sample(
            reference_paths,
            n_select,
        )

        # ----------------------------------------------------
        # Diversity selection
        # ----------------------------------------------------

        descriptors = (
            build_image_descriptors(
                reference_paths,
                cache,
            )
        )

        diverse_paths = (
            select_diverse_images(
                reference_paths,
                descriptors,
                n_select,
                seed=SEED,
            )
        )

        print(
            f"Reference images: "
            f"{len(reference_paths)}"
        )

        print(
            f"Selected images: "
            f"{n_select}"
        )

        print(
            "Evaluating random selection..."
        )

        random_metrics = evaluate(
            category=category,
            selected_paths=random_paths,
            normal_dev_paths=normal_dev_paths,
            defect_dev_paths=defect_dev_paths,
            cache=cache,
        )

        print(
            "Evaluating diversity selection..."
        )

        diverse_metrics = evaluate(
            category=category,
            selected_paths=diverse_paths,
            normal_dev_paths=normal_dev_paths,
            defect_dev_paths=defect_dev_paths,
            cache=cache,
        )

        results["categories"][
            category
        ] = {
            "canonical_reference_count": len(
                reference_paths
            ),
            "selected_reference_count": n_select,
            "random": {
                "metrics": random_metrics,
                "selected_paths": random_paths,
            },
            "diversity": {
                "metrics": diverse_metrics,
                "selected_paths": diverse_paths,
            },
        }

        print(
            f"\n  Random 50/50 fusion: "
            f"{random_metrics['fusion_50_50']['AUROC']:.4f}"
        )

        print(
            f"  Diversity 50/50 fusion: "
            f"{diverse_metrics['fusion_50_50']['AUROC']:.4f}"
        )

        print(
            f"  Difference: "
            f"{diverse_metrics['fusion_50_50']['AUROC'] - random_metrics['fusion_50_50']['AUROC']:+.4f}"
        )

    # --------------------------------------------------------
    # Macro comparison
    # --------------------------------------------------------

    random_macro = []
    diversity_macro = []

    for category in CATEGORIES:
        random_value = results[
            "categories"
        ][category]["random"]["metrics"][
            "fusion_50_50"
        ]["AUROC"]

        diversity_value = results[
            "categories"
        ][category]["diversity"]["metrics"][
            "fusion_50_50"
        ]["AUROC"]

        random_macro.append(
            random_value
        )

        diversity_macro.append(
            diversity_value
        )

    random_macro = float(
        np.mean(random_macro)
    )

    diversity_macro = float(
        np.mean(diversity_macro)
    )

    results["macro"] = {
        "random_fusion_AUROC": random_macro,
        "diversity_fusion_AUROC": diversity_macro,
        "difference": (
            diversity_macro
            - random_macro
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

    print("\n" + "=" * 78)
    print("MACRO RESULT")
    print("=" * 78)

    print(
        f"Random 25% fusion: "
        f"{random_macro:.4f}"
    )

    print(
        f"Diversity 25% fusion: "
        f"{diversity_macro:.4f}"
    )

    print(
        f"Difference: "
        f"{diversity_macro - random_macro:+.4f}"
    )

    print("\nResults saved to:")
    print(OUTPUT_PATH)

    print(
        "\nDAY 12 EXPERIMENT 4 COMPLETE"
    )


if __name__ == "__main__":
    main()