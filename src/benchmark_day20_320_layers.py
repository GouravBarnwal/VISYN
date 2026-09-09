import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    mobilenet_v3_small,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

CATEGORIES = (
    "capsule",
    "screw",
)

IMAGE_SIZE = 320
TOP_K = 5
CHUNK_SIZE = 8192

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
    model = mobilenet_v3_small(weights=weights)
    model.eval()
    return model


def resolve_path(path_string):
    path = Path(path_string)

    if path.is_absolute():
        resolved = path
    else:
        resolved = DATA_ROOT / path

    if not resolved.exists():
        raise FileNotFoundError(
            f"Image does not exist: {resolved}"
        )

    return resolved


def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")

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

    tensor = torch.from_numpy(array)
    tensor = tensor.permute(2, 0, 1)
    tensor = tensor.unsqueeze(0)

    return (tensor - MEAN) / STD


@torch.no_grad()
def extract_features(model, image_path):
    x = preprocess_image(image_path)

    l4 = None
    l8 = None

    for index, layer in enumerate(model.features):
        x = layer(x)

        if index == 4:
            l4 = x

        elif index == 8:
            l8 = x

    if l4 is None or l8 is None:
        raise RuntimeError(
            "Failed to capture L4/L8 features."
        )

    l4 = l4.permute(
        0,
        2,
        3,
        1,
    ).reshape(
        -1,
        l4.shape[1],
    )

    l8 = l8.permute(
        0,
        2,
        3,
        1,
    ).reshape(
        -1,
        l8.shape[1],
    )

    l4 = F.normalize(
        l4,
        p=2,
        dim=1,
    )

    l8 = F.normalize(
        l8,
        p=2,
        dim=1,
    )

    return l4, l8


def build_reference_bank(
    model,
    image_paths,
):
    l4_parts = []
    l8_parts = []

    for image_path in image_paths:
        l4, l8 = extract_features(
            model,
            image_path,
        )

        l4_parts.append(l4)
        l8_parts.append(l8)

    return (
        torch.cat(l4_parts, dim=0),
        torch.cat(l8_parts, dim=0),
    )


def layer_score(
    query,
    reference,
):
    best_distances = []

    for start in range(
        0,
        reference.shape[0],
        CHUNK_SIZE,
    ):
        reference_chunk = reference[
            start:start + CHUNK_SIZE
        ]

        similarity = query @ reference_chunk.T

        distance = torch.sqrt(
            torch.clamp(
                2.0 - 2.0 * similarity,
                min=0.0,
            )
        )

        best_distances.append(
            distance.min(dim=1).values
        )

    patch_scores = torch.cat(
        best_distances,
        dim=0,
    )

    return torch.topk(
        patch_scores,
        k=min(
            TOP_K,
            patch_scores.numel(),
        ),
    ).values.mean().item()


def score_image(
    model,
    image_path,
    reference_l4,
    reference_l8,
):
    l4, l8 = extract_features(
        model,
        image_path,
    )

    return {
        "L4": layer_score(
            l4,
            reference_l4,
        ),
        "L8": layer_score(
            l8,
            reference_l8,
        ),
    }


def fuse_scores(
    l4,
    l8,
    weight_l4,
):
    weight_l8 = 1.0 - weight_l4

    return (
        weight_l4 * l4
        + weight_l8 * l8
    )


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

    y_scores = np.concatenate(
        [
            normal_scores,
            defect_scores,
        ]
    )

    return {
        "auroc": float(
            roc_auc_score(
                y_true,
                y_scores,
            )
        ),
        "average_precision": float(
            average_precision_score(
                y_true,
                y_scores,
            )
        ),
    }


def evaluate(
    normal_scores,
    defect_scores,
):
    threshold = float(
        np.percentile(
            normal_scores,
            99.0,
        )
    )

    normal_fp = int(
        np.sum(
            normal_scores >= threshold
        )
    )

    detected = int(
        np.sum(
            defect_scores >= threshold
        )
    )

    metrics = calculate_metrics(
        normal_scores,
        defect_scores,
    )

    return {
        "threshold": threshold,
        "auroc": metrics["auroc"],
        "average_precision": metrics[
            "average_precision"
        ],
        "detected": detected,
        "defect_count": len(
            defect_scores
        ),
        "detection_rate": (
            detected / len(defect_scores)
        ),
        "false_positives": normal_fp,
        "normal_count": len(
            normal_scores
        ),
        "false_positive_rate": (
            normal_fp / len(normal_scores)
        ),
    }


def evaluate_layer_configuration(
    normal_l4,
    normal_l8,
    defect_l4,
    defect_l8,
    weight_l4,
):
    normal_scores = fuse_scores(
        normal_l4,
        normal_l8,
        weight_l4,
    )

    defect_scores = fuse_scores(
        defect_l4,
        defect_l8,
        weight_l4,
    )

    return evaluate(
        normal_scores,
        defect_scores,
    )


def main():
    print("=" * 80)
    print(
        "VISIONFORGE — DAY 20 "
        "320x320 FEATURE-LAYER TUNING"
    )
    print("=" * 80)

    print(
        f"Categories: "
        f"{', '.join(CATEGORIES)}"
    )
    print(
        f"Resolution: {IMAGE_SIZE}x{IMAGE_SIZE}"
    )
    print(
        f"TOP-K: {TOP_K}"
    )

    normal_path = (
        PROJECT_ROOT
        / "artifacts"
        / "splits"
        / "normal_splits.json"
    )

    defect_path = (
        PROJECT_ROOT
        / "artifacts"
        / "splits"
        / "defect_splits.json"
    )

    with open(
        normal_path,
        "r",
        encoding="utf-8",
    ) as file:
        normal_splits = json.load(file)

    with open(
        defect_path,
        "r",
        encoding="utf-8",
    ) as file:
        defect_splits = json.load(file)

    model = load_model()

    results = {}

    configurations = {
        "L4_ONLY": 1.00,
        "L8_ONLY": 0.00,
        "L4_75_L8_25": 0.75,
        "L4_60_L8_40": 0.60,
        "L4_50_L8_50": 0.50,
        "L4_40_L8_60": 0.40,
        "L4_25_L8_75": 0.25,
    }

    for category in CATEGORIES:
        print()
        print("#" * 80)
        print(
            f"CATEGORY: {category.upper()}"
        )
        print("#" * 80)

        normal_category = (
            normal_splits[
                "categories"
            ][category]
        )

        defect_category = (
            defect_splits[
                "categories"
            ][category]
        )

        reference_images = [
            resolve_path(path)
            for path in normal_category[
                "reference"
            ]
        ]

        normal_images = [
            resolve_path(path)
            for path in normal_category[
                "development"
            ]
        ]

        defect_images = [
            resolve_path(path)
            for path in defect_category[
                "development"
            ]
        ]

        print(
            f"Reference: {len(reference_images)}"
        )
        print(
            f"Normal development: "
            f"{len(normal_images)}"
        )
        print(
            f"Defect development: "
            f"{len(defect_images)}"
        )

        print()
        print(
            "Building 320x320 reference bank..."
        )

        reference_l4, reference_l8 = (
            build_reference_bank(
                model,
                reference_images,
            )
        )

        print(
            f"L4 bank: "
            f"{tuple(reference_l4.shape)}"
        )
        print(
            f"L8 bank: "
            f"{tuple(reference_l8.shape)}"
        )

        print()
        print(
            "Scoring normal development..."
        )

        normal_layer_scores = [
            score_image(
                model,
                image_path,
                reference_l4,
                reference_l8,
            )
            for image_path in normal_images
        ]

        print(
            "Scoring defect development..."
        )

        defect_layer_scores = [
            score_image(
                model,
                image_path,
                reference_l4,
                reference_l8,
            )
            for image_path in defect_images
        ]

        normal_l4 = np.asarray(
            [
                item["L4"]
                for item in normal_layer_scores
            ],
            dtype=np.float64,
        )

        normal_l8 = np.asarray(
            [
                item["L8"]
                for item in normal_layer_scores
            ],
            dtype=np.float64,
        )

        defect_l4 = np.asarray(
            [
                item["L4"]
                for item in defect_layer_scores
            ],
            dtype=np.float64,
        )

        defect_l8 = np.asarray(
            [
                item["L8"]
                for item in defect_layer_scores
            ],
            dtype=np.float64,
        )

        category_results = {}

        for name, weight_l4 in configurations.items():
            result = evaluate_layer_configuration(
                normal_l4,
                normal_l8,
                defect_l4,
                defect_l8,
                weight_l4,
            )

            category_results[name] = result

            print(
                f"{name:<16} "
                f"AUROC={result['auroc']:.4f} "
                f"AP={result['average_precision']:.4f} "
                f"Detection="
                f"{result['detected']}/"
                f"{result['defect_count']} "
                f"("
                f"{result['detection_rate'] * 100:.1f}%"
                f") "
                f"FP="
                f"{result['false_positives']}/"
                f"{result['normal_count']}"
            )

        results[category] = category_results

    output = {
        "experiment": (
            "Day 20 320x320 "
            "feature-layer tuning"
        ),
        "resolution": IMAGE_SIZE,
        "top_k": TOP_K,
        "categories": list(CATEGORIES),
        "configurations": configurations,
        "reference_split": (
            "normal_splits.json"
        ),
        "defect_split": (
            "defect_splits.json "
            "development only"
        ),
        "results": results,
    }

    output_path = (
        PROJECT_ROOT
        / "artifacts"
        / "evaluation"
        / "day20_320_layers.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
        )

    print()
    print("=" * 80)
    print(
        f"Saved to: {output_path}"
    )
    print(
        "DAY 20 FEATURE-LAYER "
        "EXPERIMENT: PASS"
    )


if __name__ == "__main__":
    main()