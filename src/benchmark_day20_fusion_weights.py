import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
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

IMAGE_SIZE = 224
TOP_K = 5

L4_WEIGHTS = (
    1.00,
    0.75,
    0.60,
    0.50,
    0.40,
    0.25,
    0.00,
)

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
    image = Image.open(
        image_path
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

    tensor = torch.from_numpy(array)
    tensor = tensor.permute(2, 0, 1)
    tensor = tensor.unsqueeze(0)

    return (tensor - MEAN) / STD


@torch.no_grad()
def extract_features(
    model,
    image_path,
):
    x = preprocess_image(
        image_path
    )

    l4 = None
    l8 = None

    for index, layer in enumerate(
        model.features
    ):
        x = layer(x)

        if index == 4:
            l4 = x

        elif index == 8:
            l8 = x

    if l4 is None or l8 is None:
        raise RuntimeError(
            "Failed to capture L4/L8."
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
        torch.cat(
            l4_parts,
            dim=0,
        ),
        torch.cat(
            l8_parts,
            dim=0,
        ),
    )


def patch_scores(
    query,
    reference,
):
    similarity = query @ reference.T

    distance = torch.sqrt(
        torch.clamp(
            2.0 - 2.0 * similarity,
            min=0.0,
        )
    )

    return distance.min(
        dim=1
    ).values


def top5_score(
    query,
    reference,
):
    scores = patch_scores(
        query,
        reference,
    )

    return torch.topk(
        scores,
        k=min(
            TOP_K,
            scores.numel(),
        ),
    ).values.mean().item()


def extract_layer_scores(
    model,
    image_path,
    reference_l4,
    reference_l8,
):
    l4, l8 = extract_features(
        model,
        image_path,
    )

    return (
        top5_score(
            l4,
            reference_l4,
        ),
        top5_score(
            l8,
            reference_l8,
        ),
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


def main():
    print("=" * 80)
    print(
        "VISIONFORGE — DAY 20 "
        "L4/L8 FUSION WEIGHT TUNING"
    )
    print("=" * 80)

    print(
        f"Categories: "
        f"{', '.join(CATEGORIES)}"
    )

    print(
        f"Image size: "
        f"{IMAGE_SIZE}x{IMAGE_SIZE}"
    )

    print(
        f"TOP-K: {TOP_K}"
    )

    print()

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

    for category in CATEGORIES:
        print()
        print("#" * 80)
        print(
            f"CATEGORY: "
            f"{category.upper()}"
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
            f"Reference images: "
            f"{len(reference_images)}"
        )

        print(
            f"Normal development: "
            f"{len(normal_images)}"
        )

        print(
            f"Defect development: "
            f"{len(defect_images)}"
        )

        print(
            "Building reference bank..."
        )

        reference_l4, reference_l8 = (
            build_reference_bank(
                model,
                reference_images,
            )
        )

        normal_layer_scores = [
            extract_layer_scores(
                model,
                image_path,
                reference_l4,
                reference_l8,
            )
            for image_path in normal_images
        ]

        defect_layer_scores = [
            extract_layer_scores(
                model,
                image_path,
                reference_l4,
                reference_l8,
            )
            for image_path in defect_images
        ]

        category_results = {}

        for l4_weight in L4_WEIGHTS:
            l8_weight = 1.0 - l4_weight

            normal_scores = np.asarray(
                [
                    (
                        l4_weight * scores[0]
                        + l8_weight * scores[1]
                    )
                    for scores in normal_layer_scores
                ],
                dtype=np.float64,
            )

            defect_scores = np.asarray(
                [
                    (
                        l4_weight * scores[0]
                        + l8_weight * scores[1]
                    )
                    for scores in defect_layer_scores
                ],
                dtype=np.float64,
            )

            threshold = float(
                np.percentile(
                    normal_scores,
                    99.0,
                )
            )

            false_positives = int(
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

            label = (
                f"{int(l4_weight * 100)}"
                f"/"
                f"{int(l8_weight * 100)}"
            )

            category_results[label] = {
                "l4_weight": l4_weight,
                "l8_weight": l8_weight,
                "threshold": threshold,
                "auroc": metrics["auroc"],
                "average_precision": (
                    metrics[
                        "average_precision"
                    ]
                ),
                "defect_detection": detected,
                "defect_count": len(
                    defect_scores
                ),
                "detection_rate": (
                    detected
                    / len(defect_scores)
                ),
                "normal_false_positives": (
                    false_positives
                ),
                "normal_count": len(
                    normal_scores
                ),
                "false_positive_rate": (
                    false_positives
                    / len(normal_scores)
                ),
            }

            print(
                f"{label}  "
                f"AUROC={metrics['auroc']:.4f}  "
                f"AP={metrics['average_precision']:.4f}  "
                f"Detection="
                f"{detected}/"
                f"{len(defect_scores)}  "
                f"FP="
                f"{false_positives}/"
                f"{len(normal_scores)}"
            )

        results[category] = category_results

    output = {
        "experiment": (
            "Day 20 L4/L8 fusion weight tuning"
        ),
        "categories": list(
            CATEGORIES
        ),
        "image_size": IMAGE_SIZE,
        "top_k": TOP_K,
        "weights": [
            {
                "l4": weight,
                "l8": 1.0 - weight,
            }
            for weight in L4_WEIGHTS
        ],
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
        / "day20_fusion_weight_tuning.json"
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
        "DAY 20 FUSION WEIGHT "
        "EXPERIMENT: PASS"
    )


if __name__ == "__main__":
    main()