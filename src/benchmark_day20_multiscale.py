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

RESOLUTIONS = (
    224,
    320,
)

TOP_K = 5

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


def preprocess_image(
    image_path,
    image_size,
):
    image = Image.open(
        image_path
    ).convert("RGB")

    image = image.resize(
        (image_size, image_size),
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
    image_size,
):
    x = preprocess_image(
        image_path,
        image_size,
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
    image_size,
):
    l4_parts = []
    l8_parts = []

    for image_path in image_paths:
        l4, l8 = extract_features(
            model,
            image_path,
            image_size,
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


def top5_score(
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

    patch_scores = distance.min(
        dim=1
    ).values

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
    image_size,
    reference_l4,
    reference_l8,
):
    l4, l8 = extract_features(
        model,
        image_path,
        image_size,
    )

    l4_score = top5_score(
        l4,
        reference_l4,
    )

    l8_score = top5_score(
        l8,
        reference_l8,
    )

    return (
        0.5 * l4_score
        + 0.5 * l8_score
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


def combine_scores(
    scores_224,
    scores_320,
    method,
):
    if method == "224_ONLY":
        return scores_224

    if method == "320_ONLY":
        return scores_320

    if method == "25_75":
        return (
            0.25 * scores_224
            + 0.75 * scores_320
        )

    if method == "50_50":
        return (
            0.50 * scores_224
            + 0.50 * scores_320
        )

    if method == "75_25":
        return (
            0.75 * scores_224
            + 0.25 * scores_320
        )

    if method == "MAX":
        return np.maximum(
            scores_224,
            scores_320,
        )

    raise ValueError(
        f"Unknown method: {method}"
    )


def evaluate_method(
    normal_scores,
    defect_scores,
):
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

    metric = calculate_metrics(
        normal_scores,
        defect_scores,
    )

    return {
        "threshold": threshold,
        "auroc": metric["auroc"],
        "average_precision": metric[
            "average_precision"
        ],
        "detected": detected,
        "defect_count": len(
            defect_scores
        ),
        "detection_rate": (
            detected
            / len(defect_scores)
        ),
        "false_positives": (
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


def main():
    print("=" * 80)
    print(
        "VISYN — DAY 20 "
        "MULTI-SCALE SCORING"
    )
    print("=" * 80)

    print(
        f"Categories: "
        f"{', '.join(CATEGORIES)}"
    )

    print(
        "Resolutions: 224 and 320"
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

    methods = (
        "224_ONLY",
        "320_ONLY",
        "25_75",
        "50_50",
        "75_25",
        "MAX",
    )

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
            f"Reference: "
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

        scores = {}

        for image_size in RESOLUTIONS:
            print()
            print(
                f"Building "
                f"{image_size}x{image_size} "
                f"reference bank..."
            )

            reference_l4, reference_l8 = (
                build_reference_bank(
                    model,
                    reference_images,
                    image_size,
                )
            )

            normal_scores = []

            for image_path in normal_images:
                normal_scores.append(
                    score_image(
                        model,
                        image_path,
                        image_size,
                        reference_l4,
                        reference_l8,
                    )
                )

            defect_scores = []

            for image_path in defect_images:
                defect_scores.append(
                    score_image(
                        model,
                        image_path,
                        image_size,
                        reference_l4,
                        reference_l8,
                    )
                )

            scores[
                image_size
            ] = {
                "normal": np.asarray(
                    normal_scores,
                    dtype=np.float64,
                ),
                "defect": np.asarray(
                    defect_scores,
                    dtype=np.float64,
                ),
            }

        category_results = {}

        for method in methods:
            normal_scores = combine_scores(
                scores[224]["normal"],
                scores[320]["normal"],
                method,
            )

            defect_scores = combine_scores(
                scores[224]["defect"],
                scores[320]["defect"],
                method,
            )

            result = evaluate_method(
                normal_scores,
                defect_scores,
            )

            category_results[
                method
            ] = result

            print(
                f"{method:<10} "
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

        results[category] = (
            category_results
        )

    output = {
        "experiment": (
            "Day 20 multi-scale "
            "224 + 320 scoring"
        ),
        "categories": list(
            CATEGORIES
        ),
        "resolutions": [
            224,
            320,
        ],
        "top_k": TOP_K,
        "methods": list(
            methods
        ),
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
        / "day20_multiscale.json"
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
        "DAY 20 MULTI-SCALE "
        "EXPERIMENT: PASS"
    )


if __name__ == "__main__":
    main()