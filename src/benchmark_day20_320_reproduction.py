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
        return path

    return DATA_ROOT / path


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


def build_bank(
    model,
    image_paths,
):
    l4_parts = []
    l8_parts = []

    for path in image_paths:
        l4, l8 = extract_features(
            model,
            path,
        )

        l4_parts.append(l4)
        l8_parts.append(l8)

    return (
        torch.cat(l4_parts, dim=0),
        torch.cat(l8_parts, dim=0),
    )


def score_chunked(
    query,
    reference,
):
    best = []

    for start in range(
        0,
        reference.shape[0],
        CHUNK_SIZE,
    ):
        chunk = reference[
            start:start + CHUNK_SIZE
        ]

        similarity = query @ chunk.T

        distance = torch.sqrt(
            torch.clamp(
                2.0 - 2.0 * similarity,
                min=0.0,
            )
        )

        best.append(
            distance.min(
                dim=1
            ).values
        )

    patch_scores = torch.cat(
        best,
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
    path,
    bank_l4,
    bank_l8,
):
    l4, l8 = extract_features(
        model,
        path,
    )

    l4_score = score_chunked(
        l4,
        bank_l4,
    )

    l8_score = score_chunked(
        l8,
        bank_l8,
    )

    return l4_score, l8_score


def evaluate(
    normal_scores,
    defect_scores,
):
    normal_scores = np.asarray(
        normal_scores
    )

    defect_scores = np.asarray(
        defect_scores
    )

    y_true = np.concatenate(
        [
            np.zeros(
                len(normal_scores)
            ),
            np.ones(
                len(defect_scores)
            ),
        ]
    )

    y_score = np.concatenate(
        [
            normal_scores,
            defect_scores,
        ]
    )

    threshold = np.percentile(
        normal_scores,
        99.0,
    )

    detected = int(
        np.sum(
            defect_scores >= threshold
        )
    )

    false_positive = int(
        np.sum(
            normal_scores >= threshold
        )
    )

    return {
        "threshold": float(threshold),
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
        "detected": detected,
        "defect_count": len(
            defect_scores
        ),
        "detection_rate": (
            detected
            / len(defect_scores)
        ),
        "false_positives": false_positive,
        "normal_count": len(
            normal_scores
        ),
    }


def main():
    print("=" * 80)
    print(
        "VISYN — DAY 20 "
        "320x320 REPRODUCTION"
    )
    print("=" * 80)

    with open(
        PROJECT_ROOT
        / "artifacts"
        / "splits"
        / "normal_splits.json",
        "r",
        encoding="utf-8",
    ) as file:
        normal_splits = json.load(file)

    with open(
        PROJECT_ROOT
        / "artifacts"
        / "splits"
        / "defect_splits.json",
        "r",
        encoding="utf-8",
    ) as file:
        defect_splits = json.load(file)

    model = load_model()

    for category in CATEGORIES:
        print()
        print("#" * 80)
        print(
            f"{category.upper()}"
        )
        print("#" * 80)

        normal = normal_splits[
            "categories"
        ][category]

        defects = defect_splits[
            "categories"
        ][category]

        reference = [
            resolve_path(path)
            for path in normal[
                "reference"
            ]
        ]

        normal_dev = [
            resolve_path(path)
            for path in normal[
                "development"
            ]
        ]

        defect_dev = [
            resolve_path(path)
            for path in defects[
                "development"
            ]
        ]

        print(
            f"Reference images: "
            f"{len(reference)}"
        )

        print(
            "Building 320x320 bank..."
        )

        bank_l4, bank_l8 = build_bank(
            model,
            reference,
        )

        print(
            f"L4 bank: "
            f"{tuple(bank_l4.shape)}"
        )

        print(
            f"L8 bank: "
            f"{tuple(bank_l8.shape)}"
        )

        normal_l4 = []
        normal_l8 = []

        defect_l4 = []
        defect_l8 = []

        print(
            "Scoring normal development..."
        )

        for path in normal_dev:
            l4, l8 = score_image(
                model,
                path,
                bank_l4,
                bank_l8,
            )

            normal_l4.append(l4)
            normal_l8.append(l8)

        print(
            "Scoring defect development..."
        )

        for path in defect_dev:
            l4, l8 = score_image(
                model,
                path,
                bank_l4,
                bank_l8,
            )

            defect_l4.append(l4)
            defect_l8.append(l8)

        normal_l4 = np.asarray(
            normal_l4
        )
        normal_l8 = np.asarray(
            normal_l8
        )

        defect_l4 = np.asarray(
            defect_l4
        )
        defect_l8 = np.asarray(
            defect_l8
        )

        configurations = {
            "L4_ONLY": (
                normal_l4,
                defect_l4,
            ),
            "L8_ONLY": (
                normal_l8,
                defect_l8,
            ),
            "50_50": (
                0.5 * normal_l4
                + 0.5 * normal_l8,
                0.5 * defect_l4
                + 0.5 * defect_l8,
            ),
        }

        print()
        print(
            "RESULTS"
        )
        print("-" * 80)

        for name, (
            normal_scores,
            defect_scores,
        ) in configurations.items():
            result = evaluate(
                normal_scores,
                defect_scores,
            )

            print(
                f"{name:<10} "
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
                f"{result['normal_count']} "
                f"Threshold="
                f"{result['threshold']:.6f}"
            )

        print()
        print(
            "Score distribution:"
        )

        print(
            f"L4 normal  mean="
            f"{normal_l4.mean():.6f} "
            f"max="
            f"{normal_l4.max():.6f}"
        )

        print(
            f"L4 defect  mean="
            f"{defect_l4.mean():.6f} "
            f"max="
            f"{defect_l4.max():.6f}"
        )

        print(
            f"L8 normal  mean="
            f"{normal_l8.mean():.6f} "
            f"max="
            f"{normal_l8.max():.6f}"
        )

        print(
            f"L8 defect  mean="
            f"{defect_l8.mean():.6f} "
            f"max="
            f"{defect_l8.max():.6f}"
        )

    print()
    print("=" * 80)
    print(
        "DAY 20 REPRODUCTION COMPLETE"
    )


if __name__ == "__main__":
    main()