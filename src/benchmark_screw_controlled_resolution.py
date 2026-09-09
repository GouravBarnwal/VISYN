import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    mobilenet_v3_small,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

CATEGORY = "screw"

RESOLUTIONS = [224, 320]

TOP_K = 5
PERCENTILE = 99.0

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


def resolve_split_path(path_string):
    path = Path(path_string)

    if path.is_absolute():
        return path

    return DATA_ROOT / path


def preprocess_image(image_path, image_size):
    image = Image.open(image_path).convert("RGB")

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
def extract_features(model, image_path, image_size):
    batch = preprocess_image(
        image_path,
        image_size,
    )

    x = batch

    l4 = None
    l8 = None

    for index, layer in enumerate(model.features):
        x = layer(x)

        if index == 4:
            l4 = x

        elif index == 8:
            l8 = x

    if l4 is None or l8 is None:
        raise RuntimeError("Failed to capture L4/L8 features.")

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


def top5_score(query, reference):
    similarity = query @ reference.T

    distance = torch.sqrt(
        torch.clamp(
            2.0 - 2.0 * similarity,
            min=0.0,
        )
    )

    patch_scores = distance.min(dim=1).values

    top_k = min(
        TOP_K,
        patch_scores.numel(),
    )

    return (
        torch.topk(
            patch_scores,
            k=top_k,
        )
        .values.mean()
        .item()
    )


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

    anomaly_score = 0.5 * l4_score + 0.5 * l8_score

    return {
        "l4": l4_score,
        "l8": l8_score,
        "fusion": anomaly_score,
    }


def auc_metrics(
    normal_scores,
    defect_scores,
):
    from sklearn.metrics import (
        average_precision_score,
        roc_auc_score,
    )

    y_true = [0] * len(normal_scores) + [1] * len(defect_scores)

    y_score = list(normal_scores) + list(defect_scores)

    return {
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
    }


def load_locked_splits():
    normal_path = PROJECT_ROOT / "artifacts" / "splits" / "normal_splits.json"

    defect_path = PROJECT_ROOT / "artifacts" / "splits" / "defect_splits.json"

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

    return normal_splits, defect_splits


def resolve_paths(paths):
    resolved = []

    for path in paths:
        resolved_path = resolve_split_path(path)

        if not resolved_path.exists():
            raise FileNotFoundError(f"Split image does not exist: " f"{resolved_path}")

        resolved.append(resolved_path)

    return resolved


def main():
    model = load_model()

    normal_splits, defect_splits = load_locked_splits()

    category_normal = normal_splits["categories"][CATEGORY]

    category_defect = defect_splits["categories"][CATEGORY]

    reference_images = resolve_paths(category_normal["reference"])

    normal_images = resolve_paths(category_normal["development"])

    defect_images = resolve_paths(category_defect["development"])

    print("=" * 72)
    print("VISIONFORGE — CONTROLLED SCREW " "RESOLUTION EVALUATION")
    print("=" * 72)

    print(f"Dataset root: {DATA_ROOT}")

    print(f"Reference images: " f"{len(reference_images)}")

    print(f"Normal development: " f"{len(normal_images)}")

    print(f"Defect development: " f"{len(defect_images)}")

    print()

    results = {}

    for image_size in RESOLUTIONS:
        print("=" * 72)
        print(f"RESOLUTION: " f"{image_size}x{image_size}")
        print("=" * 72)

        start = time.perf_counter()

        reference_l4, reference_l8 = build_reference_bank(
            model,
            reference_images,
            image_size,
        )

        bank_time = time.perf_counter() - start

        print(f"Reference bank built in " f"{bank_time:.2f}s")

        normal_scores = []

        for image_path in normal_images:
            result = score_image(
                model,
                image_path,
                image_size,
                reference_l4,
                reference_l8,
            )

            normal_scores.append(result["fusion"])

        normal_scores = np.asarray(
            normal_scores,
            dtype=np.float64,
        )

        threshold = float(
            np.percentile(
                normal_scores,
                PERCENTILE,
            )
        )

        print(f"P99 threshold: " f"{threshold:.6f}")

        defect_scores = []

        for image_path in defect_images:
            result = score_image(
                model,
                image_path,
                image_size,
                reference_l4,
                reference_l8,
            )

            defect_scores.append(result["fusion"])

        defect_scores = np.asarray(
            defect_scores,
            dtype=np.float64,
        )

        detected = int(np.sum(defect_scores >= threshold))

        normal_false_positives = int(np.sum(normal_scores >= threshold))

        metrics = auc_metrics(
            normal_scores,
            defect_scores,
        )

        print(
            f"Defects detected: "
            f"{detected}/"
            f"{len(defect_scores)} "
            f"("
            f"{detected / len(defect_scores) * 100:.1f}%"
            f")"
        )

        print(
            f"Normal false positives: "
            f"{normal_false_positives}/"
            f"{len(normal_scores)}"
        )

        print(f"AUROC: " f"{metrics['auroc']:.4f}")

        print(f"AP: " f"{metrics['average_precision']:.4f}")

        results[str(image_size)] = {
            "resolution": image_size,
            "threshold": threshold,
            "normal": {
                "count": len(normal_scores),
                "mean": float(np.mean(normal_scores)),
                "median": float(np.median(normal_scores)),
                "p95": float(
                    np.percentile(
                        normal_scores,
                        95,
                    )
                ),
                "max": float(np.max(normal_scores)),
                "false_positives": (normal_false_positives),
                "false_positive_rate": (normal_false_positives / len(normal_scores)),
            },
            "defects": {
                "count": len(defect_scores),
                "detected": detected,
                "detection_rate": (detected / len(defect_scores)),
                "mean": float(np.mean(defect_scores)),
                "median": float(np.median(defect_scores)),
                "min": float(np.min(defect_scores)),
                "max": float(np.max(defect_scores)),
            },
            "metrics": metrics,
            "reference_bank": {
                "reference_images": len(reference_images),
                "L4_shape": list(reference_l4.shape),
                "L8_shape": list(reference_l8.shape),
            },
        }

        print()

    output = {
        "category": CATEGORY,
        "resolutions": RESOLUTIONS,
        "reference_images": len(reference_images),
        "normal_development_images": len(normal_images),
        "defect_development_images": len(defect_images),
        "reference_source": ("normal_splits.json"),
        "defect_source": ("defect_splits.json"),
        "threshold_method": ("P99 normal development"),
        "top_k": TOP_K,
        "fusion": "0.5 L4 + 0.5 L8",
        "results": results,
    }

    output_path = (
        PROJECT_ROOT
        / "artifacts"
        / "evaluation"
        / "day19_screw_controlled_resolution.json"
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

    print("=" * 72)
    print(f"Saved to: {output_path}")
    print("CONTROLLED RESOLUTION " "EVALUATION: PASS")


if __name__ == "__main__":
    main()
