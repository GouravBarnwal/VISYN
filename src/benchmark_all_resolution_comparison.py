import json
import time
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
    "bottle",
    "hazelnut",
    "cable",
    "capsule",
    "screw",
    "metal_nut",
)

RESOLUTIONS = (224, 320)

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


def resolve_path(path_string):
    path = Path(path_string)

    if path.is_absolute():
        resolved = path
    else:
        resolved = DATA_ROOT / path

    if not resolved.exists():
        raise FileNotFoundError(f"Image does not exist: {resolved}")

    return resolved


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
def extract_features(
    model,
    image_path,
    image_size,
):
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

    start = time.perf_counter()

    for image_path in image_paths:
        l4, l8 = extract_features(
            model,
            image_path,
            image_size,
        )

        l4_parts.append(l4)
        l8_parts.append(l8)

    l4_bank = torch.cat(
        l4_parts,
        dim=0,
    )

    l8_bank = torch.cat(
        l8_parts,
        dim=0,
    )

    elapsed = time.perf_counter() - start

    return (
        l4_bank,
        l8_bank,
        elapsed,
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
    start = time.perf_counter()

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

    fusion = 0.5 * l4_score + 0.5 * l8_score

    elapsed = time.perf_counter() - start

    return {
        "l4": l4_score,
        "l8": l8_score,
        "fusion": fusion,
        "latency_ms": elapsed * 1000.0,
    }


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
    print("VISIONFORGE — ALL-CATEGORY " "224 vs 320 RESOLUTION COMPARISON")
    print("=" * 80)
    print(f"Dataset root: {DATA_ROOT}")
    print(f"Categories: {len(CATEGORIES)}")
    print()

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

    model = load_model()

    results = {}

    for category in CATEGORIES:
        print()
        print("#" * 80)
        print(f"CATEGORY: {category.upper()}")
        print("#" * 80)

        normal_category = normal_splits["categories"][category]

        defect_category = defect_splits["categories"][category]

        reference_images = [resolve_path(path) for path in normal_category["reference"]]

        normal_images = [resolve_path(path) for path in normal_category["development"]]

        defect_images = [resolve_path(path) for path in defect_category["development"]]

        print(f"Reference: {len(reference_images)}")
        print(f"Normal development: " f"{len(normal_images)}")
        print(f"Defect development: " f"{len(defect_images)}")

        results[category] = {}

        for image_size in RESOLUTIONS:
            print()
            print(f"--- {image_size}x{image_size} ---")

            reference_l4, reference_l8, bank_time = build_reference_bank(
                model,
                reference_images,
                image_size,
            )

            normal_scores = []
            normal_latencies = []

            for image_path in normal_images:
                result = score_image(
                    model,
                    image_path,
                    image_size,
                    reference_l4,
                    reference_l8,
                )

                normal_scores.append(result["fusion"])

                normal_latencies.append(result["latency_ms"])

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

            defect_scores = []
            defect_latencies = []

            for image_path in defect_images:
                result = score_image(
                    model,
                    image_path,
                    image_size,
                    reference_l4,
                    reference_l8,
                )

                defect_scores.append(result["fusion"])

                defect_latencies.append(result["latency_ms"])

            defect_scores = np.asarray(
                defect_scores,
                dtype=np.float64,
            )

            metrics = calculate_metrics(
                normal_scores,
                defect_scores,
            )

            false_positives = int(np.sum(normal_scores >= threshold))

            detected = int(np.sum(defect_scores >= threshold))

            all_latencies = normal_latencies + defect_latencies

            result = {
                "resolution": image_size,
                "threshold": threshold,
                "reference_images": len(reference_images),
                "reference_bank": {
                    "L4_shape": list(reference_l4.shape),
                    "L8_shape": list(reference_l8.shape),
                    "L4_MB": (reference_l4.numel() * 4 / 1024 / 1024),
                    "L8_MB": (reference_l8.numel() * 4 / 1024 / 1024),
                },
                "normal": {
                    "count": len(normal_scores),
                    "mean": float(np.mean(normal_scores)),
                    "p95": float(
                        np.percentile(
                            normal_scores,
                            95,
                        )
                    ),
                    "max": float(np.max(normal_scores)),
                    "false_positives": (false_positives),
                    "false_positive_rate": (false_positives / len(normal_scores)),
                },
                "defects": {
                    "count": len(defect_scores),
                    "detected": detected,
                    "detection_rate": (detected / len(defect_scores)),
                    "mean": float(np.mean(defect_scores)),
                    "min": float(np.min(defect_scores)),
                    "max": float(np.max(defect_scores)),
                },
                "metrics": metrics,
                "latency": {
                    "reference_build_seconds": (bank_time),
                    "mean_ms": float(np.mean(all_latencies)),
                    "p50_ms": float(
                        np.percentile(
                            all_latencies,
                            50,
                        )
                    ),
                    "p95_ms": float(
                        np.percentile(
                            all_latencies,
                            95,
                        )
                    ),
                },
            }

            results[category][str(image_size)] = result

            print(f"Threshold: {threshold:.6f}")
            print(f"AUROC: " f"{metrics['auroc']:.4f}")
            print(f"AP: " f"{metrics['average_precision']:.4f}")
            print(
                f"Defect detection: "
                f"{detected}/{len(defect_scores)} "
                f"("
                f"{detected / len(defect_scores) * 100:.1f}%"
                f")"
            )
            print(f"Normal FP: " f"{false_positives}/" f"{len(normal_scores)}")
            print(f"Mean latency: " f"{np.mean(all_latencies):.2f} ms")
            print(f"P95 latency: " f"{np.percentile(all_latencies, 95):.2f} ms")

    output = {
        "experiment": ("controlled all-category " "resolution comparison"),
        "resolutions": list(RESOLUTIONS),
        "categories": list(CATEGORIES),
        "reference_split": ("normal_splits.json"),
        "defect_split": ("defect_splits.json " "development only"),
        "threshold_method": ("P99 normal development"),
        "scoring": {
            "feature_layers": [
                "L4",
                "L8",
            ],
            "top_k": TOP_K,
            "fusion": ("0.5 L4 + 0.5 L8"),
        },
        "results": results,
    }

    output_path = (
        PROJECT_ROOT
        / "artifacts"
        / "evaluation"
        / "day19_all_resolution_comparison.json"
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
    print(f"Saved to: {output_path}")
    print("ALL-CATEGORY RESOLUTION " "COMPARISON: PASS")


if __name__ == "__main__":
    main()
