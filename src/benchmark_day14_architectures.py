import json
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.models as models
from PIL import Image
from torchvision.models import MobileNet_V3_Small_Weights


DATA_ROOT = Path("data")
ARTIFACT_ROOT = Path("artifacts")

CATEGORIES = [
    "bottle",
    "hazelnut",
    "cable",
    "capsule",
    "screw",
    "metal_nut",
]

NORMAL_SPLIT_PATH = (
    ARTIFACT_ROOT / "splits" / "normal_splits.json"
)

OUTPUT_PATH = (
    ARTIFACT_ROOT
    / "evaluation"
    / "day14_architecture_benchmark.json"
)

DEVICE = torch.device("cpu")

L4_INDEX = 4
L8_INDEX = 8
TOP_K = 5

WARMUP_RUNS = 3
TIMED_RUNS = 20


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_image_path(relative_path):
    path = Path(relative_path)

    if path.is_absolute():
        return path

    return DATA_ROOT / path


def load_model():
    weights = MobileNet_V3_Small_Weights.DEFAULT

    model = models.mobilenet_v3_small(
        weights=weights
    )

    model.eval()
    model.to(DEVICE)

    return model


def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")

    transform = MobileNet_V3_Small_Weights.DEFAULT.transforms()

    tensor = transform(image).unsqueeze(0)

    return tensor.to(DEVICE)


@torch.no_grad()
def extract_layers(model, image_tensor):
    x = image_tensor

    outputs = {}

    for index, layer in enumerate(model.features):
        x = layer(x)

        if index in (L4_INDEX, L8_INDEX):
            outputs[index] = x

    return outputs


def spatial_features(feature_map):
    feature_map = feature_map.squeeze(0)

    return feature_map.permute(
        1,
        2,
        0,
    ).reshape(
        -1,
        feature_map.shape[0],
    )


def normalize_rows(features):
    return features / (
        torch.linalg.norm(
            features,
            dim=1,
            keepdim=True,
        )
        + 1e-12
    )


def build_reference_bank(
    model,
    reference_paths,
):
    l4_bank = []
    l8_bank = []

    for relative_path in reference_paths:
        image_path = resolve_image_path(
            relative_path
        )

        image_tensor = preprocess_image(
            image_path
        )

        outputs = extract_layers(
            model,
            image_tensor,
        )

        l4 = normalize_rows(
            spatial_features(
                outputs[L4_INDEX]
            )
        )

        l8 = normalize_rows(
            spatial_features(
                outputs[L8_INDEX]
            )
        )

        l4_bank.append(
            l4.cpu()
        )

        l8_bank.append(
            l8.cpu()
        )

    return {
        "L4": torch.cat(
            l4_bank,
            dim=0,
        ),
        "L8": torch.cat(
            l8_bank,
            dim=0,
        ),
    }


def score_layer(
    query_features,
    reference_bank,
):
    query_features = normalize_rows(
        query_features
    )

    distances = torch.cdist(
        query_features,
        reference_bank,
    )

    nearest = distances.min(
        dim=1
    ).values

    k = min(
        TOP_K,
        nearest.numel(),
    )

    return torch.topk(
        nearest,
        k=k,
        largest=True,
    ).values.mean().item()


def tensor_memory_mb(tensor):
    return (
        tensor.numel()
        * tensor.element_size()
        / (1024 ** 2)
    )


def median(values):
    return float(
        np.median(
            np.asarray(values)
        )
    )


def benchmark_category(
    model,
    category,
    reference_paths,
    query_paths,
):
    print()
    print("=" * 70)
    print(f"CATEGORY: {category}")
    print("=" * 70)

    print(
        f"Reference images: {len(reference_paths)}"
    )

    print(
        f"Benchmark query images: "
        f"{len(query_paths)}"
    )

    reference_bank = build_reference_bank(
        model,
        reference_paths,
    )

    print(
        f"L4 bank shape: "
        f"{tuple(reference_bank['L4'].shape)}"
    )

    print(
        f"L8 bank shape: "
        f"{tuple(reference_bank['L8'].shape)}"
    )

    print(
        f"L4 bank memory: "
        f"{tensor_memory_mb(reference_bank['L4']):.2f} MB"
    )

    print(
        f"L8 bank memory: "
        f"{tensor_memory_mb(reference_bank['L8']):.2f} MB"
    )

    image_times = []
    extraction_times = []
    l4_scoring_times = []
    l8_scoring_times = []
    fusion_scoring_times = []

    for relative_path in query_paths:
        image_path = resolve_image_path(
            relative_path
        )

        image_tensor = preprocess_image(
            image_path
        )

        # Warmup.
        for _ in range(WARMUP_RUNS):
            with torch.no_grad():
                _ = extract_layers(
                    model,
                    image_tensor,
                )

        # Extraction benchmark.
        extraction_run_times = []

        for _ in range(TIMED_RUNS):
            start = time.perf_counter()

            with torch.no_grad():
                outputs = extract_layers(
                    model,
                    image_tensor,
                )

            elapsed = (
                time.perf_counter()
                - start
            ) * 1000

            extraction_run_times.append(
                elapsed
            )

        extraction_times.append(
            median(extraction_run_times)
        )

        l4_features = spatial_features(
            outputs[L4_INDEX]
        )

        l8_features = spatial_features(
            outputs[L8_INDEX]
        )

        # L4 scoring.
        for _ in range(WARMUP_RUNS):
            _ = score_layer(
                l4_features,
                reference_bank["L4"],
            )

        l4_times = []

        for _ in range(TIMED_RUNS):
            start = time.perf_counter()

            _ = score_layer(
                l4_features,
                reference_bank["L4"],
            )

            elapsed = (
                time.perf_counter()
                - start
            ) * 1000

            l4_times.append(
                elapsed
            )

        l4_scoring_times.append(
            median(l4_times)
        )

        # L8 scoring.
        for _ in range(WARMUP_RUNS):
            _ = score_layer(
                l8_features,
                reference_bank["L8"],
            )

        l8_times = []

        for _ in range(TIMED_RUNS):
            start = time.perf_counter()

            _ = score_layer(
                l8_features,
                reference_bank["L8"],
            )

            elapsed = (
                time.perf_counter()
                - start
            ) * 1000

            l8_times.append(
                elapsed
            )

        l8_scoring_times.append(
            median(l8_times)
        )

        # Fusion scoring.
        for _ in range(WARMUP_RUNS):
            _ = score_layer(
                l4_features,
                reference_bank["L4"],
            )
            _ = score_layer(
                l8_features,
                reference_bank["L8"],
            )

        fusion_times = []

        for _ in range(TIMED_RUNS):
            start = time.perf_counter()

            l4_score = score_layer(
                l4_features,
                reference_bank["L4"],
            )

            l8_score = score_layer(
                l8_features,
                reference_bank["L8"],
            )

            _ = (
                0.5 * l4_score
                + 0.5 * l8_score
            )

            elapsed = (
                time.perf_counter()
                - start
            ) * 1000

            fusion_times.append(
                elapsed
            )

        fusion_scoring_times.append(
            median(fusion_times)
        )

        image_times.append(
            median(extraction_run_times)
            + median(fusion_times)
        )

    result = {
        "category": category,
        "reference_images": len(reference_paths),
        "benchmark_query_images": len(query_paths),
        "reference_bank": {
            "L4_shape": list(
                reference_bank["L4"].shape
            ),
            "L8_shape": list(
                reference_bank["L8"].shape
            ),
            "L4_memory_mb": tensor_memory_mb(
                reference_bank["L4"]
            ),
            "L8_memory_mb": tensor_memory_mb(
                reference_bank["L8"]
            ),
            "combined_memory_mb": (
                tensor_memory_mb(
                    reference_bank["L4"]
                )
                + tensor_memory_mb(
                    reference_bank["L8"]
                )
            ),
        },
        "latency_ms_median": {
            "feature_extraction": median(
                extraction_times
            ),
            "L4_scoring": median(
                l4_scoring_times
            ),
            "L8_scoring": median(
                l8_scoring_times
            ),
            "L4_only_total": (
                median(extraction_times)
                + median(l4_scoring_times)
            ),
            "L8_only_total": (
                median(extraction_times)
                + median(l8_scoring_times)
            ),
            "L4_L8_fusion_total": (
                median(extraction_times)
                + median(fusion_scoring_times)
            ),
        },
    }

    print()
    print("Median latency:")
    print(
        f"  Feature extraction: "
        f"{result['latency_ms_median']['feature_extraction']:.2f} ms"
    )
    print(
        f"  L4 scoring: "
        f"{result['latency_ms_median']['L4_scoring']:.2f} ms"
    )
    print(
        f"  L8 scoring: "
        f"{result['latency_ms_median']['L8_scoring']:.2f} ms"
    )
    print(
        f"  L4 only total: "
        f"{result['latency_ms_median']['L4_only_total']:.2f} ms"
    )
    print(
        f"  L8 only total: "
        f"{result['latency_ms_median']['L8_only_total']:.2f} ms"
    )
    print(
        f"  L4+L8 total: "
        f"{result['latency_ms_median']['L4_L8_fusion_total']:.2f} ms"
    )

    return result


def main():
    print("=" * 70)
    print("VISYN — DAY 14 ARCHITECTURE COST BENCHMARK")
    print("=" * 70)

    print(
        "\nCPU device:",
        DEVICE,
    )

    split = load_json(
        NORMAL_SPLIT_PATH
    )

    model = load_model()

    results = {}

    for category in CATEGORIES:
        category_split = split[
            "categories"
        ][category]

        reference_paths = category_split[
            "reference"
        ]

        development_paths = category_split[
            "development"
        ]

        # Use development normals as the fixed
        # benchmark query set.
        results[category] = benchmark_category(
            model,
            category,
            reference_paths,
            development_paths,
        )

    output = {
        "experiment": (
            "Day 14 architecture computational cost"
        ),
        "device": str(DEVICE),
        "model": "MobileNetV3-Small",
        "layers": {
            "L4": {
                "index": L4_INDEX,
                "channels": 40,
                "spatial": "14x14",
            },
            "L8": {
                "index": L8_INDEX,
                "channels": 48,
                "spatial": "14x14",
            },
        },
        "aggregation": "TOP5",
        "reference_bank": "canonical 80% normal reference split",
        "query_set": "canonical normal development split",
        "warmup_runs": WARMUP_RUNS,
        "timed_runs": TIMED_RUNS,
        "categories": results,
    }

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
            output,
            f,
            indent=2,
        )

    print()
    print("=" * 70)
    print("DAY 14 ARCHITECTURE BENCHMARK COMPLETE")
    print("=" * 70)

    print(
        f"Saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()