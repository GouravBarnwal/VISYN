import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

# ============================================================
# VISIONFORGE — DAY 15
# PRODUCTION LATENCY BASELINE
# ============================================================

NORMAL_SPLIT_PATH = Path("artifacts/splits/normal_splits.json")

OUTPUT_PATH = Path("artifacts/evaluation/day15_production_baseline.json")

DATA_ROOT = Path("data")

CATEGORIES = [
    "bottle",
    "hazelnut",
    "cable",
    "capsule",
    "screw",
    "metal_nut",
]

IMAGE_SIZE = 224
TOP_K = 5

WARMUP_RUNS = 3
TIMED_RUNS = 20

MEAN = torch.tensor(
    [0.485, 0.456, 0.406],
    dtype=torch.float32,
).view(1, 3, 1, 1)

STD = torch.tensor(
    [0.229, 0.224, 0.225],
    dtype=torch.float32,
).view(1, 3, 1, 1)


# ============================================================
# DATA
# ============================================================


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(path):
    path = Path(path)

    if path.is_absolute():
        return path

    return DATA_ROOT / path


# ============================================================
# MODEL
# ============================================================


def load_model():
    weights = MobileNet_V3_Small_Weights.DEFAULT

    model = mobilenet_v3_small(weights=weights)

    model.eval()

    return model


# ============================================================
# CANONICAL PREPROCESSING
# ============================================================


def preprocess_image(image_path):
    image_path = resolve_path(image_path)

    image = Image.open(image_path).convert("RGB")

    # IMPORTANT:
    # This is the frozen VisionForge protocol.
    # Direct resize — no center crop.
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

    tensor = (tensor - MEAN) / STD

    return tensor


# ============================================================
# FEATURE EXTRACTION
# ============================================================


@torch.no_grad()
def extract_layers(model, image_tensor):

    x = image_tensor

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

    # [1, C, H, W]
    #
    # -> [H*W, C]

    l4 = l4.squeeze(0)
    l4 = l4.permute(1, 2, 0)
    l4 = l4.reshape(
        -1,
        l4.shape[-1],
    )

    l8 = l8.squeeze(0)
    l8 = l8.permute(1, 2, 0)
    l8 = l8.reshape(
        -1,
        l8.shape[-1],
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


# ============================================================
# REFERENCE BANK
# ============================================================


@torch.no_grad()
def build_reference_bank(
    model,
    reference_paths,
):

    l4_parts = []
    l8_parts = []

    for path in reference_paths:

        image = preprocess_image(path)

        l4, l8 = extract_layers(
            model,
            image,
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

    return l4_bank, l8_bank


# ============================================================
# SCORING
# ============================================================


@torch.no_grad()
def score_features(
    query,
    reference_bank,
):

    distances = torch.cdist(
        query,
        reference_bank,
        p=2,
    )

    nearest = distances.min(dim=1).values

    topk = torch.topk(
        nearest,
        k=TOP_K,
        largest=True,
    ).values

    return float(topk.mean().item())


@torch.no_grad()
def score_image(
    model,
    image_tensor,
    l4_bank,
    l8_bank,
):

    l4, l8 = extract_layers(
        model,
        image_tensor,
    )

    l4_score = score_features(
        l4,
        l4_bank,
    )

    l8_score = score_features(
        l8,
        l8_bank,
    )

    fusion_score = 0.5 * l4_score + 0.5 * l8_score

    return {
        "l4": l4_score,
        "l8": l8_score,
        "fusion": fusion_score,
    }


# ============================================================
# TIMING
# ============================================================


def benchmark_category(
    model,
    category,
    reference_paths,
    query_paths,
):

    print("=" * 72)
    print(f"CATEGORY: {category}")
    print("=" * 72)

    print(f"Reference images: " f"{len(reference_paths)}")

    print(f"Query images: " f"{len(query_paths)}")

    print()
    print("Building reference bank...")

    bank_start = time.perf_counter()

    l4_bank, l8_bank = build_reference_bank(
        model,
        reference_paths,
    )

    bank_time = time.perf_counter() - bank_start

    l4_memory_mb = l4_bank.numel() * l4_bank.element_size() / (1024**2)

    l8_memory_mb = l8_bank.numel() * l8_bank.element_size() / (1024**2)

    combined_memory_mb = l4_memory_mb + l8_memory_mb

    print(f"L4 bank: " f"{tuple(l4_bank.shape)} " f"({l4_memory_mb:.2f} MB)")

    print(f"L8 bank: " f"{tuple(l8_bank.shape)} " f"({l8_memory_mb:.2f} MB)")

    print(f"Combined bank memory: " f"{combined_memory_mb:.2f} MB")

    # --------------------------------------------------------
    # Preload query tensors.
    #
    # This prevents disk I/O from contaminating the
    # model/scoring latency measurement.
    # --------------------------------------------------------

    print()
    print("Preprocessing query images...")

    query_tensors = []

    for path in query_paths:
        query_tensors.append(preprocess_image(path))

    # Use at most 10 representative development images
    # to keep the benchmark reasonably fast.
    benchmark_queries = query_tensors[:10]

    if not benchmark_queries:
        raise RuntimeError("No query images available.")

    print(f"Benchmark query count: " f"{len(benchmark_queries)}")

    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    print()
    print("Warmup...")

    for image_tensor in benchmark_queries:

        for _ in range(WARMUP_RUNS):
            score_image(
                model,
                image_tensor,
                l4_bank,
                l8_bank,
            )

    # --------------------------------------------------------
    # Timing
    # --------------------------------------------------------

    extraction_times = []
    l4_times = []
    l8_times = []
    total_times = []

    print("Timed benchmark...")

    for image_tensor in benchmark_queries:

        # Feature extraction
        start = time.perf_counter()

        l4, l8 = extract_layers(
            model,
            image_tensor,
        )

        extraction_time = time.perf_counter() - start

        # L4
        start = time.perf_counter()

        l4_score = score_features(
            l4,
            l4_bank,
        )

        l4_time = time.perf_counter() - start

        # L8
        start = time.perf_counter()

        l8_score = score_features(
            l8,
            l8_bank,
        )

        l8_time = time.perf_counter() - start

        # Full production path
        start = time.perf_counter()

        fusion_score = 0.5 * l4_score + 0.5 * l8_score

        total_time = time.perf_counter() - start

        extraction_times.append(extraction_time * 1000)

        l4_times.append(l4_time * 1000)

        l8_times.append(l8_time * 1000)

        # NOTE:
        # This currently measures fusion arithmetic only,
        # while extraction + scoring are measured separately.
        #
        # We therefore also record a true end-to-end timing
        # below.

    # --------------------------------------------------------
    # True end-to-end timing
    # --------------------------------------------------------

    end_to_end_times = []

    for image_tensor in benchmark_queries:

        start = time.perf_counter()

        score_image(
            model,
            image_tensor,
            l4_bank,
            l8_bank,
        )

        elapsed = time.perf_counter() - start

        end_to_end_times.append(elapsed * 1000)

    def stats(values):

        values = np.asarray(
            values,
            dtype=np.float64,
        )

        return {
            "mean_ms": float(values.mean()),
            "median_ms": float(np.median(values)),
            "p95_ms": float(np.percentile(values, 95)),
            "min_ms": float(values.min()),
            "max_ms": float(values.max()),
        }

    result = {
        "category": category,
        "reference_count": len(reference_paths),
        "query_count": len(benchmark_queries),
        "reference_bank": {
            "l4_shape": list(l4_bank.shape),
            "l8_shape": list(l8_bank.shape),
            "l4_memory_mb": float(l4_memory_mb),
            "l8_memory_mb": float(l8_memory_mb),
            "combined_memory_mb": float(combined_memory_mb),
        },
        "timing_ms": {
            "feature_extraction": stats(extraction_times),
            "l4_scoring": stats(l4_times),
            "l8_scoring": stats(l8_times),
            "end_to_end": stats(end_to_end_times),
        },
    }

    print()
    print(
        "Feature extraction: "
        f"{result['timing_ms']['feature_extraction']['mean_ms']:.2f} ms"
    )

    print("L4 scoring: " f"{result['timing_ms']['l4_scoring']['mean_ms']:.2f} ms")

    print("L8 scoring: " f"{result['timing_ms']['l8_scoring']['mean_ms']:.2f} ms")

    print("End-to-end: " f"{result['timing_ms']['end_to_end']['mean_ms']:.2f} ms")

    print("End-to-end P95: " f"{result['timing_ms']['end_to_end']['p95_ms']:.2f} ms")

    return result


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 72)
    print("VISIONFORGE — DAY 15 " "PRODUCTION LATENCY BASELINE")
    print("=" * 72)

    print()
    print("Frozen architecture:")
    print("  MobileNetV3-Small")
    print("  L4 + L8")
    print("  Raw 50/50 fusion")
    print("  Euclidean distance")
    print("  TOP5 aggregation")
    print("  Direct 224x224 preprocessing")
    print()

    split = load_json(NORMAL_SPLIT_PATH)

    print("Loading MobileNetV3-Small...")

    model_start = time.perf_counter()

    model = load_model()

    model_load_time = (time.perf_counter() - model_start) * 1000

    print(f"Model load time: " f"{model_load_time:.2f} ms")

    results = {
        "experiment": ("VisionForge Day 15 " "production latency baseline"),
        "device": "CPU",
        "model": "MobileNetV3-Small",
        "layers": ["L4", "L8"],
        "fusion": "raw_50_50",
        "distance": "Euclidean",
        "aggregation": "TOP5",
        "preprocessing": {
            "resize": "direct_224x224",
            "scaling": "/255",
            "normalization": "ImageNet mean/std",
            "center_crop": False,
        },
        "warmup_runs": WARMUP_RUNS,
        "timed_runs": TIMED_RUNS,
        "model_load_time_ms": float(model_load_time),
        "categories": {},
    }

    for category in CATEGORIES:

        reference_paths = split["categories"][category]["reference"]

        query_paths = split["categories"][category]["development"]

        result = benchmark_category(
            model,
            category,
            reference_paths,
            query_paths,
        )

        results["categories"][category] = result

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

    print()
    print("=" * 72)
    print("DAY 15 BASELINE COMPLETE")
    print("=" * 72)
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
