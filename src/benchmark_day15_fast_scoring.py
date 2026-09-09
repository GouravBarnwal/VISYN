import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


# ============================================================
# VISYN — DAY 15
# FAST SCORING EQUIVALENCE BENCHMARK
# ============================================================

NORMAL_SPLIT_PATH = Path(
    "artifacts/splits/normal_splits.json"
)

OUTPUT_PATH = Path(
    "artifacts/evaluation/day15_fast_scoring.json"
)

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

    model = mobilenet_v3_small(
        weights=weights
    )

    model.eval()

    return model


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_image(image_path):
    image_path = resolve_path(image_path)

    image = Image.open(
        image_path
    ).convert("RGB")

    image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR,
    )

    array = np.asarray(
        image,
        dtype=np.float32,
    ) / 255.0

    tensor = torch.from_numpy(
        array
    )

    tensor = tensor.permute(
        2, 0, 1
    )

    tensor = tensor.unsqueeze(0)

    tensor = (
        tensor - MEAN
    ) / STD

    return tensor


# ============================================================
# FEATURE EXTRACTION
# ============================================================

@torch.no_grad()
def extract_layers(model, image_tensor):

    x = image_tensor

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
            "Failed to capture L4/L8 features."
        )

    l4 = l4.squeeze(0)
    l4 = l4.permute(
        1, 2, 0
    )
    l4 = l4.reshape(
        -1,
        l4.shape[-1],
    )

    l8 = l8.squeeze(0)
    l8 = l8.permute(
        1, 2, 0
    )
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

        image = preprocess_image(
            path
        )

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
# ORIGINAL SCORING
# ============================================================

@torch.no_grad()
def score_cdist(
    query,
    reference_bank,
):

    distances = torch.cdist(
        query,
        reference_bank,
        p=2,
    )

    nearest = distances.min(
        dim=1
    ).values

    topk = torch.topk(
        nearest,
        k=TOP_K,
        largest=True,
    ).values

    return topk.mean()


# ============================================================
# OPTIMIZED SCORING
# ============================================================

@torch.no_grad()
def score_matmul(
    query,
    reference_bank,
):

    # Query and reference features are already
    # L2-normalized.
    #
    # Therefore:
    #
    # ||q-r||^2 = 2 - 2(q.r)
    #
    # Minimizing Euclidean distance is equivalent
    # to maximizing dot-product similarity.

    similarity = (
        query @ reference_bank.T
    )

    nearest_similarity = similarity.max(
        dim=1
    ).values

    # Convert similarity back to Euclidean distance.
    #
    # Clamp prevents tiny floating-point errors
    # from producing negative values inside sqrt.

    squared_distance = (
        2.0
        - 2.0 * nearest_similarity
    )

    squared_distance = torch.clamp(
        squared_distance,
        min=0.0,
    )

    nearest_distance = torch.sqrt(
        squared_distance
    )

    topk = torch.topk(
        nearest_distance,
        k=TOP_K,
        largest=True,
    ).values

    return topk.mean()


# ============================================================
# COMPARISON
# ============================================================

def compare_scores(
    query,
    reference_bank,
):

    original = score_cdist(
        query,
        reference_bank,
    )

    optimized = score_matmul(
        query,
        reference_bank,
    )

    difference = (
        optimized
        - original
    )

    absolute_difference = abs(
        difference.item()
    )

    return {
        "cdist_score": float(
            original.item()
        ),
        "matmul_score": float(
            optimized.item()
        ),
        "absolute_difference": float(
            absolute_difference
        ),
    }


# ============================================================
# TIMING
# ============================================================

def benchmark_function(
    function,
    query,
    reference_bank,
):

    for _ in range(
        WARMUP_RUNS
    ):
        function(
            query,
            reference_bank,
        )

    times = []

    for _ in range(
        TIMED_RUNS
    ):
        start = time.perf_counter()

        function(
            query,
            reference_bank,
        )

        elapsed = (
            time.perf_counter()
            - start
        ) * 1000

        times.append(
            elapsed
        )

    values = np.asarray(
        times,
        dtype=np.float64,
    )

    return {
        "mean_ms": float(
            values.mean()
        ),
        "median_ms": float(
            np.median(values)
        ),
        "p95_ms": float(
            np.percentile(
                values,
                95,
            )
        ),
        "min_ms": float(
            values.min()
        ),
        "max_ms": float(
            values.max()
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print(
        "VISYN — DAY 15"
    )
    print(
        "FAST SCORING EQUIVALENCE BENCHMARK"
    )
    print("=" * 72)

    print()
    print("Comparing:")
    print("  Original: torch.cdist")
    print("  Optimized: normalized feature matmul")
    print()
    print("Both use:")
    print("  Euclidean-equivalent scoring")
    print("  TOP5 aggregation")
    print("  L2-normalized patch features")
    print()

    split = load_json(
        NORMAL_SPLIT_PATH
    )

    print("Loading model...")

    model = load_model()

    results = {
        "experiment": (
            "Day 15 fast scoring "
            "equivalence benchmark"
        ),
        "device": "CPU",
        "top_k": TOP_K,
        "original_method": (
            "torch.cdist"
        ),
        "optimized_method": (
            "normalized_feature_matmul"
        ),
        "mathematical_equivalence": (
            "Euclidean distance ranking "
            "via normalized dot product"
        ),
        "categories": {},
    }

    global_max_difference = 0.0

    for category in CATEGORIES:

        print()
        print("=" * 72)
        print(
            f"CATEGORY: {category}"
        )
        print("=" * 72)

        reference_paths = (
            split["categories"]
            [category]["reference"]
        )

        query_paths = (
            split["categories"]
            [category]["development"]
        )

        print(
            f"Reference images: "
            f"{len(reference_paths)}"
        )

        print(
            f"Development images: "
            f"{len(query_paths)}"
        )

        print()
        print(
            "Building reference bank..."
        )

        l4_bank, l8_bank = (
            build_reference_bank(
                model,
                reference_paths,
            )
        )

        # Use a few representative query images
        benchmark_paths = query_paths[:10]

        print(
            f"Benchmark images: "
            f"{len(benchmark_paths)}"
        )

        category_results = {
            "reference_count": len(
                reference_paths
            ),
            "query_count": len(
                benchmark_paths
            ),
            "layers": {},
        }

        for layer_name, bank in [
            ("L4", l4_bank),
            ("L8", l8_bank),
        ]:

            print()
            print(
                f"--- {layer_name} ---"
            )

            layer_differences = []

            for path in benchmark_paths:

                image = preprocess_image(
                    path
                )

                l4, l8 = extract_layers(
                    model,
                    image,
                )

                if layer_name == "L4":
                    query = l4
                else:
                    query = l8

                comparison = compare_scores(
                    query,
                    bank,
                )

                layer_differences.append(
                    comparison[
                        "absolute_difference"
                    ]
                )

            max_difference = max(
                layer_differences
            )

            mean_difference = float(
                np.mean(
                    layer_differences
                )
            )

            global_max_difference = max(
                global_max_difference,
                max_difference,
            )

            print(
                f"Max score difference: "
                f"{max_difference:.10f}"
            )

            print(
                f"Mean score difference: "
                f"{mean_difference:.10f}"
            )

            print(
                "Benchmarking cdist..."
            )

            cdist_times = []

            for path in benchmark_paths:

                image = preprocess_image(
                    path
                )

                l4, l8 = extract_layers(
                    model,
                    image,
                )

                query = (
                    l4
                    if layer_name == "L4"
                    else l8
                )

                timing = benchmark_function(
                    score_cdist,
                    query,
                    bank,
                )

                cdist_times.append(
                    timing
                )

            print(
                "Benchmarking matmul..."
            )

            matmul_times = []

            for path in benchmark_paths:

                image = preprocess_image(
                    path
                )

                l4, l8 = extract_layers(
                    model,
                    image,
                )

                query = (
                    l4
                    if layer_name == "L4"
                    else l8
                )

                timing = benchmark_function(
                    score_matmul,
                    query,
                    bank,
                )

                matmul_times.append(
                    timing
                )

            cdist_mean = float(
                np.mean(
                    [
                        x["mean_ms"]
                        for x in cdist_times
                    ]
                )
            )

            matmul_mean = float(
                np.mean(
                    [
                        x["mean_ms"]
                        for x in matmul_times
                    ]
                )
            )

            speedup = (
                cdist_mean
                / matmul_mean
            )

            improvement = (
                1.0
                - (
                    matmul_mean
                    / cdist_mean
                )
            ) * 100.0

            print(
                f"cdist mean: "
                f"{cdist_mean:.2f} ms"
            )

            print(
                f"matmul mean: "
                f"{matmul_mean:.2f} ms"
            )

            print(
                f"Speedup: "
                f"{speedup:.2f}x"
            )

            print(
                f"Latency reduction: "
                f"{improvement:.2f}%"
            )

            category_results[
                "layers"
            ][layer_name] = {
                "max_score_difference":
                    float(
                        max_difference
                    ),
                "mean_score_difference":
                    float(
                        mean_difference
                    ),
                "cdist_mean_ms":
                    cdist_mean,
                "matmul_mean_ms":
                    matmul_mean,
                "speedup":
                    float(speedup),
                "latency_reduction_percent":
                    float(improvement),
            }

        results[
            "categories"
        ][category] = category_results

    results[
        "global_max_score_difference"
    ] = float(
        global_max_difference
    )

    results[
        "equivalence_pass"
    ] = bool(
        global_max_difference < 1e-5
    )

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
    print(
        "FINAL EQUIVALENCE CHECK"
    )
    print("=" * 72)

    print(
        f"Global maximum score difference: "
        f"{global_max_difference:.10f}"
    )

    if results["equivalence_pass"]:
        print(
            "Numerical equivalence: PASS"
        )
    else:
        print(
            "Numerical equivalence: FAIL"
        )

    print()
    print(
        f"Saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()