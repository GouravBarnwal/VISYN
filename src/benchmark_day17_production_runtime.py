from pathlib import Path
import json
import statistics
import time

import psutil

from src.production_config import (
    BULK_BATCH_SIZE,
    CATEGORIES,
    DATA_ROOT,
    SPLITS_PATH,
)

from src.production_inference import (
    ProductionInferenceEngine,
)


OUTPUT_PATH = (
    Path("artifacts/evaluation")
    / "day17_production_runtime.json"
)

WARMUPS = 3
ITERATIONS = 20
SINGLE_IMAGE_COUNT = 5


def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def percentile(values, percentile):
    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (
        (len(ordered) - 1)
        * percentile
        / 100.0
    )

    lower = int(position)
    upper = min(
        lower + 1,
        len(ordered) - 1,
    )

    fraction = position - lower

    return (
        ordered[lower]
        + fraction
        * (
            ordered[upper]
            - ordered[lower]
        )
    )


def measure_single_image(
    engine,
    image_path,
    category,
):
    latencies = []

    for _ in range(WARMUPS):
        engine.inspect(
            image_path,
            category,
        )

    for _ in range(ITERATIONS):
        start = time.perf_counter()

        engine.inspect(
            image_path,
            category,
        )

        end = time.perf_counter()

        latencies.append(
            (end - start)
            * 1000.0
        )

    return latencies


def measure_batch(
    engine,
    image_paths,
    category,
):
    latencies = []

    for _ in range(WARMUPS):
        engine.inspect_batch(
            image_paths,
            category,
        )

    for _ in range(ITERATIONS):
        start = time.perf_counter()

        engine.inspect_batch(
            image_paths,
            category,
        )

        end = time.perf_counter()

        latencies.append(
            (end - start)
            * 1000.0
        )

    return latencies


def summarize(latencies):
    total_ms = sum(latencies)

    return {
        "mean_ms": statistics.mean(
            latencies
        ),
        "median_ms": statistics.median(
            latencies
        ),
        "p50_ms": percentile(
            latencies,
            50.0,
        ),
        "p95_ms": percentile(
            latencies,
            95.0,
        ),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "samples": len(latencies),
        "total_ms": total_ms,
    }


def main():
    print("=" * 72)
    print(
        "VISYN — DAY 17"
    )
    print(
        "PRODUCTION RUNTIME BENCHMARK"
    )
    print("=" * 72)

    split = load_json(
        SPLITS_PATH
    )

    process = psutil.Process()

    print()
    print("Loading production engine...")

    cold_start = time.perf_counter()

    engine = ProductionInferenceEngine(
        device="cpu",
    )

    cold_end = time.perf_counter()

    cold_start_ms = (
        cold_end
        - cold_start
    ) * 1000.0

    memory_after_startup_mb = (
        process.memory_info().rss
        / (
            1024.0
            * 1024.0
        )
    )

    print(
        f"Cold startup: "
        f"{cold_start_ms:.2f} ms"
    )

    print(
        f"RSS after startup: "
        f"{memory_after_startup_mb:.2f} MB"
    )

    results = {
        "device": "cpu",
        "warmups": WARMUPS,
        "iterations": ITERATIONS,
        "bulk_batch_size": BULK_BATCH_SIZE,
        "cold_start_ms": cold_start_ms,
        "rss_after_startup_mb": (
            memory_after_startup_mb
        ),
        "categories": {},
    }

    for category in CATEGORIES:
        print()
        print("=" * 72)
        print(
            f"CATEGORY: {category}"
        )
        print("=" * 72)

        development_paths = split[
            "categories"
        ][category]["development"]

        relative_paths = (
            development_paths[
                :SINGLE_IMAGE_COUNT
            ]
        )

        image_paths = [
            DATA_ROOT / path
            for path in relative_paths
        ]

        # ----------------------------------------------------
        # Single-image runtime
        # ----------------------------------------------------

        print()
        print(
            "Measuring single-image "
            "inference..."
        )

        single_latencies = []

        for image_path in image_paths:
            image_latencies = (
                measure_single_image(
                    engine,
                    image_path,
                    category,
                )
            )

            single_latencies.extend(
                image_latencies
            )

        single_summary = summarize(
            single_latencies
        )

        # ----------------------------------------------------
        # Batch runtime
        # ----------------------------------------------------

        batch_paths = image_paths[
            :BULK_BATCH_SIZE
        ]

        print(
            "Measuring batch inference..."
        )

        batch_latencies = measure_batch(
            engine,
            batch_paths,
            category,
        )

        batch_summary = summarize(
            batch_latencies
        )

        batch_size = len(
            batch_paths
        )

        batch_summary[
            "batch_size"
        ] = batch_size

        batch_summary[
            "images_per_second"
        ] = (
            batch_size
            / (
                batch_summary[
                    "mean_ms"
                ]
                / 1000.0
            )
        )

        batch_summary[
            "ms_per_image_mean"
        ] = (
            batch_summary[
                "mean_ms"
            ]
            / batch_size
        )

        category_result = {
            "single_image": (
                single_summary
            ),
            "batch": batch_summary,
        }

        results[
            "categories"
        ][category] = category_result

        print()
        print(
            "Single image:"
        )
        print(
            f"  p50: "
            f"{single_summary['p50_ms']:.2f} ms"
        )
        print(
            f"  p95: "
            f"{single_summary['p95_ms']:.2f} ms"
        )

        print()
        print(
            f"Batch size: {batch_size}"
        )
        print(
            f"  p50: "
            f"{batch_summary['p50_ms']:.2f} ms"
        )
        print(
            f"  p95: "
            f"{batch_summary['p95_ms']:.2f} ms"
        )
        print(
            f"  Mean/image: "
            f"{batch_summary['ms_per_image_mean']:.2f} ms"
        )
        print(
            f"  Throughput: "
            f"{batch_summary['images_per_second']:.2f} "
            f"images/s"
        )

    # --------------------------------------------------------
    # Final memory measurement
    # --------------------------------------------------------

    memory_after_benchmark_mb = (
        process.memory_info().rss
        / (
            1024.0
            * 1024.0
        )
    )

    results[
        "rss_after_benchmark_mb"
    ] = memory_after_benchmark_mb

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
        "PRODUCTION RUNTIME BENCHMARK COMPLETE"
    )
    print("=" * 72)

    print(
        f"Cold startup: "
        f"{cold_start_ms:.2f} ms"
    )

    print(
        f"RSS after benchmark: "
        f"{memory_after_benchmark_mb:.2f} MB"
    )

    print()
    print(
        f"Saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()