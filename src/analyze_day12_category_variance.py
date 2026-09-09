import json
from pathlib import Path

import numpy as np


INPUT_PATH = Path(
    "artifacts/evaluation/day12_reference_bank_robustness.json"
)

OUTPUT_PATH = Path(
    "artifacts/evaluation/day12_category_variance.json"
)

CATEGORIES = [
    "bottle",
    "hazelnut",
    "cable",
    "capsule",
    "screw",
    "metal_nut",
]

FRACTIONS = [
    1.0,
    0.75,
    0.50,
    0.25,
]

METHODS = [
    "L4",
    "L8",
    "fusion_50_50",
]


def main():
    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    analysis = {
        "categories": {},
    }

    print("=" * 78)
    print("DAY 12 — CATEGORY REFERENCE-BANK VARIANCE")
    print("=" * 78)

    for category in CATEGORIES:
        runs = data["categories"][category]["runs"]

        analysis["categories"][category] = {}

        print(f"\n{'=' * 78}")
        print(f"CATEGORY: {category}")
        print(f"{'=' * 78}")

        for fraction in FRACTIONS:
            fraction_runs = [
                run
                for run in runs
                if run["fraction"] == fraction
            ]

            analysis["categories"][category][
                str(fraction)
            ] = {}

            print(
                f"\nReference fraction: {fraction:.0%}"
            )

            for method in METHODS:
                values = np.asarray(
                    [
                        run["metrics"][method]["AUROC"]
                        for run in fraction_runs
                    ],
                    dtype=np.float64,
                )

                stats = {
                    "values": values.tolist(),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=0)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "range": float(
                        values.max() - values.min()
                    ),
                }

                analysis["categories"][category][
                    str(fraction)
                ][method] = stats

                print(
                    f"  {method:15s} "
                    f"mean={stats['mean']:.4f} "
                    f"std={stats['std']:.4f} "
                    f"range={stats['range']:.4f} "
                    f"min={stats['min']:.4f} "
                    f"max={stats['max']:.4f}"
                )

    # --------------------------------------------------------
    # 100% -> 25% degradation
    # --------------------------------------------------------

    print("\n" + "=" * 78)
    print("100% → 25% DEGRADATION BY CATEGORY")
    print("=" * 78)

    analysis["degradation"] = {}

    for category in CATEGORIES:
        analysis["degradation"][category] = {}

        print(f"\n{category}")

        for method in METHODS:
            full = analysis["categories"][
                category
            ]["1.0"][method]["mean"]

            quarter = analysis["categories"][
                category
            ]["0.25"][method]["mean"]

            delta = quarter - full

            analysis["degradation"][category][
                method
            ] = float(delta)

            print(
                f"  {method:15s} "
                f"ΔAUROC={delta:+.4f}"
            )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

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
            analysis,
            f,
            indent=2,
        )

    print("\nResults saved to:")
    print(OUTPUT_PATH)

    print("\nDAY 12 EXPERIMENT 3 COMPLETE")


if __name__ == "__main__":
    main()