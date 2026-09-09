import json
from pathlib import Path

import numpy as np


INPUT_PATH = Path(
    "artifacts/evaluation/day12_reference_bank_robustness.json"
)

OUTPUT_PATH = Path(
    "artifacts/evaluation/day12_reference_bank_analysis.json"
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


def load_results():
    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def analyze_category(category_data):
    runs = category_data["runs"]

    result = {}

    for fraction in FRACTIONS:
        fraction_runs = [
            run
            for run in runs
            if run["fraction"] == fraction
        ]

        result[str(fraction)] = {}

        for method in METHODS:
            values = np.asarray(
                [
                    run["metrics"][method]["AUROC"]
                    for run in fraction_runs
                ],
                dtype=np.float64,
            )

            result[str(fraction)][method] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "min": float(values.min()),
                "max": float(values.max()),
            }

    return result


def main():
    data = load_results()

    analysis = {
        "categories": {},
        "macro": {},
    }

    # --------------------------------------------------------
    # Category analysis
    # --------------------------------------------------------

    for category in CATEGORIES:
        category_result = analyze_category(
            data["categories"][category]
        )

        # Add degradation from full bank.
        full = category_result["1.0"]
        quarter = category_result["0.25"]

        for method in METHODS:
            full_mean = full[method]["mean"]
            quarter_mean = quarter[method]["mean"]

            category_result["degradation_100_to_25"] = (
                category_result.get(
                    "degradation_100_to_25",
                    {}
                )
            )

            category_result[
                "degradation_100_to_25"
            ][method] = {
                "absolute": float(
                    quarter_mean - full_mean
                ),
                "relative_percent": float(
                    (
                        (quarter_mean - full_mean)
                        / full_mean
                    )
                    * 100.0
                ),
            }

        analysis["categories"][category] = (
            category_result
        )

    # --------------------------------------------------------
    # Macro analysis
    # --------------------------------------------------------

    for fraction in FRACTIONS:
        fraction_key = str(fraction)

        analysis["macro"][fraction_key] = {}

        for method in METHODS:
            category_means = []

            category_stds = []

            category_mins = []

            category_maxs = []

            for category in CATEGORIES:
                stats = analysis[
                    "categories"
                ][category][fraction_key][method]

                category_means.append(
                    stats["mean"]
                )

                category_stds.append(
                    stats["std"]
                )

                category_mins.append(
                    stats["min"]
                )

                category_maxs.append(
                    stats["max"]
                )

            analysis["macro"][
                fraction_key
            ][method] = {
                "mean_of_category_means": float(
                    np.mean(category_means)
                ),
                "mean_category_std": float(
                    np.mean(category_stds)
                ),
                "worst_category_mean": float(
                    np.min(category_means)
                ),
                "best_category_mean": float(
                    np.max(category_means)
                ),
            }

    # --------------------------------------------------------
    # Macro degradation
    # --------------------------------------------------------

    analysis["macro_degradation_100_to_25"] = {}

    for method in METHODS:
        full = analysis["macro"]["1.0"][method][
            "mean_of_category_means"
        ]

        quarter = analysis["macro"]["0.25"][method][
            "mean_of_category_means"
        ]

        analysis[
            "macro_degradation_100_to_25"
        ][method] = {
            "absolute": float(
                quarter - full
            ),
            "relative_percent": float(
                (
                    (quarter - full)
                    / full
                )
                * 100.0
            ),
        }

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

    # --------------------------------------------------------
    # Console report
    # --------------------------------------------------------

    print("=" * 78)
    print("DAY 12 — REFERENCE-BANK SENSITIVITY ANALYSIS")
    print("=" * 78)

    print("\nMACRO SUMMARY")
    print("-" * 78)

    for fraction in FRACTIONS:
        key = str(fraction)

        print(
            f"\nReference fraction: {fraction:.0%}"
        )

        for method in METHODS:
            stats = analysis["macro"][key][method]

            print(
                f"  {method:15s} "
                f"mean={stats['mean_of_category_means']:.4f} "
                f"avg_std={stats['mean_category_std']:.4f} "
                f"worst={stats['worst_category_mean']:.4f} "
                f"best={stats['best_category_mean']:.4f}"
            )

    print("\n" + "=" * 78)
    print("CATEGORY-LEVEL FUSION SENSITIVITY")
    print("=" * 78)

    for category in CATEGORIES:
        values = analysis[
            "categories"
        ][category][
            "degradation_100_to_25"
        ][
            "fusion_50_50"
        ]

        print(
            f"{category:12s} "
            f"ΔAUROC={values['absolute']:+.4f} "
            f"({values['relative_percent']:+.2f}%)"
        )

    print("\n" + "=" * 78)
    print("100% → 25% MACRO DEGRADATION")
    print("=" * 78)

    for method in METHODS:
        values = analysis[
            "macro_degradation_100_to_25"
        ][method]

        print(
            f"{method:15s} "
            f"ΔAUROC={values['absolute']:+.4f} "
            f"({values['relative_percent']:+.2f}%)"
        )

    print("\nResults saved to:")
    print(OUTPUT_PATH)

    print("\nDAY 12 EXPERIMENT 2 COMPLETE")


if __name__ == "__main__":
    main()