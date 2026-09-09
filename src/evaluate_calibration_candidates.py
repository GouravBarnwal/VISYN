from pathlib import Path
import json

import numpy as np


# ============================================================
# CONFIG
# ============================================================

SCORE_PATH = Path(
    "artifacts/evaluation/mobilenet_l8_normal_dev_scores.json"
)

OUTPUT_PATH = Path(
    "artifacts/evaluation/calibration_candidates.json"
)

CATEGORIES = [
    "bottle",
    "hazelnut",
    "cable",
    "capsule",
    "screw",
    "metal_nut",
]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print(
        "VISIONFORGE — CALIBRATION CANDIDATE ANALYSIS"
    )
    print("=" * 72)

    with open(
        SCORE_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    percentiles = [
        95.0,
        97.5,
        99.0,
        99.5,
    ]

    output = {
        "source": str(SCORE_PATH),
        "categories": {},
    }

    for category in CATEGORIES:

        print()
        print("-" * 72)
        print(
            f"CATEGORY: {category}"
        )
        print("-" * 72)

        category_data = (
            data[
                "categories"
            ][category]
        )

        scores = np.asarray(
            category_data[
                "scores"
            ],
            dtype=np.float64,
        )

        print(
            f"Normal development samples: "
            f"{len(scores)}"
        )

        results = []

        for p in percentiles:

            threshold = float(
                np.percentile(
                    scores,
                    p,
                )
            )

            flagged = (
                scores >= threshold
            )

            flagged_count = int(
                flagged.sum()
            )

            false_positive_rate = float(
                flagged.mean()
            )

            results.append({
                "percentile": p,
                "threshold": threshold,
                "flagged_count": flagged_count,
                "false_positive_rate":
                    false_positive_rate,
            })

            print(
                f"P{p:<4} "
                f"threshold="
                f"{threshold:.6f}  "
                f"flagged="
                f"{flagged_count:2d}/"
                f"{len(scores):2d}  "
                f"rate="
                f"{false_positive_rate:.4f}"
            )

        # ----------------------------------------------------
        # Score spread
        # ----------------------------------------------------

        mean = float(
            np.mean(scores)
        )

        std = float(
            np.std(scores)
        )

        # Standardized candidate boundaries
        # are useful as a secondary diagnostic.
        sigma_candidates = {}

        for multiplier in [
            2.0,
            2.5,
            3.0,
        ]:

            threshold = (
                mean
                +
                multiplier * std
            )

            flagged = (
                scores >= threshold
            )

            sigma_candidates[
                f"mean_plus_{multiplier}std"
            ] = {
                "threshold":
                    float(threshold),
                "flagged_count":
                    int(flagged.sum()),
                "false_positive_rate":
                    float(flagged.mean()),
            }

        output[
            "categories"
        ][category] = {
            "count": int(
                len(scores)
            ),
            "mean": mean,
            "std": std,
            "percentile_candidates":
                results,
            "sigma_candidates":
                sigma_candidates,
        }

    # ========================================================
    # MACRO SUMMARY
    # ========================================================

    print()
    print("=" * 72)
    print(
        "MACRO FALSE-POSITIVE SUMMARY"
    )
    print("=" * 72)

    for p in percentiles:

        rates = []

        flagged_total = 0
        total = 0

        for category in CATEGORIES:

            category_results = (
                output[
                    "categories"
                ][category][
                    "percentile_candidates"
                ]
            )

            for result in category_results:

                if (
                    result["percentile"]
                    == p
                ):

                    rates.append(
                        result[
                            "false_positive_rate"
                        ]
                    )

                    flagged_total += (
                        result[
                            "flagged_count"
                        ]
                    )

                    total += (
                        output[
                            "categories"
                        ][category]["count"]
                    )

                    break

        print(
            f"P{p:<4} "
            f"macro_category_rate="
            f"{np.mean(rates):.4f}  "
            f"pooled_rate="
            f"{flagged_total / total:.4f}  "
            f"flagged="
            f"{flagged_total}/"
            f"{total}"
        )

    # ========================================================
    # SAVE
    # ========================================================

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
    print("=" * 72)
    print(
        "CALIBRATION CANDIDATE ANALYSIS COMPLETE"
    )
    print(
        f"Saved to: {OUTPUT_PATH}"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
    