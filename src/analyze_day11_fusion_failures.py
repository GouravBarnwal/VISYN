import json
from pathlib import Path

import numpy as np


INPUT_PATH = Path(
    "artifacts/evaluation/day11_l4_l8_fusion.json"
)

OUTPUT_PATH = Path(
    "artifacts/evaluation/day11_fusion_failure_analysis.json"
)

CATEGORIES = [
    "bottle",
    "hazelnut",
    "cable",
    "capsule",
    "screw",
    "metal_nut",
]

L4_WEIGHT = 0.5
L8_WEIGHT = 0.5


def main():
    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    results = {}
    all_defect_rows = []

    print()
    print("=" * 72)
    print("VISIONFORGE — DAY 11 FUSION FAILURE ANALYSIS")
    print("=" * 72)
    print()
    print("Comparison: 50/50 L4+L8 fusion vs L8")
    print("Source: canonical development sample-level scores")
    print("Final locked defect set: NOT USED")
    print()

    for category in CATEGORIES:
        sample_scores = data[
            "categories"
        ][category]["sample_scores"]

        normal = sample_scores["normal"]
        defect = sample_scores["defect"]

        # ----------------------------------------------------
        # RECONSTRUCT FUSION SCORES
        # ----------------------------------------------------

        normal_l4 = np.asarray(
            normal["L4_normalized"],
            dtype=np.float64,
        )

        normal_l8 = np.asarray(
            normal["L8_normalized"],
            dtype=np.float64,
        )

        defect_l4 = np.asarray(
            defect["L4_normalized"],
            dtype=np.float64,
        )

        defect_l8 = np.asarray(
            defect["L8_normalized"],
            dtype=np.float64,
        )

        normal_fusion = (
            L4_WEIGHT * normal_l4
            + L8_WEIGHT * normal_l8
        )

        defect_fusion = (
            L4_WEIGHT * defect_l4
            + L8_WEIGHT * defect_l8
        )

        # ----------------------------------------------------
        # SAMPLE-LEVEL DEFECT SHIFTS
        # ----------------------------------------------------

        defect_rows = []

        for index in range(len(defect["paths"])):
            l8 = float(
                defect_l8[index]
            )

            fusion = float(
                defect_fusion[index]
            )

            path_string = defect["paths"][index]

            defect_type = Path(
                path_string
            ).parent.name

            row = {
                "path": path_string,
                "defect_type": defect_type,
                "L8_normalized": l8,
                "fusion_normalized": fusion,
                "fusion_minus_L8": fusion - l8,
            }

            defect_rows.append(row)

            all_defect_rows.append(
                {
                    **row,
                    "category": category,
                }
            )

        # ----------------------------------------------------
        # DEFECT-TYPE SUMMARY
        # ----------------------------------------------------

        defect_type_summary = {}

        defect_types = sorted(
            set(
                row["defect_type"]
                for row in defect_rows
            )
        )

        for defect_type in defect_types:
            rows = [
                row
                for row in defect_rows
                if row["defect_type"] == defect_type
            ]

            shifts = np.asarray(
                [
                    row["fusion_minus_L8"]
                    for row in rows
                ],
                dtype=np.float64,
            )

            defect_type_summary[
                defect_type
            ] = {
                "count": len(rows),
                "mean_score_shift": float(
                    shifts.mean()
                ),
                "median_score_shift": float(
                    np.median(shifts)
                ),
                "min_score_shift": float(
                    shifts.min()
                ),
                "max_score_shift": float(
                    shifts.max()
                ),
                "samples_improved": int(
                    np.sum(shifts > 0)
                ),
                "samples_regressed": int(
                    np.sum(shifts < 0)
                ),
                "samples_unchanged": int(
                    np.sum(shifts == 0)
                ),
            }

        # ----------------------------------------------------
        # LARGEST REGRESSIONS
        # ----------------------------------------------------

        largest_regressions = sorted(
            defect_rows,
            key=lambda row: row["fusion_minus_L8"],
        )[:10]

        # ----------------------------------------------------
        # LARGEST IMPROVEMENTS
        # ----------------------------------------------------

        largest_improvements = sorted(
            defect_rows,
            key=lambda row: row["fusion_minus_L8"],
            reverse=True,
        )[:10]

        # ----------------------------------------------------
        # CATEGORY SUMMARY
        # ----------------------------------------------------

        shifts = np.asarray(
            [
                row["fusion_minus_L8"]
                for row in defect_rows
            ],
            dtype=np.float64,
        )

        results[category] = {
            "normal_count": len(
                normal["paths"]
            ),
            "defect_count": len(
                defect["paths"]
            ),
            "defect_mean_score_shift": float(
                shifts.mean()
            ),
            "defect_median_score_shift": float(
                np.median(shifts)
            ),
            "defect_samples_improved": int(
                np.sum(shifts > 0)
            ),
            "defect_samples_regressed": int(
                np.sum(shifts < 0)
            ),
            "defect_samples_unchanged": int(
                np.sum(shifts == 0)
            ),
            "defect_types": defect_type_summary,
            "largest_regressions": largest_regressions,
            "largest_improvements": largest_improvements,
        }

        # ----------------------------------------------------
        # CONSOLE OUTPUT
        # ----------------------------------------------------

        print("=" * 72)
        print(f"CATEGORY: {category}")
        print("=" * 72)

        print(
            "Defect samples:",
            len(defect_rows),
        )

        print(
            "Improved:",
            int(np.sum(shifts > 0)),
        )

        print(
            "Regressed:",
            int(np.sum(shifts < 0)),
        )

        print(
            "Unchanged:",
            int(np.sum(shifts == 0)),
        )

        print(
            "Mean score shift:",
            f"{shifts.mean():+.4f}",
        )

        print()
        print("Defect-type mean score shifts:")

        for defect_type in defect_types:
            summary = defect_type_summary[
                defect_type
            ]

            print(
                f"  {defect_type:25s} "
                f"{summary['mean_score_shift']:+.4f} "
                f"(+{summary['samples_improved']}/"
                f"-{summary['samples_regressed']})"
            )

        print()
        print("Largest regressions:")

        for row in largest_regressions[:5]:
            print(
                f"  {row['defect_type']:20s} "
                f"{row['fusion_minus_L8']:+.4f} "
                f"{row['path']}"
            )

        print()
        print("Largest improvements:")

        for row in largest_improvements[:5]:
            print(
                f"  {row['defect_type']:20s} "
                f"{row['fusion_minus_L8']:+.4f} "
                f"{row['path']}"
            )

        print()

    # ========================================================
    # GLOBAL SUMMARY
    # ========================================================

    global_shifts = np.asarray(
        [
            row["fusion_minus_L8"]
            for row in all_defect_rows
        ],
        dtype=np.float64,
    )

    results["summary"] = {
        "total_defect_samples": int(
            len(global_shifts)
        ),
        "samples_improved": int(
            np.sum(global_shifts > 0)
        ),
        "samples_regressed": int(
            np.sum(global_shifts < 0)
        ),
        "samples_unchanged": int(
            np.sum(global_shifts == 0)
        ),
        "mean_defect_score_shift": float(
            global_shifts.mean()
        ),
        "median_defect_score_shift": float(
            np.median(global_shifts)
        ),
        "interpretation": (
            "Positive values indicate that the "
            "50/50 fusion strengthens defect "
            "evidence relative to L8. Negative "
            "values indicate that fusion weakens "
            "the L8 defect score."
        ),
    }

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

    print("=" * 72)
    print("GLOBAL SUMMARY")
    print("=" * 72)

    print(
        "Total defect samples:",
        len(global_shifts),
    )

    print(
        "Improved:",
        int(np.sum(global_shifts > 0)),
    )

    print(
        "Regressed:",
        int(np.sum(global_shifts < 0)),
    )

    print(
        "Unchanged:",
        int(np.sum(global_shifts == 0)),
    )

    print(
        "Mean defect score shift:",
        f"{global_shifts.mean():+.4f}",
    )

    print(
        "Median defect score shift:",
        f"{np.median(global_shifts):+.4f}",
    )

    print()
    print("Saved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()