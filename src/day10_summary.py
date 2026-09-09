from pathlib import Path
import json


CATEGORY_RESULTS = Path(
    "artifacts/evaluation/day10_category_metrics.json"
)

DEFECT_TYPE_RESULTS = Path(
    "artifacts/evaluation/day10_defect_type_metrics.json"
)

OUTPUT_PATH = Path(
    "artifacts/evaluation/day10_summary.json"
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    category_results = load_json(CATEGORY_RESULTS)
    defect_results = load_json(DEFECT_TYPE_RESULTS)

    weakest_auroc = defect_results[
        "weakest_by_auroc"
    ]

    weakest_detection = defect_results[
        "weakest_by_detection"
    ]

    summary = {
        "day": 10,
        "model": "MobileNetV3-Small-L8",
        "distance_metric": "euclidean_cdist_p2",
        "aggregation": "TOP5",
        "evaluation_split": "canonical_development",
        "final_set_used": False,

        "category_metrics": {
            category: {
                "auroc": metrics["auroc"],
                "average_precision": metrics[
                    "average_precision"
                ],
            }
            for category, metrics
            in category_results["categories"].items()
        },

        "macro_category_metrics": (
            category_results["macro"]
        ),

        "defect_type_metrics": (
            defect_results["macro_defect_type"]
        ),

        "weakest_defect_types_by_auroc": (
            weakest_auroc[:10]
        ),

        "weakest_defect_types_by_detection": (
            weakest_detection[:10]
        ),

        "failure_analysis": {
            "primary_failure_mode": (
                "Subtle localized defects show "
                "reduced separability in the "
                "MobileNet L8 representation."
            ),
            "strong_categories": [
                "bottle",
                "hazelnut",
                "cable",
                "metal_nut",
            ],
            "weaker_categories": [
                "capsule",
                "screw",
            ],
            "key_screw_failures": [
                "scratch_head",
                "thread_side",
                "manipulated_front",
                "thread_top",
            ],
            "key_capsule_failures": [
                "faulty_imprint",
                "poke",
                "crack",
                "scratch",
            ],
            "key_cable_failure": (
                "missing_wire"
            ),
            "threshold_change_recommended": False,
            "representation_investigation_recommended": True,
        },

        "artifacts": {
            "category_metrics": str(
                CATEGORY_RESULTS
            ),
            "defect_type_metrics": str(
                DEFECT_TYPE_RESULTS
            ),
        },
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
            summary,
            f,
            indent=2,
        )

    print("=" * 72)
    print("VISYN — DAY 10 SUMMARY")
    print("=" * 72)
    print()
    print("Category macro AUROC:")
    print(
        f"  {category_results['macro']['auroc']:.4f}"
    )
    print()
    print("Category macro AP:")
    print(
        f"  {category_results['macro']['average_precision']:.4f}"
    )
    print()
    print("Defect-type macro AUROC:")
    print(
        f"  {defect_results['macro_defect_type']['auroc']:.4f}"
    )
    print()
    print("Primary failure mode:")
    print(
        "  Subtle localized defects "
        "in the L8 representation"
    )
    print()
    print(
        "Threshold change recommended: NO"
    )
    print(
        "Representation investigation: YES"
    )
    print()
    print(f"Saved to: {OUTPUT_PATH}")
    print("=" * 72)


if __name__ == "__main__":
    main()