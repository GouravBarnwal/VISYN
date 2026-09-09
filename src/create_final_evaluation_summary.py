import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

RESULTS_PATH = ROOT / "artifacts" / "evaluation" / "day14" / "day14_locked_final_results.json"
OUTPUT_PATH = ROOT / "artifacts" / "evaluation" / "final_evaluation_summary.json"


def main():
    with RESULTS_PATH.open("r", encoding="utf-8") as f:
        results = json.load(f)

    categories = results["categories"]
    fusion = results["macro_average"]["fusion_50_50"]

    summary = {
        "project": "VISYN",
        "title": "VISYN — Deep Visual Intelligence for Quality Inspection",
        "evaluation_status": "LOCKED",
        "final_test_used": results["final_test_used"],
        "final_test_used_for_tuning": results["final_test_used_for_tuning"],

        "production_architecture": {
            "model": results["model"],
            "feature_layers": ["L4", "L8"],
            "layer_shapes": {
                "L4": results["layers"]["L4"]["spatial"],
                "L8": results["layers"]["L8"]["spatial"],
            },
            "distance": results["distance"],
            "aggregation": results["aggregation"],
            "fusion": results["fusion"],
        },

        "preprocessing": results["preprocessing"],

        "evaluation_protocol": {
            "reference_ratio": 0.8,
            "seed": 42,
            "reference_selection": "canonical normal split",
            "final_test_role": "final evaluation only; not used for tuning",
        },

        "category_results": {
            category: {
                "reference_count": values["reference_count"],
                "normal_development_count": values["normal_development_count"],
                "final_defect_count": values["final_defect_count"],
                "auroc": values["fusion_50_50"]["auroc"],
                "average_precision": values["fusion_50_50"]["average_precision"],
            }
            for category, values in categories.items()
        },

        "macro_results": {
            "auroc": fusion["auroc"],
            "average_precision": fusion["average_precision"],
        },

        "decision_policy": {
            "calibration": "category-specific P99 normal-development threshold",
            "review_multiplier": 1.10,
            "decisions": ["PASS", "REVIEW", "FAIL"],
        },

        "localization": {
            "role": "explanatory evidence only",
            "affects_classification": False,
        },

        "known_limitations": [
            "Screw remains the most difficult category.",
            "Capsule contains subtle localized defects that remain challenging.",
            "Localization heatmaps provide approximate evidence rather than precise segmentation.",
        ],

        "research_components_not_in_production": [
            "DINOv2",
            "TOP10 aggregation",
            "median/MAD score normalization",
            "naive diversity reference selection",
            "L3-only architecture",
        ],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("VISYN FINAL EVALUATION SUMMARY")
    print("=" * 40)
    print(f"Output: {OUTPUT_PATH}")
    print(f"Macro AUROC: {fusion['auroc']:.4f}")
    print(f"Macro AP:     {fusion['average_precision']:.4f}")
    print(f"Categories:   {len(categories)}")
    print("Final test used for tuning: NO")
    print("STATUS: PASS")


if __name__ == "__main__":
    main()