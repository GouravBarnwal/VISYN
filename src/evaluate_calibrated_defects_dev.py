import json
from pathlib import Path
from collections import defaultdict

CALIBRATION_PATH = Path(
    "artifacts/evaluation/calibration_candidates.json"
)

DEFECT_SCORES_PATH = Path(
    "artifacts/evaluation/mobilenet_l8_dev_scores.json"
)

OUTPUT_PATH = Path(
    "artifacts/evaluation/calibrated_defects_dev.json"
)

CATEGORIES = [
    "bottle",
    "hazelnut",
    "cable",
    "capsule",
    "screw",
    "metal_nut",
]

TARGET_PERCENTILE = 99.0


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_percentile_threshold(category_data, percentile):
    for candidate in category_data["percentile_candidates"]:
        if candidate["percentile"] == percentile:
            return candidate["threshold"]

    raise ValueError(
        f"P{percentile} threshold not found."
    )


def main():
    calibration = load_json(CALIBRATION_PATH)
    defect_scores = load_json(DEFECT_SCORES_PATH)

    calibration_categories = calibration["categories"]

    print("=" * 72)
    print("VISIONFORGE — CALIBRATED DEFECT DEVELOPMENT EVALUATION")
    print("=" * 72)
    print()
    print("Threshold source:")
    print(f"  {CALIBRATION_PATH}")
    print("Defect source:")
    print(f"  {DEFECT_SCORES_PATH}")
    print()
    print("Operating threshold: P99.0")
    print("Score used: TOP5")
    print("IMPORTANT: final defect data is NOT used.")
    print()

    results = {}

    pooled_detected = 0
    pooled_total = 0

    for category in CATEGORIES:

        print("-" * 72)
        print(f"CATEGORY: {category}")
        print()

        threshold = get_percentile_threshold(
            calibration_categories[category],
            TARGET_PERCENTILE
        )

        samples = defect_scores[category]["defect_dev"]

        detected = 0
        missed = 0

        by_type = defaultdict(
            lambda: {
                "total": 0,
                "detected": 0,
            }
        )

        missed_samples = []

        for sample in samples:

            score = sample["top5"]
            defect_type = sample["defect_type"]

            by_type[defect_type]["total"] += 1

            if score >= threshold:

                detected += 1
                by_type[defect_type]["detected"] += 1

            else:

                missed += 1

                missed_samples.append({
                    "path": sample["path"],
                    "defect_type": defect_type,
                    "score": score,
                    "threshold": threshold,
                })

        total = len(samples)

        recall = (
            detected / total
            if total > 0
            else 0.0
        )

        pooled_detected += detected
        pooled_total += total

        print(f"Threshold: {threshold:.6f}")
        print(f"Defect development samples: {total}")
        print(f"Detected: {detected}/{total}")
        print(f"Missed:   {missed}/{total}")
        print(f"Recall:   {recall:.4f}")
        print()

        print("DEFECT TYPE BREAKDOWN")
        print()

        type_results = {}

        for defect_type in sorted(by_type):

            total_type = by_type[defect_type]["total"]
            detected_type = by_type[defect_type]["detected"]

            missed_type = total_type - detected_type

            type_recall = (
                detected_type / total_type
                if total_type > 0
                else 0.0
            )

            type_results[defect_type] = {
                "total": total_type,
                "detected": detected_type,
                "missed": missed_type,
                "recall": type_recall,
            }

            print(
                f"{defect_type:24s} "
                f"{detected_type:3d}/{total_type:<3d} "
                f"recall={type_recall:.4f}"
            )

        print()

        if missed_samples:

            print("MISSED DEFECTS")

            for sample in sorted(
                missed_samples,
                key=lambda x: x["score"]
            ):
                print(
                    f"  {sample['defect_type']:24s} "
                    f"score={sample['score']:.6f} "
                    f"path={sample['path']}"
                )

            print()

        results[category] = {
            "threshold": threshold,
            "total": total,
            "detected": detected,
            "missed": missed,
            "recall": recall,
            "defect_types": type_results,
            "missed_samples": missed_samples,
        }

    pooled_recall = (
        pooled_detected / pooled_total
        if pooled_total > 0
        else 0.0
    )

    macro_recall = (
        sum(
            results[c]["recall"]
            for c in CATEGORIES
        )
        / len(CATEGORIES)
    )

    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print()

    print(
        f"Pooled defect recall: "
        f"{pooled_detected}/{pooled_total} "
        f"= {pooled_recall:.4f}"
    )

    print(
        f"Macro category recall: "
        f"{macro_recall:.4f}"
    )

    print()
    print("CATEGORY RECALL")
    print()

    for category in CATEGORIES:

        result = results[category]

        print(
            f"{category:12s} "
            f"{result['detected']:3d}/"
            f"{result['total']:<3d} "
            f"recall={result['recall']:.4f}"
        )

    output = {
        "threshold_policy": "P99.0",
        "score_aggregation": "TOP5",
        "threshold_source": str(CALIBRATION_PATH),
        "defect_source": str(DEFECT_SCORES_PATH),
        "final_defect_set_used": False,
        "categories": results,
        "pooled": {
            "detected": pooled_detected,
            "total": pooled_total,
            "missed": pooled_total - pooled_detected,
            "recall": pooled_recall,
        },
        "macro_category_recall": macro_recall,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2
        )

    print()
    print("=" * 72)
    print("CALIBRATED DEFECT DEVELOPMENT EVALUATION COMPLETE")
    print("=" * 72)
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()