import json
from pathlib import Path

CALIBRATION_PATH = Path(
    "artifacts/evaluation/calibration_candidates.json"
)

DEFECT_SCORES_PATH = Path(
    "artifacts/evaluation/mobilenet_l8_dev_scores.json"
)

OUTPUT_PATH = Path(
    "artifacts/evaluation/calibration_margins.json"
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


def get_threshold(category_data):
    for candidate in category_data["percentile_candidates"]:
        if candidate["percentile"] == TARGET_PERCENTILE:
            return candidate["threshold"]

    raise ValueError(
        f"P{TARGET_PERCENTILE} threshold not found."
    )


def main():
    calibration = load_json(CALIBRATION_PATH)
    defect_scores = load_json(DEFECT_SCORES_PATH)

    results = {}

    print("=" * 72)
    print("VISYN — CALIBRATION MARGIN ANALYSIS")
    print("=" * 72)
    print()
    print("Threshold: P99.0")
    print("Score: TOP5")
    print("Source: defect-development split only")
    print()

    for category in CATEGORIES:

        threshold = get_threshold(
            calibration["categories"][category]
        )

        samples = defect_scores[category]["defect_dev"]

        scored = []

        for sample in samples:
            score = sample["top5"]

            scored.append({
                "path": sample["path"],
                "defect_type": sample["defect_type"],
                "score": score,
                "margin": score - threshold,
            })

        scored.sort(key=lambda x: x["margin"])

        weakest = scored[0]

        margins = [x["margin"] for x in scored]

        print("-" * 72)
        print(f"CATEGORY: {category}")
        print(f"Threshold:              {threshold:.6f}")
        print(f"Weakest defect score:   {weakest['score']:.6f}")
        print(f"Minimum margin:         {weakest['margin']:.6f}")
        print(f"Weakest defect type:    {weakest['defect_type']}")
        print(f"Weakest path:           {weakest['path']}")
        print(
            f"Mean defect margin:     "
            f"{sum(margins) / len(margins):.6f}"
        )

        print()
        print("CLOSEST 5 DEFECTS")

        for item in scored[:5]:
            print(
                f"{item['defect_type']:24s} "
                f"score={item['score']:.6f} "
                f"margin={item['margin']:.6f}"
            )

        results[category] = {
            "threshold": threshold,
            "minimum_margin": weakest["margin"],
            "weakest_defect_score": weakest["score"],
            "weakest_defect_type": weakest["defect_type"],
            "weakest_path": weakest["path"],
            "mean_margin": sum(margins) / len(margins),
            "closest_five": scored[:5],
        }

        print()

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
            {
                "threshold_policy": "P99.0",
                "score_aggregation": "TOP5",
                "source": str(DEFECT_SCORES_PATH),
                "final_defect_set_used": False,
                "categories": results,
            },
            f,
            indent=2
        )

    print("=" * 72)
    print("CALIBRATION MARGIN ANALYSIS COMPLETE")
    print("=" * 72)
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()