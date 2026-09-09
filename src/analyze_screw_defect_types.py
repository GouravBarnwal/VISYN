import json
from pathlib import Path

from src.production_inference import ProductionInferenceEngine


CATEGORY = "screw"

DEFECT_TYPES = [
    "manipulated_front",
    "scratch_head",
    "scratch_neck",
    "thread_side",
    "thread_top",
]

IMAGES_PER_TYPE = 10

OUTPUT_PATH = Path(
    "artifacts/evaluation/day19_screw_defect_types.json"
)


def main():
    engine = ProductionInferenceEngine()

    threshold = engine.thresholds[CATEGORY]

    print("=" * 72)
    print("VISYN — SCREW DEFECT-TYPE ANALYSIS")
    print("=" * 72)
    print(f"Threshold: {threshold:.6f}")
    print()

    results = {}

    for defect_type in DEFECT_TYPES:
        directory = (
            Path("data")
            / CATEGORY
            / "test"
            / defect_type
        )

        images = sorted(directory.glob("*.png"))[
            :IMAGES_PER_TYPE
        ]

        if not images:
            raise FileNotFoundError(
                f"No images found for {defect_type}: "
                f"{directory}"
            )

        decisions = []
        scores = []

        for image_path in images:
            result = engine.inspect(
                category=CATEGORY,
                image_path=image_path,
            )

            decisions.append(result["decision"])
            scores.append(result["anomaly_score"])

        fail_count = decisions.count("FAIL")
        review_count = decisions.count("REVIEW")
        pass_count = decisions.count("PASS")

        detected_count = (
            fail_count + review_count
        )

        detection_rate = (
            detected_count / len(images)
        )

        fail_rate = (
            fail_count / len(images)
        )

        results[defect_type] = {
            "images": len(images),
            "pass": pass_count,
            "review": review_count,
            "fail": fail_count,
            "detected": detected_count,
            "detection_rate": detection_rate,
            "fail_rate": fail_rate,
            "min_score": min(scores),
            "max_score": max(scores),
            "mean_score": (
                sum(scores) / len(scores)
            ),
        }

        print(f"{defect_type}")
        print(
            f"  PASS:     {pass_count}/{len(images)}"
        )
        print(
            f"  REVIEW:   {review_count}/{len(images)}"
        )
        print(
            f"  FAIL:     {fail_count}/{len(images)}"
        )
        print(
            f"  DETECTED: {detected_count}/{len(images)} "
            f"({detection_rate * 100:.1f}%)"
        )
        print(
            f"  SCORE:    {min(scores):.4f} - "
            f"{max(scores):.4f}"
        )
        print()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "category": CATEGORY,
        "threshold": threshold,
        "images_per_type": IMAGES_PER_TYPE,
        "defect_types": results,
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
        )

    print("=" * 72)
    print(
        f"Saved to: {OUTPUT_PATH}"
    )
    print(
        "SCREW DEFECT-TYPE ANALYSIS: PASS"
    )


if __name__ == "__main__":
    main()