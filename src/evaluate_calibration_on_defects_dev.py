from pathlib import Path
import json
import numpy as np


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

CALIBRATION_PATH = Path(
    "artifacts/evaluation/calibration_candidates.json"
)

DEFECT_SCORES_PATH = Path(
    "artifacts/evaluation/mobilenet_l8_dev_scores.json"
)

OUTPUT_DIR = Path(
    "artifacts/evaluation/calibration"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "p99_defect_dev_evaluation.json"


# ------------------------------------------------------------
# Load artifacts
# ------------------------------------------------------------

with open(CALIBRATION_PATH, "r") as f:
    calibration = json.load(f)

with open(DEFECT_SCORES_PATH, "r") as f:
    defect_scores = json.load(f)

print("Loaded calibration candidates.")
print("Loaded defect-development scores.")


# ------------------------------------------------------------
# Extract P99 thresholds
# ------------------------------------------------------------

thresholds = {}

for category, data in calibration["categories"].items():

    p99_entry = None

    for candidate in data["percentile_candidates"]:
        if candidate["percentile"] == 99.0:
            p99_entry = candidate
            break

    if p99_entry is None:
        raise RuntimeError(
            f"P99 threshold not found for category: {category}"
        )

    thresholds[category] = p99_entry["threshold"]


print("\nFrozen P99 thresholds:")
for category, threshold in thresholds.items():
    print(f"{category:12s} {threshold:.6f}")


# ------------------------------------------------------------
# Helper: extract TOP5 scores
# ------------------------------------------------------------

def get_top5_scores(category_data):

    # Expected structure:
    #
    # {
    #   "max": [...],
    #   "top5": [...],
    #   "top10": [...]
    # }
    #
    # But support common capitalization variants.

    for key in ["top5", "TOP5", "top_5", "TOP_5"]:

        if key in category_data:
            return np.asarray(
                category_data[key],
                dtype=float
            )

    raise RuntimeError(
        "TOP5 scores not found. "
        f"Available keys: {list(category_data.keys())}"
    )


# ------------------------------------------------------------
# Evaluate defect-development samples
# ------------------------------------------------------------

results = {}

all_detected = 0
all_defects = 0

category_recalls = []


for category, threshold in thresholds.items():

    if category not in defect_scores:
        raise RuntimeError(
            f"No defect-development scores found for {category}"
        )

    category_data = defect_scores[category]

    scores = get_top5_scores(category_data)

    detected = scores >= threshold

    tp = int(np.sum(detected))
    fn = int(np.sum(~detected))

    total = len(scores)

    recall = tp / total if total > 0 else 0.0
    miss_rate = 1.0 - recall

    category_recalls.append(recall)

    all_detected += tp
    all_defects += total

    results[category] = {
        "threshold": float(threshold),
        "defect_count": int(total),
        "detected": tp,
        "missed": fn,
        "recall": float(recall),
        "miss_rate": float(miss_rate),
    }

    print("\n" + "=" * 60)
    print(f"CATEGORY: {category}")
    print("=" * 60)

    print(f"Threshold : {threshold:.6f}")
    print(f"Defects   : {total}")
    print(f"Detected  : {tp}")
    print(f"Missed    : {fn}")
    print(f"Recall    : {recall:.4f}")
    print(f"Miss rate : {miss_rate:.4f}")


# ------------------------------------------------------------
# Macro / pooled metrics
# ------------------------------------------------------------

macro_recall = float(
    np.mean(category_recalls)
)

pooled_recall = (
    all_detected / all_defects
    if all_defects > 0
    else 0.0
)


print("\n" + "=" * 60)
print("DEFECT-DEVELOPMENT CALIBRATION RESULTS")
print("=" * 60)

print(f"Macro recall  = {macro_recall:.4f}")
print(f"Pooled recall = {pooled_recall:.4f}")

print(f"\nDetected defects: {all_detected}/{all_defects}")


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

output = {
    "policy": "P99",
    "score_aggregation": "TOP5",
    "threshold_source": str(CALIBRATION_PATH),
    "evaluation_split": "defect_development_only",
    "final_test_used": False,
    "thresholds": thresholds,
    "categories": results,
    "macro_recall": macro_recall,
    "pooled_recall": pooled_recall,
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f, indent=2)


print("\n" + "=" * 60)
print("CALIBRATION DEFECT-DEVELOPMENT EVALUATION COMPLETE")
print("=" * 60)
print(f"Saved to: {OUTPUT_PATH}")