from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    precision_recall_curve,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
)


# ============================================================
# Configuration
# ============================================================

CATEGORY = "hazelnut"

DATA_DIR = Path("data") / CATEGORY

SEED = 42
VALIDATION_FRACTION = 0.20

DEVICE = torch.device("cpu")


# ============================================================
# Reproducibility
# ============================================================

rng = np.random.default_rng(SEED)


# ============================================================
# Model
# ============================================================

weights = MobileNet_V3_Small_Weights.DEFAULT

full_model = mobilenet_v3_small(weights=weights)

model = torch.nn.Sequential(
    *list(full_model.features[:9])
)

model.eval()
model.to(DEVICE)

preprocess = weights.transforms()


# ============================================================
# Feature extraction
# ============================================================

def extract_features(image_path):

    image = Image.open(image_path).convert("RGB")

    tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        features = model(tensor)

    # (1, 48, 14, 14)
    features = features.squeeze(0)

    # (48, 14, 14) -> (14, 14, 48)
    features = features.permute(1, 2, 0)

    # -> (196, 48)
    return features.reshape(
        -1,
        features.shape[2]
    ).numpy()


# ============================================================
# Load normal training images
# ============================================================

train_dir = DATA_DIR / "train" / "good"

all_train_images = sorted(
    train_dir.glob("*.png")
)

print("=" * 65)
print("THRESHOLD CALIBRATION")
print("=" * 65)

print(f"Category: {CATEGORY}")
print(
    f"Total normal images: "
    f"{len(all_train_images)}"
)


# ============================================================
# Split normal images
# ============================================================

indices = np.arange(
    len(all_train_images)
)

rng.shuffle(indices)

validation_size = int(
    len(indices) * VALIDATION_FRACTION
)

validation_indices = indices[
    :validation_size
]

reference_indices = indices[
    validation_size:
]

reference_images = [
    all_train_images[i]
    for i in reference_indices
]

validation_images = [
    all_train_images[i]
    for i in validation_indices
]

print(
    f"Reference images:   "
    f"{len(reference_images)}"
)

print(
    f"Validation images:  "
    f"{len(validation_images)}"
)


# ============================================================
# Build reference patch bank
# ============================================================

print()
print("Building reference patch bank...")

all_patches = []

start = time.perf_counter()

for i, image_path in enumerate(
    reference_images,
    start=1
):

    all_patches.append(
        extract_features(image_path)
    )

    print(
        f"\rReference features: "
        f"{i}/{len(reference_images)}",
        end=""
    )

print()

patch_bank = np.concatenate(
    all_patches,
    axis=0
)

print(
    f"Reference patch bank: "
    f"{patch_bank.shape}"
)

print(
    f"Build time: "
    f"{time.perf_counter() - start:.2f} s"
)


# ============================================================
# Build nearest-neighbor index
# ============================================================

nn = NearestNeighbors(
    n_neighbors=1,
    metric="euclidean"
)

nn.fit(patch_bank)


# ============================================================
# Generate normal validation scores
# ============================================================

print()
print("Scoring normal validation images...")

normal_scores = []

for i, image_path in enumerate(
    validation_images,
    start=1
):

    features = extract_features(
        image_path
    )

    distances, _ = nn.kneighbors(
        features
    )

    score = float(
        np.max(distances[:, 0])
    )

    normal_scores.append(score)

    print(
        f"\rValidation: "
        f"{i}/{len(validation_images)}",
        end=""
    )

print()

normal_scores = np.asarray(
    normal_scores
)


# ============================================================
# Normal score statistics
# ============================================================

print()
print("=" * 65)
print("NORMAL SCORE DISTRIBUTION")
print("=" * 65)

print(
    f"Minimum:   "
    f"{np.min(normal_scores):.6f}"
)

print(
    f"Mean:      "
    f"{np.mean(normal_scores):.6f}"
)

print(
    f"Median:    "
    f"{np.median(normal_scores):.6f}"
)

print(
    f"95th pct:  "
    f"{np.percentile(normal_scores, 95):.6f}"
)

print(
    f"99th pct:  "
    f"{np.percentile(normal_scores, 99):.6f}"
)

print(
    f"Maximum:   "
    f"{np.max(normal_scores):.6f}"
)


# ============================================================
# Candidate thresholds
# ============================================================

# Candidate values are derived from the observed
# normal distribution rather than invented constants.

thresholds = np.percentile(
    normal_scores,
    [90, 92.5, 95, 97.5, 99]
)


# ============================================================
# Evaluate false-positive behavior
# ============================================================

print()
print("=" * 65)
print("NORMAL-ONLY THRESHOLD ANALYSIS")
print("=" * 65)

print(
    f"{'Percentile':>12} "
    f"{'Threshold':>12} "
    f"{'False Positives':>18}"
)

print("-" * 50)

for percentile, threshold in zip(
    [90, 92.5, 95, 97.5, 99],
    thresholds
):

    predictions = (
        normal_scores >= threshold
    )

    false_positives = int(
        np.sum(predictions)
    )

    print(
        f"{percentile:>11.1f}% "
        f"{threshold:>12.6f} "
        f"{false_positives:>18}"
    )


# ============================================================
# Production candidate
# ============================================================

# We use the 99th percentile as the initial
# high-confidence defect boundary.
#
# This is NOT the final production threshold.
# It is a calibration candidate derived only
# from normal validation data.

defect_threshold = float(
    np.percentile(
        normal_scores,
        99
    )
)

# REVIEW boundary is placed at the 95th
# percentile as an initial uncertainty zone.

review_threshold = float(
    np.percentile(
        normal_scores,
        95
    )
)


# ============================================================
# Decision function
# ============================================================

def classify(score):

    if score < review_threshold:
        return "NORMAL"

    if score < defect_threshold:
        return "REVIEW"

    return "DEFECT"


# ============================================================
# Show validation decisions
# ============================================================

print()
print("=" * 65)
print("INITIAL DECISION POLICY")
print("=" * 65)

print(
    f"NORMAL:  score < "
    f"{review_threshold:.6f}"
)

print(
    f"REVIEW:  "
    f"{review_threshold:.6f} <= score < "
    f"{defect_threshold:.6f}"
)

print(
    f"DEFECT:  score >= "
    f"{defect_threshold:.6f}"
)


print()
print("Validation examples:")

for score in sorted(normal_scores)[-10:]:

    print(
        f"Score: {score:.6f} "
        f"-> {classify(score)}"
    )