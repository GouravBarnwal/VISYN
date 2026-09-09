from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
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

    features = features.squeeze(0)

    features = features.permute(1, 2, 0)

    return features.reshape(
        -1,
        features.shape[2]
    ).numpy()


# ============================================================
# Split normal training images
# ============================================================

train_images = sorted(
    (DATA_DIR / "train" / "good").glob("*.png")
)

indices = np.arange(len(train_images))

rng.shuffle(indices)

validation_size = int(
    len(indices) * VALIDATION_FRACTION
)

validation_indices = indices[:validation_size]
reference_indices = indices[validation_size:]

reference_images = [
    train_images[i]
    for i in reference_indices
]

normal_validation_images = [
    train_images[i]
    for i in validation_indices
]


# ============================================================
# Build normal reference bank
# ============================================================

print("=" * 70)
print("THRESHOLD SWEEP")
print("=" * 70)

print(f"Category: {CATEGORY}")
print(f"Reference images: {len(reference_images)}")
print(f"Normal validation images: {len(normal_validation_images)}")

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


# ============================================================
# Nearest-neighbor index
# ============================================================

nn = NearestNeighbors(
    n_neighbors=1,
    metric="euclidean"
)

nn.fit(patch_bank)


# ============================================================
# Score an image
# ============================================================

def anomaly_score(image_path):

    features = extract_features(image_path)

    distances, _ = nn.kneighbors(features)

    return float(
        np.max(distances[:, 0])
    )


# ============================================================
# Score normal validation
# ============================================================

print()
print("Scoring normal validation images...")

normal_scores = []

for i, image_path in enumerate(
    normal_validation_images,
    start=1
):

    normal_scores.append(
        anomaly_score(image_path)
    )

    print(
        f"\rNormal: "
        f"{i}/{len(normal_validation_images)}",
        end=""
    )

print()


# ============================================================
# Collect defective images
# ============================================================

defect_images = []

test_dir = DATA_DIR / "test"

for defect_dir in sorted(test_dir.iterdir()):

    if not defect_dir.is_dir():
        continue

    if defect_dir.name == "good":
        continue

    defect_images.extend(
        sorted(defect_dir.glob("*.png"))
    )


# ============================================================
# Score defects
# ============================================================

print()
print("Scoring defective images...")

defect_scores = []

for i, image_path in enumerate(
    defect_images,
    start=1
):

    defect_scores.append(
        anomaly_score(image_path)
    )

    print(
        f"\rDefects: "
        f"{i}/{len(defect_images)}",
        end=""
    )

print()


# ============================================================
# Combine validation data
# ============================================================

scores = np.concatenate(
    [
        np.asarray(normal_scores),
        np.asarray(defect_scores),
    ]
)

labels = np.concatenate(
    [
        np.zeros(len(normal_scores), dtype=int),
        np.ones(len(defect_scores), dtype=int),
    ]
)


# ============================================================
# Score distributions
# ============================================================

print()
print("=" * 70)
print("SCORE DISTRIBUTIONS")
print("=" * 70)

print()
print("NORMAL")
print(
    f"Min:      {np.min(normal_scores):.6f}"
)
print(
    f"Median:   {np.median(normal_scores):.6f}"
)
print(
    f"Mean:     {np.mean(normal_scores):.6f}"
)
print(
    f"95th:     {np.percentile(normal_scores, 95):.6f}"
)
print(
    f"99th:     {np.percentile(normal_scores, 99):.6f}"
)
print(
    f"Max:      {np.max(normal_scores):.6f}"
)

print()
print("DEFECT")
print(
    f"Min:      {np.min(defect_scores):.6f}"
)
print(
    f"Median:   {np.median(defect_scores):.6f}"
)
print(
    f"Mean:     {np.mean(defect_scores):.6f}"
)
print(
    f"25th:     {np.percentile(defect_scores, 25):.6f}"
)
print(
    f"50th:     {np.percentile(defect_scores, 50):.6f}"
)
print(
    f"75th:     {np.percentile(defect_scores, 75):.6f}"
)
print(
    f"Max:      {np.max(defect_scores):.6f}"
)


# ============================================================
# Threshold sweep
# ============================================================

candidate_thresholds = np.unique(
    np.percentile(
        scores,
        np.arange(50, 100, 2.5)
    )
)

print()
print("=" * 90)
print("THRESHOLD SWEEP")
print("=" * 90)

print(
    f"{'Threshold':>12} "
    f"{'Precision':>12} "
    f"{'Recall':>12} "
    f"{'F1':>12} "
    f"{'FP':>8} "
    f"{'FN':>8}"
)

print("-" * 90)

results = []

for threshold in candidate_thresholds:

    predictions = (
        scores >= threshold
    ).astype(int)

    precision = precision_score(
        labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        labels,
        predictions
    )

    tn, fp, fn, tp = cm.ravel()

    results.append(
        (
            threshold,
            precision,
            recall,
            f1,
            fp,
            fn
        )
    )

    print(
        f"{threshold:>12.6f} "
        f"{precision:>12.4f} "
        f"{recall:>12.4f} "
        f"{f1:>12.4f} "
        f"{fp:>8} "
        f"{fn:>8}"
    )


# ============================================================
# Best F1 threshold
# ============================================================

best = max(
    results,
    key=lambda x: x[3]
)

print()
print("=" * 70)
print("BEST F1 CANDIDATE")
print("=" * 70)

print(
    f"Threshold: {best[0]:.6f}"
)

print(
    f"Precision: {best[1]:.4f}"
)

print(
    f"Recall:    {best[2]:.4f}"
)

print(
    f"F1:        {best[3]:.4f}"
)

print(
    f"False positives: {best[4]}"
)

print(
    f"False negatives: {best[5]}"
)

print()
print(
    "NOTE: This is a calibration experiment. "
    "The threshold is NOT locked."
)