from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
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
# Split normal training data
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
# Build reference bank
# ============================================================

print("=" * 75)
print("PATCH-SCORE AGGREGATION ABLATION")
print("=" * 75)

print(f"Category: {CATEGORY}")
print(f"Reference images: {len(reference_images)}")
print(f"Normal validation: {len(normal_validation_images)}")

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
    f"Feature extraction: "
    f"{time.perf_counter() - start:.2f} s"
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
# Image score aggregation methods
# ============================================================

def aggregate_scores(patch_scores):

    sorted_scores = np.sort(
        patch_scores
    )[::-1]

    return {
        "max": float(
            sorted_scores[0]
        ),

        "top5_mean": float(
            np.mean(
                sorted_scores[:5]
            )
        ),

        "top10_mean": float(
            np.mean(
                sorted_scores[:10]
            )
        ),

        "p95": float(
            np.percentile(
                patch_scores,
                95
            )
        ),
    }


# ============================================================
# Collect all evaluation images
# ============================================================

evaluation_images = []
labels = []

# Normal validation images
for image_path in normal_validation_images:

    evaluation_images.append(
        image_path
    )

    labels.append(0)


# Defective test images
test_dir = DATA_DIR / "test"

for defect_dir in sorted(test_dir.iterdir()):

    if not defect_dir.is_dir():
        continue

    if defect_dir.name == "good":
        continue

    for image_path in sorted(
        defect_dir.glob("*.png")
    ):

        evaluation_images.append(
            image_path
        )

        labels.append(1)


labels = np.asarray(labels)


# ============================================================
# Extract features and calculate patch distances
# ============================================================

print()
print("Calculating patch distances...")

all_image_patch_scores = []

for i, image_path in enumerate(
    evaluation_images,
    start=1
):

    features = extract_features(
        image_path
    )

    distances, _ = nn.kneighbors(
        features
    )

    patch_scores = distances[:, 0]

    all_image_patch_scores.append(
        patch_scores
    )

    print(
        f"\rImages: "
        f"{i}/{len(evaluation_images)}",
        end=""
    )

print()


# ============================================================
# Evaluate aggregation methods
# ============================================================

methods = [
    "max",
    "top5_mean",
    "top10_mean",
    "p95",
]

results = {}

for method in methods:

    scores = []

    for patch_scores in all_image_patch_scores:

        aggregated = aggregate_scores(
            patch_scores
        )

        scores.append(
            aggregated[method]
        )

    scores = np.asarray(scores)

    auroc = roc_auc_score(
        labels,
        scores
    )

    results[method] = {
        "scores": scores,
        "auroc": auroc,
    }


# ============================================================
# Results
# ============================================================

print()
print("=" * 75)
print("IMAGE-LEVEL AGGREGATION RESULTS")
print("=" * 75)

print(
    f"{'Method':<18}"
    f"{'AUROC':>12}"
)

print("-" * 35)

for method in methods:

    print(
        f"{method:<18}"
        f"{results[method]['auroc']:>12.4f}"
    )


# ============================================================
# Score distribution information
# ============================================================

print()
print("=" * 75)
print("SCORE SEPARATION")
print("=" * 75)

normal_count = len(normal_validation_images)

for method in methods:

    scores = results[method]["scores"]

    normal_scores = scores[:normal_count]
    defect_scores = scores[normal_count:]

    print()
    print(method.upper())

    print(
        f"Normal median: "
        f"{np.median(normal_scores):.6f}"
    )

    print(
        f"Normal 95th:  "
        f"{np.percentile(normal_scores, 95):.6f}"
    )

    print(
        f"Defect median: "
        f"{np.median(defect_scores):.6f}"
    )

    print(
        f"Defect 25th:   "
        f"{np.percentile(defect_scores, 25):.6f}"
    )