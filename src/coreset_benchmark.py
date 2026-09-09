from pathlib import Path
import time

import cv2
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
CORESET_FRACTION = 0.10

DEVICE = torch.device("cpu")


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
# Build full patch bank
# ============================================================

train_dir = DATA_DIR / "train" / "good"
train_images = sorted(train_dir.glob("*.png"))

print("=" * 65)
print("CORESET BENCHMARK")
print("=" * 65)

print(f"Category: {CATEGORY}")
print(f"Normal images: {len(train_images)}")

all_patches = []

start = time.perf_counter()

for i, image_path in enumerate(train_images, start=1):

    all_patches.append(
        extract_features(image_path)
    )

    print(
        f"\rExtracting training features: "
        f"{i}/{len(train_images)}",
        end=""
    )

print()

patch_bank = np.concatenate(
    all_patches,
    axis=0
)

print(
    f"Full patch bank: "
    f"{patch_bank.shape}"
)

print(
    f"Feature extraction time: "
    f"{time.perf_counter() - start:.2f} s"
)


# ============================================================
# Normalize descriptors
# ============================================================

norms = np.linalg.norm(
    patch_bank,
    axis=1,
    keepdims=True
)

patch_bank = patch_bank / np.maximum(
    norms,
    1e-12
)


# ============================================================
# Farthest-point coreset selection
# ============================================================

def build_coreset(features, fraction, seed):

    rng = np.random.default_rng(seed)

    n = len(features)

    target_size = max(
        1,
        int(n * fraction)
    )

    selected = np.empty(
        target_size,
        dtype=np.int32
    )

    # Start from a random patch
    selected[0] = rng.integers(n)

    # Distance from every point to current selection
    min_distances = np.linalg.norm(
        features - features[selected[0]],
        axis=1
    )

    for i in range(1, target_size):

        # Select the point furthest from
        # the current selected set
        next_index = np.argmax(
            min_distances
        )

        selected[i] = next_index

        # Update distance to nearest selected point
        new_distances = np.linalg.norm(
            features - features[next_index],
            axis=1
        )

        min_distances = np.minimum(
            min_distances,
            new_distances
        )

        if i % 500 == 0 or i == target_size - 1:
            print(
                f"\rCoreset selection: "
                f"{i + 1}/{target_size}",
                end=""
            )

    print()

    return selected


print()
print(
    f"Building intelligent "
    f"{CORESET_FRACTION * 100:.0f}% coreset..."
)

coreset_start = time.perf_counter()

selected_indices = build_coreset(
    patch_bank,
    CORESET_FRACTION,
    SEED
)

coreset_time = (
    time.perf_counter() -
    coreset_start
)

coreset = patch_bank[selected_indices]

print(
    f"Coreset shape: "
    f"{coreset.shape}"
)

print(
    f"Coreset construction time: "
    f"{coreset_time:.2f} s"
)


# ============================================================
# Test images
# ============================================================

test_images = []
labels = []

test_dir = DATA_DIR / "test"

for defect_type_dir in sorted(test_dir.iterdir()):

    if not defect_type_dir.is_dir():
        continue

    is_defect = (
        defect_type_dir.name != "good"
    )

    for image_path in sorted(
        defect_type_dir.glob("*.png")
    ):

        test_images.append(image_path)
        labels.append(int(is_defect))

labels = np.asarray(labels)


# ============================================================
# Extract test features once
# ============================================================

print()
print("Extracting test features...")

test_features = []

for i, image_path in enumerate(
    test_images,
    start=1
):

    test_features.append(
        extract_features(image_path)
    )

    print(
        f"\rTest features: "
        f"{i}/{len(test_images)}",
        end=""
    )

print()


# ============================================================
# Ground-truth helper
# ============================================================

ground_truth_dir = (
    DATA_DIR / "ground_truth"
)


def load_mask(image_path):

    defect_type = image_path.parent.name

    mask_path = (
        ground_truth_dir
        / defect_type
        / f"{image_path.stem}_mask.png"
    )

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:
        raise FileNotFoundError(
            mask_path
        )

    return (mask > 0).astype(np.uint8)


# ============================================================
# Build NN index
# ============================================================

index_start = time.perf_counter()

nn = NearestNeighbors(
    n_neighbors=1,
    metric="euclidean"
)

nn.fit(coreset)

index_time = (
    time.perf_counter() -
    index_start
)


# ============================================================
# Evaluation
# ============================================================

image_scores = []

pixel_scores = []
pixel_labels = []

latencies = []

for i, (image_path, features) in enumerate(
    zip(test_images, test_features),
    start=1
):

    start = time.perf_counter()

    distances, _ = nn.kneighbors(
        features
    )

    patch_distances = distances[:, 0]

    image_score = float(
        np.max(patch_distances)
    )

    latency = (
        time.perf_counter() -
        start
    ) * 1000

    image_scores.append(
        image_score
    )

    latencies.append(
        latency
    )

    # Localization for defect images
    if image_path.parent.name != "good":

        anomaly_map = (
            patch_distances
            .reshape(14, 14)
        )

        image = cv2.imread(
            str(image_path)
        )

        height, width = image.shape[:2]

        anomaly_map = cv2.resize(
            anomaly_map,
            (width, height),
            interpolation=cv2.INTER_LINEAR
        )

        mask = load_mask(
            image_path
        )

        pixel_scores.extend(
            anomaly_map.flatten()
        )

        pixel_labels.extend(
            mask.flatten()
        )

    print(
        f"\rEvaluating: "
        f"{i}/{len(test_images)}",
        end=""
    )

print()


# ============================================================
# Metrics
# ============================================================

image_scores = np.asarray(
    image_scores
)

pixel_scores = np.asarray(
    pixel_scores
)

pixel_labels = np.asarray(
    pixel_labels
)

image_auroc = roc_auc_score(
    labels,
    image_scores
)

pixel_auroc = roc_auc_score(
    pixel_labels,
    pixel_scores
)


# ============================================================
# Results
# ============================================================

print()
print("=" * 65)
print("INTELLIGENT CORESET RESULTS")
print("=" * 65)

print(
    f"Category:           {CATEGORY}"
)

print(
    f"Coreset fraction:   "
    f"{CORESET_FRACTION * 100:.0f}%"
)

print(
    f"Full bank:          "
    f"{len(patch_bank):,}"
)

print(
    f"Coreset:            "
    f"{len(coreset):,}"
)

print(
    f"Image AUROC:        "
    f"{image_auroc:.4f}"
)

print(
    f"Pixel AUROC:        "
    f"{pixel_auroc:.4f}"
)

print(
    f"Mean search:        "
    f"{np.mean(latencies):.2f} ms"
)

print(
    f"Median search:      "
    f"{np.median(latencies):.2f} ms"
)

print(
    f"Index build:        "
    f"{index_time:.4f} s"
)

print(
    f"Coreset creation:   "
    f"{coreset_time:.2f} s"
)