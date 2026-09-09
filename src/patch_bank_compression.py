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

FRACTIONS = [1.0, 0.5, 0.25, 0.10]

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

    # (48, 14, 14)
    # -> (14, 14, 48)
    features = features.permute(1, 2, 0)

    # -> (196, 48)
    patches = features.reshape(-1, features.shape[2])

    return patches.numpy()


# ============================================================
# Build full normal patch bank
# ============================================================

train_dir = DATA_DIR / "train" / "good"

train_images = sorted(train_dir.glob("*.png"))

print("=" * 60)
print("PATCH BANK COMPRESSION BENCHMARK")
print("=" * 60)

print(f"Category: {CATEGORY}")
print(f"Normal training images: {len(train_images)}")

all_patches = []

build_start = time.perf_counter()

for i, image_path in enumerate(train_images, start=1):

    patches = extract_features(image_path)
    all_patches.append(patches)

    print(
        f"\rBuilding full patch bank: "
        f"{i}/{len(train_images)}",
        end=""
    )

print()

patch_bank = np.concatenate(all_patches, axis=0)

build_time = time.perf_counter() - build_start

print(f"Full patch bank: {patch_bank.shape}")
print(f"Build time: {build_time:.2f} s")


# ============================================================
# Prepare test images
# ============================================================

test_images = []
labels = []

test_dir = DATA_DIR / "test"

for defect_type_dir in sorted(test_dir.iterdir()):

    if not defect_type_dir.is_dir():
        continue

    is_defect = defect_type_dir.name != "good"

    for image_path in sorted(defect_type_dir.glob("*.png")):

        test_images.append(image_path)
        labels.append(int(is_defect))


labels = np.asarray(labels)


# ============================================================
# Pre-extract test features
# ============================================================

print()
print("Extracting test features...")

test_features = []

for i, image_path in enumerate(test_images, start=1):

    features = extract_features(image_path)
    test_features.append(features)

    print(
        f"\rTest features: {i}/{len(test_images)}",
        end=""
    )

print()


# ============================================================
# Pixel ground-truth
# ============================================================

ground_truth_dir = DATA_DIR / "ground_truth"

def get_mask(image_path):

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
        raise FileNotFoundError(mask_path)

    return (mask > 0).astype(np.uint8)


# ============================================================
# Run one compression level
# ============================================================

def evaluate_fraction(fraction):

    rng = np.random.default_rng(SEED)

    full_size = len(patch_bank)

    if fraction == 1.0:
        selected_indices = np.arange(full_size)
    else:
        selected_size = int(full_size * fraction)

        selected_indices = rng.choice(
            full_size,
            size=selected_size,
            replace=False,
        )

    reduced_bank = patch_bank[selected_indices]

    print()
    print("-" * 60)
    print(
        f"Testing {fraction * 100:.0f}% "
        f"({len(reduced_bank):,} patches)"
    )
    print("-" * 60)

    # Build nearest-neighbor index
    index_start = time.perf_counter()

    nn = NearestNeighbors(
        n_neighbors=1,
        metric="euclidean",
    )

    nn.fit(reduced_bank)

    index_time = time.perf_counter() - index_start

    image_scores = []
    pixel_scores = []
    pixel_labels = []

    latencies = []

    for i, (image_path, features) in enumerate(
        zip(test_images, test_features),
        start=1,
    ):

        start = time.perf_counter()

        distances, _ = nn.kneighbors(features)

        patch_distances = distances[:, 0]

        image_score = float(
            np.max(patch_distances)
        )

        latency = (
            time.perf_counter() - start
        ) * 1000

        image_scores.append(image_score)
        latencies.append(latency)

        # Only defect images have ground-truth masks
        if image_path.parent.name != "good":

            anomaly_map = patch_distances.reshape(
                14,
                14,
            )

            image = cv2.imread(str(image_path))

            height, width = image.shape[:2]

            anomaly_map = cv2.resize(
                anomaly_map,
                (width, height),
                interpolation=cv2.INTER_LINEAR,
            )

            mask = get_mask(image_path)

            pixel_scores.extend(
                anomaly_map.flatten()
            )

            pixel_labels.extend(
                mask.flatten()
            )

        print(
            f"\rInference: {i}/{len(test_images)}",
            end=""
        )

    print()

    image_scores = np.asarray(image_scores)

    image_auroc = roc_auc_score(
        labels,
        image_scores,
    )

    pixel_scores = np.asarray(pixel_scores)
    pixel_labels = np.asarray(pixel_labels)

    pixel_auroc = roc_auc_score(
        pixel_labels,
        pixel_scores,
    )

    return {
        "fraction": fraction,
        "patches": len(reduced_bank),
        "index_time": index_time,
        "image_auroc": image_auroc,
        "pixel_auroc": pixel_auroc,
        "mean_latency": np.mean(latencies),
        "median_latency": np.median(latencies),
    }


# ============================================================
# Run benchmark
# ============================================================

results = []

for fraction in FRACTIONS:

    result = evaluate_fraction(fraction)

    results.append(result)


# ============================================================
# Final table
# ============================================================

print()
print("=" * 80)
print("FINAL COMPRESSION RESULTS")
print("=" * 80)

print(
    f"{'Bank':>8} "
    f"{'Patches':>10} "
    f"{'Img AUROC':>12} "
    f"{'Pixel AUROC':>13} "
    f"{'Mean ms':>10} "
    f"{'Median ms':>12}"
)

print("-" * 80)

for r in results:

    print(
        f"{r['fraction'] * 100:>7.0f}% "
        f"{r['patches']:>10,} "
        f"{r['image_auroc']:>12.4f} "
        f"{r['pixel_auroc']:>13.4f} "
        f"{r['mean_latency']:>10.2f} "
        f"{r['median_latency']:>12.2f}"
    )