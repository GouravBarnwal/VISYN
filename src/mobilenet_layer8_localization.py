from pathlib import Path
import time

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


# =========================
# Configuration
# =========================

CATEGORY = "hazelnut"

DATA_DIR = Path("data") / CATEGORY
DEVICE = torch.device("cpu")


# =========================
# Model
# =========================

weights = MobileNet_V3_Small_Weights.DEFAULT

full_model = mobilenet_v3_small(weights=weights)

model = torch.nn.Sequential(
    *list(full_model.features[:9])
)

model.eval()
model.to(DEVICE)

preprocess = weights.transforms()


# =========================
# Feature extraction
# =========================

def extract_features(image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        features = model(tensor)

    # Expected shape:
    # (1, 48, 14, 14)

    features = features.squeeze(0)

    # (48, 14, 14)
    #      ↓
    # (14, 14, 48)

    patches = features.permute(1, 2, 0)

    # (14, 14, 48)
    #      ↓
    # (196, 48)

    patches = patches.reshape(-1, features.shape[0])

    return patches.numpy()


# =========================
# Build normal patch bank
# =========================

train_dir = DATA_DIR / "train" / "good"

train_images = sorted(train_dir.glob("*.png"))

print(f"Category: {CATEGORY}")
print(f"Normal training images: {len(train_images)}")

all_patches = []

start = time.perf_counter()

for i, image_path in enumerate(train_images, start=1):

    patches = extract_features(image_path)
    all_patches.append(patches)

    print(
        f"\rBuilding patch bank: {i}/{len(train_images)}",
        end=""
    )

print()

patch_bank = np.concatenate(all_patches, axis=0)

build_time = time.perf_counter() - start

print(f"Patch bank shape: {patch_bank.shape}")
print(f"Reference build time: {build_time:.2f} seconds")


# =========================
# Nearest-neighbor index
# =========================

nn = NearestNeighbors(
    n_neighbors=1,
    metric="euclidean"
)

nn.fit(patch_bank)


# =========================
# Pixel localization
# =========================

test_dir = DATA_DIR / "test"
ground_truth_dir = DATA_DIR / "ground_truth"

pixel_scores = []
pixel_labels = []

latencies = []

defect_count = 0

for defect_type_dir in sorted(test_dir.iterdir()):

    if not defect_type_dir.is_dir():
        continue

    if defect_type_dir.name == "good":
        continue

    gt_dir = ground_truth_dir / defect_type_dir.name

    for image_path in sorted(defect_type_dir.glob("*.png")):

        gt_path = gt_dir / (image_path.stem + "_mask.png")

        if not gt_path.exists():
            print(f"\nMissing mask: {gt_path}")
            continue

        start = time.perf_counter()

        # Extract 14x14 local features
        patches = extract_features(image_path)

        # Find nearest normal patch
        distances, _ = nn.kneighbors(patches)

        # 196 anomaly scores
        patch_scores = distances[:, 0]

        # Convert back to 14x14
        anomaly_map = patch_scores.reshape(14, 14)

        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)

        # Resize anomaly map to original image resolution
        image = cv2.imread(str(image_path))

        height, width = image.shape[:2]

        anomaly_map = cv2.resize(
            anomaly_map,
            (width, height),
            interpolation=cv2.INTER_LINEAR
        )

        # Load ground-truth mask
        mask = cv2.imread(
            str(gt_path),
            cv2.IMREAD_GRAYSCALE
        )

        mask = (mask > 0).astype(np.uint8)

        # Flatten
        pixel_scores.extend(anomaly_map.flatten())
        pixel_labels.extend(mask.flatten())

        defect_count += 1

        print(
            f"\rEvaluating localization: {defect_count}",
            end=""
        )

print()


# =========================
# Pixel AUROC
# =========================

pixel_scores = np.asarray(pixel_scores)
pixel_labels = np.asarray(pixel_labels)

pixel_auroc = roc_auc_score(
    pixel_labels,
    pixel_scores
)


# =========================
# Results
# =========================

print()
print("=" * 50)
print("LOCALIZATION RESULTS")
print("=" * 50)

print(f"Category:          {CATEGORY}")
print(f"Defect images:     {defect_count}")
print(f"Pixel AUROC:       {pixel_auroc:.4f}")
print(f"Mean CPU:          {np.mean(latencies):.2f} ms")
print(f"Median CPU:        {np.median(latencies):.2f} ms")
print(f"Min CPU:           {np.min(latencies):.2f} ms")
print(f"Max CPU:           {np.max(latencies):.2f} ms")