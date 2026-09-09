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

IMAGE_SIZE = 224


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

    # (1, C, H, W)
    features = features.squeeze(0)

    # (C, H, W) -> (H*W, C)
    patches = features.permute(1, 2, 0).reshape(-1, features.shape[0])

    return patches.numpy()


# =========================
# Training images
# =========================

train_dir = DATA_DIR / "train" / "good"

train_images = sorted(train_dir.glob("*.png"))

print(f"Category: {CATEGORY}")
print(f"Normal training images: {len(train_images)}")


# =========================
# Build normal patch bank
# =========================

start = time.perf_counter()

all_patches = []

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
# Test images
# =========================

test_dir = DATA_DIR / "test"

test_images = []
labels = []

for defect_type_dir in sorted(test_dir.iterdir()):

    if not defect_type_dir.is_dir():
        continue

    is_defect = defect_type_dir.name != "good"

    for image_path in sorted(defect_type_dir.glob("*.png")):
        test_images.append(image_path)
        labels.append(int(is_defect))


# =========================
# Inference
# =========================

scores = []
latencies = []

for i, image_path in enumerate(test_images, start=1):

    start = time.perf_counter()

    patches = extract_features(image_path)

    distances, _ = nn.kneighbors(patches)

    patch_scores = distances[:, 0]

    image_score = float(np.max(patch_scores))

    latency = (time.perf_counter() - start) * 1000

    scores.append(image_score)
    latencies.append(latency)

    print(
        f"\rTesting: {i}/{len(test_images)}",
        end=""
    )

print()


# =========================
# Image-level evaluation
# =========================

labels = np.array(labels)
scores = np.array(scores)

image_auroc = roc_auc_score(labels, scores)

print()
print("=" * 50)
print("RESULTS")
print("=" * 50)

print(f"Category:       {CATEGORY}")
print(f"Test images:    {len(test_images)}")
print(f"Image AUROC:    {image_auroc:.4f}")
print(f"Mean CPU:       {np.mean(latencies):.2f} ms")
print(f"Median CPU:     {np.median(latencies):.2f} ms")
print(f"Min CPU:        {np.min(latencies):.2f} ms")
print(f"Max CPU:        {np.max(latencies):.2f} ms")