from pathlib import Path
import numpy as np
import torch
import torchvision
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_ROOT = Path("data")
CATEGORIES = ["bottle", "hazelnut"]

SEED = 42
IMAGE_SIZE = 224


# --------------------------------------------------
# Model
# --------------------------------------------------

weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT
model = torchvision.models.mobilenet_v3_small(weights=weights)

feature_extractor = torch.nn.Sequential(
    model.features[:9]
)

feature_extractor.eval()

transform = weights.transforms()


# --------------------------------------------------
# Feature extraction
# --------------------------------------------------

@torch.no_grad()
def extract_features(image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    features = feature_extractor(tensor)

    # Expected:
    # (1, 48, 14, 14)

    features = features.squeeze(0)

    # (48, 14, 14) -> (196, 48)
    patches = features.permute(1, 2, 0).reshape(-1, features.shape[0])

    return patches.numpy()


# --------------------------------------------------
# Load normal images
# --------------------------------------------------

def get_normal_images(category):
    train_dir = DATA_ROOT / category / "train" / "good"

    return sorted(train_dir.glob("*.png"))


# --------------------------------------------------
# Load test images
# --------------------------------------------------

def get_test_images(category):
    test_dir = DATA_ROOT / category / "test"

    normal = []
    defective = []

    for defect_dir in sorted(test_dir.iterdir()):

        if not defect_dir.is_dir():
            continue

        images = sorted(defect_dir.glob("*.png"))

        if defect_dir.name == "good":
            normal.extend(images)
        else:
            defective.extend(images)

    return normal, defective


# --------------------------------------------------
# Calculate aggregation scores
# --------------------------------------------------

def calculate_scores(distances):

    max_score = np.max(distances)

    top5_score = np.mean(
        np.sort(distances)[-5:]
    )

    top10_score = np.mean(
        np.sort(distances)[-10:]
    )

    return max_score, top5_score, top10_score


# --------------------------------------------------
# Evaluate one category
# --------------------------------------------------

def evaluate_category(category):

    print("\n" + "=" * 60)
    print(f"CATEGORY: {category}")
    print("=" * 60)

    normal_images = get_normal_images(category)

    rng = np.random.default_rng(SEED)
    indices = np.arange(len(normal_images))
    rng.shuffle(indices)

    split = int(len(indices) * 0.8)

    reference_indices = indices[:split]
    validation_indices = indices[split:]

    reference_images = [
        normal_images[i] for i in reference_indices
    ]

    validation_images = [
        normal_images[i] for i in validation_indices
    ]

    print(f"Normal images: {len(normal_images)}")
    print(f"Reference:     {len(reference_images)}")
    print(f"Validation:    {len(validation_images)}")

    # --------------------------------------------------
    # Build patch bank
    # --------------------------------------------------

    all_patches = []

    for i, image_path in enumerate(reference_images):

        patches = extract_features(image_path)
        all_patches.append(patches)

        if (i + 1) % 50 == 0:
            print(f"Reference features: {i + 1}/{len(reference_images)}")

    patch_bank = np.concatenate(all_patches, axis=0)

    print("Patch bank:", patch_bank.shape)

    # --------------------------------------------------
    # Nearest-neighbor index
    # --------------------------------------------------

    nn = NearestNeighbors(
        n_neighbors=1,
        metric="euclidean"
    )

    nn.fit(patch_bank)

    # --------------------------------------------------
    # Evaluate
    # --------------------------------------------------

    labels = []

    max_scores = []
    top5_scores = []
    top10_scores = []

    # Normal validation
    for image_path in validation_images:

        patches = extract_features(image_path)

        distances, _ = nn.kneighbors(
            patches,
            return_distance=True
        )

        distances = distances.ravel()

        max_score, top5, top10 = calculate_scores(distances)

        labels.append(0)

        max_scores.append(max_score)
        top5_scores.append(top5)
        top10_scores.append(top10)

    # Defective test images
    _, defective_images = get_test_images(category)

    print(f"Defective test images: {len(defective_images)}")

    for i, image_path in enumerate(defective_images):

        patches = extract_features(image_path)

        distances, _ = nn.kneighbors(
            patches,
            return_distance=True
        )

        distances = distances.ravel()

        max_score, top5, top10 = calculate_scores(distances)

        labels.append(1)

        max_scores.append(max_score)
        top5_scores.append(top5)
        top10_scores.append(top10)

    labels = np.array(labels)

    max_scores = np.array(max_scores)
    top5_scores = np.array(top5_scores)
    top10_scores = np.array(top10_scores)

    # --------------------------------------------------
    # AUROC
    # --------------------------------------------------

    max_auc = roc_auc_score(labels, max_scores)
    top5_auc = roc_auc_score(labels, top5_scores)
    top10_auc = roc_auc_score(labels, top10_scores)

    print("\nResults")
    print("-" * 40)

    print(f"MAX:        {max_auc:.4f}")
    print(f"TOP-5 MEAN: {top5_auc:.4f}")
    print(f"TOP-10 MEAN:{top10_auc:.4f}")

    return {
        "category": category,
        "max": max_auc,
        "top5": top5_auc,
        "top10": top10_auc,
    }


# --------------------------------------------------
# Main
# --------------------------------------------------

results = []

for category in CATEGORIES:
    result = evaluate_category(category)
    results.append(result)


# --------------------------------------------------
# Final comparison
# --------------------------------------------------

print("\n\n" + "=" * 60)
print("MULTI-CATEGORY AGGREGATION COMPARISON")
print("=" * 60)

print(
    f"{'Category':<15}"
    f"{'MAX':>10}"
    f"{'TOP5':>10}"
    f"{'TOP10':>10}"
)

print("-" * 45)

for r in results:

    print(
        f"{r['category']:<15}"
        f"{r['max']:>10.4f}"
        f"{r['top5']:>10.4f}"
        f"{r['top10']:>10.4f}"
    )

mean_max = np.mean([r["max"] for r in results])
mean_top5 = np.mean([r["top5"] for r in results])
mean_top10 = np.mean([r["top10"] for r in results])

print("-" * 45)

print(
    f"{'MEAN':<15}"
    f"{mean_max:>10.4f}"
    f"{mean_top5:>10.4f}"
    f"{mean_top10:>10.4f}"
)

print("\nDone.")