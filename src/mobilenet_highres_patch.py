import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


DATA_DIR = r"D:\VisionForge\data"

CATEGORIES = [
    "screw",
    "capsule",
    "cable",
    "bottle",
    "hazelnut",
    "metal_nut",
]

DEVICE = torch.device("cpu")
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


weights = MobileNet_V3_Small_Weights.DEFAULT
transform = weights.transforms()

model = mobilenet_v3_small(weights=weights)
model.classifier = nn.Identity()
model.eval()
model.to(DEVICE)

# Layers 0-3 -> 24 x 28 x 28
feature_extractor = nn.Sequential(
    *list(model.features[:4])
)
feature_extractor.eval()


def load_image(path):
    image = Image.open(path).convert("RGB")
    return transform(image).unsqueeze(0)


@torch.no_grad()
def extract_patches(path):
    image = load_image(path).to(DEVICE)

    feature_map = feature_extractor(image)

    # Expected: [1, 24, 28, 28]
    _, channels, height, width = feature_map.shape

    patches = feature_map[0].permute(1, 2, 0).reshape(
        height * width,
        channels
    )

    return patches.numpy()


def get_train_good(category):
    folder = os.path.join(
        DATA_DIR,
        category,
        "train",
        "good"
    )

    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])


def get_test_images(category):
    base = os.path.join(DATA_DIR, category, "test")

    good = []
    defects = []

    for defect_type in sorted(os.listdir(base)):
        folder = os.path.join(base, defect_type)

        if not os.path.isdir(folder):
            continue

        files = sorted([
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        if defect_type == "good":
            good.extend(files)
        else:
            defects.extend(files)

    return good, defects


def build_patch_bank(paths):
    all_patches = []

    for i, path in enumerate(paths):
        patches = extract_patches(path)
        all_patches.append(patches)

        if (i + 1) % 50 == 0:
            print(f"  processed {i + 1}/{len(paths)}")

    return np.concatenate(all_patches, axis=0)


def score_image(path, nn_model):
    patches = extract_patches(path)

    distances, _ = nn_model.kneighbors(
        patches,
        n_neighbors=1
    )

    patch_scores = distances[:, 0]

    # Same aggregation used by the current Layer-8 system:
    top_k = min(5, len(patch_scores))

    score = np.mean(
        np.sort(patch_scores)[-top_k:]
    )

    return score


def evaluate_category(category):
    print("\n" + "=" * 70)
    print(category.upper())
    print("=" * 70)

    train_good = get_train_good(category)
    test_good, test_defects = get_test_images(category)

    # Same 80/20 reference split used in previous experiments.
    rng = np.random.default_rng(SEED)

    indices = np.arange(len(train_good))
    rng.shuffle(indices)

    reference_count = int(len(indices) * 0.8)

    reference_paths = [
        train_good[i]
        for i in indices[:reference_count]
    ]

    print(f"Train good:       {len(train_good)}")
    print(f"Reference images: {len(reference_paths)}")
    print(f"Test good:        {len(test_good)}")
    print(f"Test defects:     {len(test_defects)}")

    print("\nBuilding 28x28 patch bank...")

    patch_bank = build_patch_bank(reference_paths)

    print(f"Patch bank shape: {patch_bank.shape}")

    nn_model = NearestNeighbors(
        n_neighbors=1,
        metric="euclidean",
        algorithm="brute",
        n_jobs=-1
    )

    nn_model.fit(patch_bank)

    scores = []
    labels = []

    print("\nScoring good images...")

    for i, path in enumerate(test_good):
        score = score_image(path, nn_model)

        scores.append(score)
        labels.append(0)

        if (i + 1) % 20 == 0:
            print(f"  good: {i + 1}/{len(test_good)}")

    print("Scoring defect images...")

    for i, path in enumerate(test_defects):
        score = score_image(path, nn_model)

        scores.append(score)
        labels.append(1)

        if (i + 1) % 20 == 0:
            print(f"  defect: {i + 1}/{len(test_defects)}")

    scores = np.asarray(scores)
    labels = np.asarray(labels)

    auroc = roc_auc_score(labels, scores)

    good_scores = scores[labels == 0]
    defect_scores = scores[labels == 1]

    print("\nRESULT")
    print("-" * 40)
    print(f"Image AUROC:       {auroc:.4f}")
    print(f"Good median:       {np.median(good_scores):.4f}")
    print(f"Defect median:     {np.median(defect_scores):.4f}")
    print(f"Good max:          {np.max(good_scores):.4f}")
    print(f"Defect min:        {np.min(defect_scores):.4f}")

    return auroc


if __name__ == "__main__":

    results = {}

    for category in CATEGORIES:
        results[category] = evaluate_category(category)

    print("\n\n" + "=" * 70)
    print("HIGH-RES REPRESENTATION SUMMARY")
    print("=" * 70)

    for category, score in results.items():
        print(f"{category:12s}: {score:.4f}")
