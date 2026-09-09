import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


DATA_DIR = r"D:\VisionForge\data"

CATEGORIES = [
    "screw",
    "capsule",
    "cable",
    "metal_nut",
]

DEVICE = torch.device("cpu")


weights = MobileNet_V3_Small_Weights.DEFAULT
transform = weights.transforms()

model = mobilenet_v3_small(weights=weights)
model.eval()
model.to(DEVICE)

# Current selected representation:
# MobileNetV3-Small features[:9] -> 48 x 14 x 14
feature_extractor = nn.Sequential(
    *list(model.features[:9])
)

feature_extractor.eval()


def extract_feature_map(path):
    image = Image.open(path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        feature_map = feature_extractor(tensor)

    return feature_map[0].permute(1, 2, 0).numpy()


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


def get_test_files(category):
    base = os.path.join(DATA_DIR, category, "test")

    results = []

    for defect_type in sorted(os.listdir(base)):
        folder = os.path.join(base, defect_type)

        if not os.path.isdir(folder):
            continue

        for f in sorted(os.listdir(folder)):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                results.append(
                    (os.path.join(folder, f), defect_type)
                )

    return results


def build_patch_bank(paths):
    banks = []

    for i, path in enumerate(paths):

        fmap = extract_feature_map(path)

        h, w, c = fmap.shape

        patches = fmap.reshape(h * w, c)

        banks.append(patches)

        if (i + 1) % 50 == 0:
            print(f"  reference: {i + 1}/{len(paths)}")

    return np.concatenate(banks, axis=0)


def analyze_category(category):

    print("\n" + "=" * 70)
    print(category.upper())
    print("=" * 70)

    train_good = get_train_good(category)

    rng = np.random.default_rng(42)

    indices = np.arange(len(train_good))
    rng.shuffle(indices)

    reference_count = int(len(indices) * 0.8)

    reference_paths = [
        train_good[i]
        for i in indices[:reference_count]
    ]

    print(f"Reference images: {len(reference_paths)}")

    print("Building patch bank...")

    patch_bank = build_patch_bank(reference_paths)

    print(f"Patch bank: {patch_bank.shape}")

    nn_model = NearestNeighbors(
        n_neighbors=1,
        metric="euclidean",
        algorithm="brute",
        n_jobs=-1
    )

    nn_model.fit(patch_bank)

    test_files = get_test_files(category)

    # Store scores for each defect type.
    type_scores = {}

    print("\nAnalyzing defect types...")

    for i, (path, defect_type) in enumerate(test_files):

        fmap = extract_feature_map(path)

        h, w, c = fmap.shape

        patches = fmap.reshape(h * w, c)

        distances, _ = nn_model.kneighbors(
            patches,
            n_neighbors=1
        )

        patch_scores = distances[:, 0]

        # Current production candidate aggregation.
        top5 = np.mean(
            np.sort(patch_scores)[-5:]
        )

        if defect_type not in type_scores:
            type_scores[defect_type] = []

        type_scores[defect_type].append({
            "path": path,
            "score": float(top5),
            "max": float(np.max(patch_scores)),
            "mean": float(np.mean(patch_scores)),
            "map": patch_scores.reshape(h, w)
        })

        if (i + 1) % 25 == 0:
            print(f"  analyzed: {i + 1}/{len(test_files)}")

    print("\nDEFECT TYPE SUMMARY")
    print("-" * 70)

    for defect_type, items in type_scores.items():

        scores = np.array([
            x["score"]
            for x in items
        ])

        max_scores = np.array([
            x["max"]
            for x in items
        ])

        mean_scores = np.array([
            x["mean"]
            for x in items
        ])

        print(
            f"{defect_type:25s} "
            f"n={len(scores):3d} "
            f"TOP5 median={np.median(scores):.4f} "
            f"MAX median={np.median(max_scores):.4f} "
            f"MEAN median={np.median(mean_scores):.4f}"
        )

    # Find the lowest-scoring defects.
    print("\nLOWEST-SCORING DEFECTS")
    print("-" * 70)

    all_items = []

    for defect_type, items in type_scores.items():
        for item in items:
            all_items.append(
                (
                    item["score"],
                    defect_type,
                    item["path"],
                    item
                )
            )

    all_items.sort(key=lambda x: x[0])

    for score, defect_type, path, item in all_items[:15]:

        print(
            f"{score:.4f} | "
            f"{defect_type:25s} | "
            f"{path}"
        )


if __name__ == "__main__":

    for category in CATEGORIES:
        analyze_category(category)

