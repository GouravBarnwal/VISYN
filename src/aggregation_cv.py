from pathlib import Path
import json
import random

import numpy as np
import torch
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from torchvision import models, transforms


DATA_ROOT = Path("data")
SPLIT_FILE = Path("artifacts/splits/normal_splits.json")
OUTPUT_FILE = Path("artifacts/aggregation_cv.json")

CATEGORIES = [
    "bottle",
    "hazelnut",
    "cable",
    "capsule",
    "screw",
    "metal_nut",
]

SEED = 42
N_FOLDS = 5

DEVICE = torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_model():
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)

    extractor = torch.nn.Sequential(
        *list(model.features.children())[:9]
    )

    extractor.eval()
    extractor.to(DEVICE)

    return extractor


def extract_patches(model, image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        features = model(tensor)

    # (1, 48, 14, 14) -> (196, 48)
    features = features.squeeze(0)
    features = features.permute(1, 2, 0)
    features = features.reshape(-1, features.shape[-1])

    features = torch.nn.functional.normalize(
        features,
        p=2,
        dim=1,
    )

    return features.cpu().numpy().astype(np.float32)


def score_image(model, image_path, index):
    patches = extract_patches(model, image_path)

    distances, _ = index.kneighbors(patches)
    distances = distances[:, 0]

    max_score = float(np.max(distances))

    sorted_distances = np.sort(distances)

    top5_score = float(
        np.mean(sorted_distances[-5:])
    )

    top10_score = float(
        np.mean(sorted_distances[-10:])
    )

    return max_score, top5_score, top10_score


def make_folds(images):
    images = list(images)

    rng = random.Random(SEED)
    rng.shuffle(images)

    folds = [[] for _ in range(N_FOLDS)]

    for i, image in enumerate(images):
        folds[i % N_FOLDS].append(image)

    return folds


def evaluate_fold(model, reference_images, development_images):
    patch_bank_parts = []

    for image_path in reference_images:
        path = DATA_ROOT / image_path
        patch_bank_parts.append(
            extract_patches(model, path)
        )

    patch_bank = np.concatenate(
        patch_bank_parts,
        axis=0,
    )

    index = NearestNeighbors(
        n_neighbors=1,
        metric="euclidean",
    )

    index.fit(patch_bank)

    scores = {
        "MAX": [],
        "TOP5": [],
        "TOP10": [],
    }

    for image_path in development_images:
        path = DATA_ROOT / image_path

        result = score_image(
            model,
            path,
            index,
        )

        scores["MAX"].append(result[0])
        scores["TOP5"].append(result[1])
        scores["TOP10"].append(result[2])

    return scores


def main():
    with open(SPLIT_FILE, "r") as f:
        split_data = json.load(f)

    model = load_model()

    all_results = {}

    for category in CATEGORIES:
        print("=" * 60)
        print(f"CATEGORY: {category}")

        images = (
            split_data["categories"][category]["reference"]
            + split_data["categories"][category]["development"]
        )

        folds = make_folds(images)

        category_results = {
            "MAX": [],
            "TOP5": [],
            "TOP10": [],
        }

        for fold_id in range(N_FOLDS):
            development = folds[fold_id]

            reference = []

            for other_fold_id in range(N_FOLDS):
                if other_fold_id != fold_id:
                    reference.extend(folds[other_fold_id])

            print(
                f"Fold {fold_id + 1}/{N_FOLDS}: "
                f"reference={len(reference)}, "
                f"development={len(development)}"
            )

            scores = evaluate_fold(
                model,
                reference,
                development,
            )

            # There are no defect labels in this CV.
            # We therefore record the score distribution
            # for stability analysis.
            for method in category_results:
                values = np.asarray(scores[method])

                category_results[method].append({
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "median": float(np.median(values)),
                })

        all_results[category] = category_results

        print()

        for method in ["MAX", "TOP5", "TOP10"]:
            means = [
                fold["mean"]
                for fold in category_results[method]
            ]

            print(
                f"{method:<6} "
                f"mean={np.mean(means):.4f} "
                f"std={np.std(means):.4f}"
            )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(OUTPUT_FILE, "w") as f:
        json.dump(
            {
                "seed": SEED,
                "folds": N_FOLDS,
                "results": all_results,
            },
            f,
            indent=2,
        )

    print()
    print("=" * 60)
    print(f"Saved: {OUTPUT_FILE}")
    print("Done.")


if __name__ == "__main__":
    main()