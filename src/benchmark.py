from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from torchvision import models, transforms


DATASET_ROOT = Path("data/bottle")
PATCH_BANK_PATH = Path("artifacts/bottle_patch_bank.npy")


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_global_model():
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    model.fc = torch.nn.Identity()
    model.eval()

    return model


def load_spatial_model():
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    model = torch.nn.Sequential(
        *list(model.children())[:-2]
    )

    model.eval()

    return model


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


def extract_global(model, image_path):
    tensor = load_image(image_path)

    with torch.no_grad():
        feature = model(tensor)

    feature = feature.squeeze(0).numpy()

    feature = feature / (
        np.linalg.norm(feature) + 1e-8
    )

    return feature


def extract_patches(model, image_path):
    tensor = load_image(image_path)

    with torch.no_grad():
        feature_map = model(tensor)

    feature_map = feature_map.squeeze(0)

    patches = feature_map.permute(
        1, 2, 0
    ).reshape(-1, 512)

    patches = patches / (
        torch.norm(
            patches,
            dim=1,
            keepdim=True
        ) + 1e-8
    )

    return patches.numpy()


def get_test_images():
    groups = {
        "good": 0,
        "broken_large": 1,
        "broken_small": 1,
        "contamination": 1,
    }

    images = []

    for group, label in groups.items():
        folder = DATASET_ROOT / "test" / group

        for path in sorted(folder.glob("*.png")):
            images.append((path, label))

    return images


def benchmark_global(model, images):
    scores = []
    labels = []
    times = []

    for image_path, label in images:

        start = time.perf_counter()

        feature = extract_global(
            model,
            image_path
        )

        elapsed = time.perf_counter() - start

        scores.append(feature)
        labels.append(label)
        times.append(elapsed)

    features = np.array(scores)

    # Build reference from normal test features
    # ONLY for measuring representation quality.
    # The production reference remains the training set.
    normal_features = features[
        np.array(labels) == 0
    ]

    reference = normal_features.mean(axis=0)

    reference = reference / (
        np.linalg.norm(reference) + 1e-8
    )

    anomaly_scores = np.linalg.norm(
        features - reference,
        axis=1
    )

    auroc = roc_auc_score(
        labels,
        anomaly_scores
    )

    return auroc, times


def benchmark_patch(model, images, patch_bank):
    nn = NearestNeighbors(
        n_neighbors=1,
        metric="euclidean"
    )

    nn.fit(patch_bank)

    image_scores = []
    image_labels = []
    pixel_scores = []
    pixel_labels = []
    times = []

    for image_path, label in images:

        start = time.perf_counter()

        patches = extract_patches(
            model,
            image_path
        )

        distances, _ = nn.kneighbors(
            patches
        )

        anomaly_map = distances.flatten()

        elapsed = time.perf_counter() - start

        times.append(elapsed)

        # Maximum patch anomaly = image score
        image_scores.append(
            anomaly_map.max()
        )

        image_labels.append(label)

        # Pixel-level evaluation only for defect images
        if label == 1:

            mask_path = (
                DATASET_ROOT
                / "ground_truth"
                / image_path.parent.name
                / f"{image_path.stem}_mask.png"
            )

            if mask_path.exists():

                anomaly_map_2d = (
                    anomaly_map.reshape(7, 7)
                )

                anomaly_map_2d = np.array(
                    Image.fromarray(
                        anomaly_map_2d.astype(
                            np.float32
                        )
                    ).resize(
                        (900, 900),
                        Image.Resampling.BILINEAR
                    )
                )

                mask = np.array(
                    Image.open(mask_path)
                    .convert("L")
                )

                mask = (
                    mask > 0
                ).astype(np.uint8)

                pixel_scores.extend(
                    anomaly_map_2d.flatten()
                )

                pixel_labels.extend(
                    mask.flatten()
                )

    image_auroc = roc_auc_score(
        image_labels,
        image_scores
    )

    pixel_auroc = roc_auc_score(
        pixel_labels,
        pixel_scores
    )

    return (
        image_auroc,
        pixel_auroc,
        times,
    )


def print_timing(times):
    times_ms = np.array(times) * 1000

    print(
        f"Mean:   {times_ms.mean():.2f} ms"
    )

    print(
        f"Median: {np.median(times_ms):.2f} ms"
    )

    print(
        f"Min:    {times_ms.min():.2f} ms"
    )

    print(
        f"Max:    {times_ms.max():.2f} ms"
    )


def main():

    print("=" * 60)
    print("VISIONFORGE CPU BENCHMARK")
    print("=" * 60)

    images = get_test_images()

    print(
        f"\nTest images: {len(images)}"
    )

    print(
        "Normal images:",
        sum(label == 0 for _, label in images)
    )

    print(
        "Defect images:",
        sum(label == 1 for _, label in images)
    )

    # ---------------------------------------------------------
    # Global ResNet18
    # ---------------------------------------------------------

    print("\nLoading ResNet18 global model...")

    global_model = load_global_model()

    print("\nGLOBAL RESNET18")
    print("-" * 40)

    global_auroc, global_times = benchmark_global(
        global_model,
        images
    )

    print(
        f"Image AUROC: {global_auroc:.4f}"
    )

    print("CPU latency:")
    print_timing(global_times)

    # ---------------------------------------------------------
    # Patch model
    # ---------------------------------------------------------

    print("\nLoading ResNet18 spatial model...")

    spatial_model = load_spatial_model()

    patch_bank = np.load(
        PATCH_BANK_PATH
    )

    print(
        f"Patch bank: {patch_bank.shape}"
    )

    print("\nPATCH ANOMALY DETECTOR")
    print("-" * 40)

    (
        patch_image_auroc,
        pixel_auroc,
        patch_times,
    ) = benchmark_patch(
        spatial_model,
        images,
        patch_bank
    )

    print(
        f"Image AUROC: {patch_image_auroc:.4f}"
    )

    print(
        f"Pixel AUROC: {pixel_auroc:.4f}"
    )

    print("CPU latency:")
    print_timing(patch_times)

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(
        f"{'Method':<25}"
        f"{'Image AUROC':>15}"
        f"{'Pixel AUROC':>15}"
    )

    print("-" * 55)

    print(
        f"{'Histogram':<25}"
        f"{0.6159:>15.4f}"
        f"{'N/A':>15}"
    )

    print(
        f"{'ResNet18 Global':<25}"
        f"{global_auroc:>15.4f}"
        f"{'N/A':>15}"
    )

    print(
        f"{'ResNet18 Patch':<25}"
        f"{patch_image_auroc:>15.4f}"
        f"{pixel_auroc:>15.4f}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()