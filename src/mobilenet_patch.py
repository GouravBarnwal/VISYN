from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from torchvision import models, transforms


DATASET_ROOT = Path("data/bottle")


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

    # Keep the convolutional feature extractor.
    model.classifier = torch.nn.Identity()

    # We need the feature extractor before pooling/classification.
    model = models.mobilenet_v3_small(
        weights=weights
    ).features

    model.eval()

    return model


def extract_feature_map(model, image_path):
    image = Image.open(
        image_path
    ).convert("RGB")

    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        feature_map = model(tensor)

    return feature_map.squeeze(0)


def extract_patches(model, image_path):
    feature_map = extract_feature_map(
        model,
        image_path
    )

    channels, height, width = feature_map.shape

    # C × H × W → H×W × C
    patches = feature_map.permute(
        1, 2, 0
    ).reshape(
        height * width,
        channels
    )

    # Normalize every patch
    patches = patches / (
        torch.norm(
            patches,
            dim=1,
            keepdim=True
        ) + 1e-8
    )

    return patches.numpy(), height, width


def main():

    model = load_model()

    print("Model: MobileNetV3-Small")
    print("Device: CPU")

    train_dir = (
        DATASET_ROOT
        / "train"
        / "good"
    )

    # ---------------------------------------------------------
    # Build normal patch bank
    # ---------------------------------------------------------

    patch_bank = []

    start = time.perf_counter()

    for image_path in sorted(
        train_dir.glob("*.png")
    ):

        patches, height, width = extract_patches(
            model,
            image_path
        )

        patch_bank.append(patches)

    patch_bank = np.concatenate(
        patch_bank,
        axis=0
    )

    build_time = (
        time.perf_counter() - start
    )

    print("\nNormal patch bank")
    print("-----------------")
    print(
        "Training images:",
        len(list(train_dir.glob("*.png")))
    )

    print(
        "Patches per image:",
        height * width
    )

    print(
        "Patch dimension:",
        patch_bank.shape[1]
    )

    print(
        "Total patches:",
        len(patch_bank)
    )

    print(
        "Patch bank shape:",
        patch_bank.shape
    )

    print(
        f"Reference-building time: "
        f"{build_time:.2f} seconds"
    )

    # ---------------------------------------------------------
    # Nearest-neighbor index
    # ---------------------------------------------------------

    print("\nBuilding nearest-neighbor index...")

    nn = NearestNeighbors(
        n_neighbors=1,
        metric="euclidean",
        algorithm="auto"
    )

    nn.fit(patch_bank)

    print("Index ready.")

    # ---------------------------------------------------------
    # Test
    # ---------------------------------------------------------

    test_groups = {
        "good": 0,
        "broken_large": 1,
        "broken_small": 1,
        "contamination": 1,
    }

    image_scores = []
    image_labels = []

    pixel_scores = []
    pixel_labels = []

    inference_times = []

    for group, label in test_groups.items():

        image_dir = (
            DATASET_ROOT
            / "test"
            / group
        )

        mask_dir = (
            DATASET_ROOT
            / "ground_truth"
            / group
        )

        print(
            f"\nEvaluating: {group}"
        )

        for image_path in sorted(
            image_dir.glob("*.png")
        ):

            start = time.perf_counter()

            patches, h, w = extract_patches(
                model,
                image_path
            )

            distances, _ = nn.kneighbors(
                patches
            )

            elapsed = (
                time.perf_counter()
                - start
            )

            inference_times.append(
                elapsed
            )

            anomaly_map = (
                distances.flatten()
            )

            # Maximum patch anomaly
            image_score = anomaly_map.max()

            image_scores.append(
                image_score
            )

            image_labels.append(
                label
            )

            # -------------------------------------------------
            # Pixel localization
            # -------------------------------------------------

            if label == 1:

                mask_path = (
                    mask_dir
                    / f"{image_path.stem}_mask.png"
                )

                if mask_path.exists():

                    anomaly_map = (
                        anomaly_map.reshape(
                            h,
                            w
                        )
                    )

                    anomaly_map = np.array(
                        Image.fromarray(
                            anomaly_map.astype(
                                np.float32
                            )
                        ).resize(
                            (900, 900),
                            Image.Resampling.BILINEAR
                        )
                    )

                    mask = np.array(
                        Image.open(
                            mask_path
                        ).convert("L")
                    )

                    mask = (
                        mask > 0
                    ).astype(np.uint8)

                    pixel_scores.extend(
                        anomaly_map.flatten()
                    )

                    pixel_labels.extend(
                        mask.flatten()
                    )

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    image_auroc = roc_auc_score(
        image_labels,
        image_scores
    )

    pixel_auroc = roc_auc_score(
        pixel_labels,
        pixel_scores
    )

    times_ms = (
        np.array(inference_times)
        * 1000
    )

    print("\n" + "=" * 55)
    print("MOBILENETV3 PATCH RESULTS")
    print("=" * 55)

    print(
        f"Image AUROC: {image_auroc:.4f}"
    )

    print(
        f"Pixel AUROC: {pixel_auroc:.4f}"
    )

    print("\nCPU latency")

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


if __name__ == "__main__":
    main()