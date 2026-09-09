from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
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

    # Remove classifier.
    model.classifier = torch.nn.Identity()

    model.eval()

    return model


def extract_feature(model, image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        feature = model(tensor)

    feature = feature.squeeze(0).numpy()

    feature = feature / (
        np.linalg.norm(feature) + 1e-8
    )

    return feature


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
    # Normal reference
    # ---------------------------------------------------------

    train_features = []

    start = time.perf_counter()

    for image_path in sorted(
        train_dir.glob("*.png")
    ):
        feature = extract_feature(
            model,
            image_path
        )

        train_features.append(feature)

    reference = np.mean(
        train_features,
        axis=0
    )

    reference = reference / (
        np.linalg.norm(reference) + 1e-8
    )

    train_time = (
        time.perf_counter() - start
    )

    print(
        "Training images:",
        len(train_features)
    )

    print(
        "Embedding dimension:",
        reference.shape[0]
    )

    # ---------------------------------------------------------
    # Test
    # ---------------------------------------------------------

    test_groups = {
        "good": 0,
        "broken_large": 1,
        "broken_small": 1,
        "contamination": 1,
    }

    scores = []
    labels = []
    inference_times = []

    for group, label in test_groups.items():

        folder = (
            DATASET_ROOT
            / "test"
            / group
        )

        for image_path in sorted(
            folder.glob("*.png")
        ):

            start = time.perf_counter()

            feature = extract_feature(
                model,
                image_path
            )

            elapsed = (
                time.perf_counter()
                - start
            )

            score = np.linalg.norm(
                feature - reference
            )

            scores.append(score)
            labels.append(label)
            inference_times.append(
                elapsed
            )

    scores = np.array(scores)
    labels = np.array(labels)

    auroc = roc_auc_score(
        labels,
        scores
    )

    times_ms = (
        np.array(inference_times)
        * 1000
    )

    print("\nEvaluation")
    print("----------")

    print(
        "Test images:",
        len(scores)
    )

    print(
        f"Image AUROC: {auroc:.4f}"
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

    print(
        "\nReference-building time:",
        f"{train_time:.2f} seconds"
    )


if __name__ == "__main__":
    main()