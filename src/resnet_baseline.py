from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from torchvision import models, transforms


DATASET_ROOT = Path("data/bottle")


# Image preprocessing expected by the pretrained model
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_model():
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    # Remove the final classification layer.
    # The remaining network produces a 512-dimensional feature vector.
    model.fc = torch.nn.Identity()

    model.eval()
    return model


def extract_feature(model, image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        feature = model(image)

    feature = feature.squeeze(0).numpy()

    # Normalize the embedding
    feature = feature / (np.linalg.norm(feature) + 1e-8)

    return feature


def main():
    model = load_model()

    print("Model: ResNet18")
    print("Device: CPU")
    print("Embedding dimension: 512")

    # ---------------------------------------------------------
    # Build reference representation from NORMAL training images
    # ---------------------------------------------------------

    train_dir = DATASET_ROOT / "train" / "good"

    train_features = []

    for image_path in sorted(train_dir.glob("*.png")):
        feature = extract_feature(model, image_path)
        train_features.append(feature)

    train_features = np.array(train_features)

    reference = train_features.mean(axis=0)

    # Normalize reference
    reference = reference / (np.linalg.norm(reference) + 1e-8)

    print("Training images:", len(train_features))

    # ---------------------------------------------------------
    # Evaluate on test images
    # ---------------------------------------------------------

    test_groups = {
        "good": 0,
        "broken_large": 1,
        "broken_small": 1,
        "contamination": 1,
    }

    scores = []
    labels = []

    for group_name, label in test_groups.items():

        folder = DATASET_ROOT / "test" / group_name

        for image_path in sorted(folder.glob("*.png")):

            feature = extract_feature(model, image_path)

            # Euclidean distance from normal reference
            anomaly_score = np.linalg.norm(feature - reference)

            scores.append(anomaly_score)
            labels.append(label)

    scores = np.array(scores)
    labels = np.array(labels)

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------

    auroc = roc_auc_score(labels, scores)

    print("\nEvaluation")
    print("----------")
    print("Test images:", len(scores))
    print(f"AUROC: {auroc:.4f}")

    # Temporary threshold for comparison only
    threshold = np.median(scores[labels == 0])

    predictions = (scores >= threshold).astype(int)

    print(f"\nThreshold: {threshold:.4f}")

    print("\nConfusion Matrix")
    print(confusion_matrix(labels, predictions))

    print("\nClassification Report")
    print(
        classification_report(
            labels,
            predictions,
            target_names=["normal", "defect"],
        )
    )


if __name__ == "__main__":
    main()