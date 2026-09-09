from pathlib import Path

import numpy as np
import torch
from PIL import Image
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


def load_model():
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    # Keep convolutional feature extractor.
    model = torch.nn.Sequential(*list(model.children())[:-2])

    model.eval()

    return model


def extract_patches(model, image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        feature_map = model(image)

    # (1, 512, 7, 7) → (49, 512)
    feature_map = feature_map.squeeze(0)

    patches = feature_map.permute(1, 2, 0).reshape(-1, 512)

    # Normalize each patch
    patches = patches / (
        torch.norm(patches, dim=1, keepdim=True) + 1e-8
    )

    return patches.numpy()


def main():
    # ---------------------------------------------------------
    # Load model
    # ---------------------------------------------------------

    model = load_model()

    print("Model loaded.")

    # ---------------------------------------------------------
    # Load normal patch bank
    # ---------------------------------------------------------

    patch_bank = np.load(PATCH_BANK_PATH)

    print("Patch bank shape:", patch_bank.shape)

    # ---------------------------------------------------------
    # Build nearest-neighbor index
    # ---------------------------------------------------------

    print("Building nearest-neighbor index...")

    nn = NearestNeighbors(
        n_neighbors=1,
        metric="euclidean",
        algorithm="auto",
    )

    nn.fit(patch_bank)

    print("Index ready.")

    # ---------------------------------------------------------
    # Test one defective image
    # ---------------------------------------------------------

    image_path = (
        DATASET_ROOT
        / "test"
        / "broken_large"
        / "000.png"
    )

    patches = extract_patches(model, image_path)

    print("Test image:", image_path)
    print("Test patches:", patches.shape)

    # Find closest normal patch for every test patch
    distances, indices = nn.kneighbors(patches)

    distances = distances.flatten()

    # 49 anomaly scores
    anomaly_map = distances.reshape(7, 7)

    # Image-level score
    image_score = anomaly_map.max()

    print("\nPatch anomaly scores")
    print("--------------------")
    print("Minimum:", f"{anomaly_map.min():.4f}")
    print("Maximum:", f"{anomaly_map.max():.4f}")
    print("Mean:", f"{anomaly_map.mean():.4f}")
    print("Image anomaly score:", f"{image_score:.4f}")

    print("\n7x7 anomaly map")
    print(anomaly_map)


if __name__ == "__main__":
    main()