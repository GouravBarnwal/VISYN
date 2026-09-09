from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms


DATASET_ROOT = Path("data/bottle")
IMAGE_PATH = (
    DATASET_ROOT
    / "test"
    / "broken_large"
    / "000.png"
)

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

    model = torch.nn.Sequential(
        *list(model.children())[:-2]
    )

    model.eval()

    return model


def extract_patches(model, image_path):
    image = Image.open(image_path).convert("RGB")

    tensor = transform(image).unsqueeze(0)

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


def main():

    model = load_model()

    patch_bank = np.load(PATCH_BANK_PATH)

    print("Patch bank:", patch_bank.shape)

    # ---------------------------------------------------------
    # Extract test-image patches
    # ---------------------------------------------------------

    patches = extract_patches(
        model,
        IMAGE_PATH
    )

    # ---------------------------------------------------------
    # Calculate nearest normal patch distance
    # ---------------------------------------------------------

    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(
        n_neighbors=1,
        metric="euclidean"
    )

    nn.fit(patch_bank)

    distances, _ = nn.kneighbors(patches)

    anomaly_map = distances.flatten().reshape(7, 7)

    # ---------------------------------------------------------
    # Resize anomaly map to image size
    # ---------------------------------------------------------

    anomaly_map = anomaly_map.astype(np.float32)

    anomaly_map = (
        anomaly_map - anomaly_map.min()
    ) / (
        anomaly_map.max()
        - anomaly_map.min()
        + 1e-8
    )

    anomaly_map = np.array(
        Image.fromarray(anomaly_map)
        .resize((900, 900), Image.Resampling.BILINEAR)
    )

    # ---------------------------------------------------------
    # Load original image
    # ---------------------------------------------------------

    image = np.array(
        Image.open(IMAGE_PATH).convert("RGB")
    )

    # ---------------------------------------------------------
    # Display
    # ---------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5)
    )

    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(image)
    axes[1].imshow(
        anomaly_map,
        alpha=0.5
    )
    axes[1].set_title("Anomaly Heatmap")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()