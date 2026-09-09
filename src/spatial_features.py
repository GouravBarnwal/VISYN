from pathlib import Path

import torch
import numpy as np
from PIL import Image
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
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    # Keep only the convolutional part.
    model = torch.nn.Sequential(*list(model.children())[:-2])

    model.eval()

    return model


def extract_feature_map(model, image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        feature_map = model(image)

    return feature_map.squeeze(0).numpy()


def main():
    model = load_model()

    image_path = (
        DATASET_ROOT
        / "test"
        / "broken_large"
        / "000.png"
    )

    feature_map = extract_feature_map(model, image_path)

    print("Image:", image_path)
    print("Feature map shape:", feature_map.shape)

    channels, height, width = feature_map.shape

    print("Channels:", channels)
    print("Spatial height:", height)
    print("Spatial width:", width)
    print("Spatial locations:", height * width)


if __name__ == "__main__":
    main()