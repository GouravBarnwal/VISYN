import torch
from PIL import Image
from torchvision import models, transforms


IMAGE_PATH = "data/bottle/test/broken_large/000.png"


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def main():

    weights = models.MobileNet_V3_Small_Weights.DEFAULT

    model = models.mobilenet_v3_small(
        weights=weights
    )

    model.eval()

    image = Image.open(
        IMAGE_PATH
    ).convert("RGB")

    tensor = transform(image).unsqueeze(0)

    print("Feature map shapes")
    print("=" * 50)

    x = tensor

    with torch.no_grad():

        for index, layer in enumerate(
            model.features
        ):

            x = layer(x)

            print(
                f"Layer {index:2d}: "
                f"{tuple(x.shape)}"
            )


if __name__ == "__main__":
    main()