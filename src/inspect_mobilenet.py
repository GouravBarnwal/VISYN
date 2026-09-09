import torch
from torchvision import models


def main():

    weights = models.MobileNet_V3_Small_Weights.DEFAULT

    model = models.mobilenet_v3_small(
        weights=weights
    )

    print("MobileNetV3-Small feature layers")
    print("=" * 50)

    for index, layer in enumerate(model.features):
        print(
            f"{index:2d}: "
            f"{layer.__class__.__name__}"
        )


if __name__ == "__main__":
    main()