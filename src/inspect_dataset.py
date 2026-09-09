from pathlib import Path
from PIL import Image


DATASET_ROOT = Path("data/bottle")


def inspect_folder(folder: Path):
    images = list(folder.glob("*.png"))

    print(f"\n{folder}")
    print(f"Images: {len(images)}")

    if not images:
        return

    sizes = {}
    corrupted = []

    for image_path in images:
        try:
            with Image.open(image_path) as image:
                size = image.size
                sizes[size] = sizes.get(size, 0) + 1
        except Exception:
            corrupted.append(image_path.name)

    print("Image sizes:")
    for size, count in sizes.items():
        print(f"  {size}: {count}")

    if corrupted:
        print("Corrupted images:", corrupted)
    else:
        print("Corrupted images: 0")


def main():
    inspect_folder(DATASET_ROOT / "train" / "good")

    for folder in (DATASET_ROOT / "test").iterdir():
        if folder.is_dir():
            inspect_folder(folder)


if __name__ == "__main__":
    main()