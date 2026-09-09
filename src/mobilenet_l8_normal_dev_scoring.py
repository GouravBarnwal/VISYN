from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from torchvision import models, transforms


# ============================================================
# CONFIG
# ============================================================

DATA_ROOT = Path("data")

NORMAL_SPLIT_PATH = Path(
    "artifacts/splits/normal_splits.json"
)

GLOBAL_BANK_ROOT = Path(
    "artifacts/patch_banks/mobilenet_l8"
)

OUTPUT_PATH = Path(
    "artifacts/evaluation/mobilenet_l8_normal_dev_scores.json"
)

CATEGORIES = [
    "bottle",
    "hazelnut",
    "cable",
    "capsule",
    "screw",
    "metal_nut",
]


# ============================================================
# MODEL
# ============================================================

class MobileNetL8(nn.Module):

    def __init__(self):
        super().__init__()

        backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT
        )

        # Layer-8 feature representation
        # Expected output: 48 x 14 x 14
        self.features = backbone.features[:9]

    def forward(self, x):
        return self.features(x)


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# FEATURE EXTRACTION
# ============================================================

@torch.no_grad()
def extract_feature_map(
    model,
    image_path,
):

    image = Image.open(
        image_path
    ).convert("RGB")

    x = transform(
        image
    ).unsqueeze(0)

    feature_map = model(x)

    feature_map = (
        feature_map
        .squeeze(0)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )

    # L2-normalize every spatial descriptor.
    #
    # This is identical to the validated MobileNet L8
    # defect-development scoring pipeline.
    norms = np.linalg.norm(
        feature_map,
        axis=2,
        keepdims=True,
    )

    feature_map = (
        feature_map
        /
        np.maximum(
            norms,
            1e-12,
        )
    )

    return feature_map


# ============================================================
# IMAGE ANOMALY SCORE
# ============================================================

def calculate_image_score(
    feature_map,
    reference_bank,
):

    h, w, d = feature_map.shape

    query = feature_map.reshape(
        -1,
        d,
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # The validated production/defect scorer uses Euclidean
    # nearest-neighbor distance:
    #
    #     torch.cdist(query, bank, p=2)
    #
    # Do NOT replace this with:
    #
    #     1 - cosine_similarity
    #
    # even though the descriptors are L2-normalized.
    #
    # Calibration and production scoring MUST use the same
    # distance metric.
    # --------------------------------------------------------

    query_tensor = torch.from_numpy(
        query.astype(np.float32)
    )

    bank_tensor = torch.from_numpy(
        reference_bank.astype(np.float32)
    )

    distances = torch.cdist(
        query_tensor,
        bank_tensor,
        p=2,
    )

    nearest_distances = (
        distances
        .min(dim=1)
        .values
    )

    values = (
        nearest_distances
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # Production aggregation:
    #
    # TOP-5 mean of the five highest patch anomaly scores.
    #
    # This is exactly the aggregation used by the validated
    # MobileNet L8 defect-development scorer.
    # --------------------------------------------------------

    values = np.sort(values)

    top_k = min(
        5,
        len(values),
    )

    image_score = float(
        np.mean(
            values[-top_k:]
        )
    )

    return image_score


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print(
        "VISIONFORGE — MOBILE L8 "
        "NORMAL DEVELOPMENT SCORING"
    )
    print("=" * 72)

    # --------------------------------------------------------
    # Load canonical normal split
    # --------------------------------------------------------

    with open(
        NORMAL_SPLIT_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        split_data = json.load(f)

    print(
        f"Loaded split: "
        f"{NORMAL_SPLIT_PATH}"
    )

    print(
        f"Seed: "
        f"{split_data.get('seed')}"
    )

    print(
        f"Reference ratio: "
        f"{split_data.get('reference_ratio')}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = MobileNetL8()
    model.eval()

    print(
        "MobileNetV3-Small L8 model loaded."
    )

    # --------------------------------------------------------
    # Output structure
    # --------------------------------------------------------

    output = {
        "method": (
            "MobileNetV3-Small-L8"
        ),
        "distance_metric": (
            "euclidean_cdist_p2"
        ),
        "aggregation": (
            "TOP5"
        ),
        "split": (
            "canonical_normal_development"
        ),
        "seed": (
            split_data.get("seed")
        ),
        "reference_ratio": (
            split_data.get(
                "reference_ratio"
            )
        ),
        "categories": {},
    }

    # ========================================================
    # CATEGORY LOOP
    # ========================================================

    for category in CATEGORIES:

        print()
        print("-" * 72)
        print(
            f"CATEGORY: {category}"
        )
        print("-" * 72)

        # ----------------------------------------------------
        # Load normal reference bank
        # ----------------------------------------------------

        bank_path = (
            GLOBAL_BANK_ROOT
            / f"{category}.npy"
        )

        if not bank_path.exists():

            raise FileNotFoundError(
                f"Missing reference bank: "
                f"{bank_path}"
            )

        reference_bank = np.load(
            bank_path
        )

        print(
            f"Reference bank: "
            f"{reference_bank.shape}"
        )

        # ----------------------------------------------------
        # Verify reference bank dimensions
        # ----------------------------------------------------

        if reference_bank.ndim != 2:

            raise RuntimeError(
                f"Unexpected reference bank "
                f"dimensions for {category}: "
                f"{reference_bank.shape}"
            )

        if reference_bank.shape[1] != 48:

            raise RuntimeError(
                f"Unexpected descriptor dimension "
                f"for {category}: "
                f"{reference_bank.shape}"
            )

        # ----------------------------------------------------
        # Development paths
        # ----------------------------------------------------

        development_paths = (
            split_data[
                "categories"
            ][category][
                "development"
            ]
        )

        if not isinstance(
            development_paths,
            list,
        ):

            raise TypeError(
                f"Expected development "
                f"paths to be a list for "
                f"{category}, got "
                f"{type(development_paths)}"
            )

        print(
            f"Normal development images: "
            f"{len(development_paths)}"
        )

        # ----------------------------------------------------
        # Score images
        # ----------------------------------------------------

        category_scores = []

        for index, relative_path in enumerate(
            development_paths,
            start=1,
        ):

            image_path = (
                DATA_ROOT
                / relative_path
            )

            if not image_path.exists():

                raise FileNotFoundError(
                    f"Missing image: "
                    f"{image_path}"
                )

            feature_map = (
                extract_feature_map(
                    model,
                    image_path,
                )
            )

            # ------------------------------------------------
            # Verify expected representation
            # ------------------------------------------------

            if feature_map.shape != (
                14,
                14,
                48,
            ):

                raise RuntimeError(
                    f"Unexpected feature "
                    f"shape for "
                    f"{relative_path}: "
                    f"{feature_map.shape}"
                )

            score = (
                calculate_image_score(
                    feature_map,
                    reference_bank,
                )
            )

            category_scores.append(
                score
            )

            if (
                index % 10 == 0
                or
                index == len(
                    development_paths
                )
            ):

                print(
                    f"Processed "
                    f"{index}/"
                    f"{len(development_paths)}",
                    end="\r",
                )

        print()

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        values = np.asarray(
            category_scores,
            dtype=np.float64,
        )

        stats = {
            "count": int(
                len(values)
            ),
            "min": float(
                np.min(values)
            ),
            "max": float(
                np.max(values)
            ),
            "mean": float(
                np.mean(values)
            ),
            "std": float(
                np.std(values)
            ),
            "median": float(
                np.median(values)
            ),
            "p90": float(
                np.percentile(
                    values,
                    90,
                )
            ),
            "p95": float(
                np.percentile(
                    values,
                    95,
                )
            ),
            "p97_5": float(
                np.percentile(
                    values,
                    97.5,
                )
            ),
            "p99": float(
                np.percentile(
                    values,
                    99,
                )
            ),
            "p99_5": float(
                np.percentile(
                    values,
                    99.5,
                )
            ),
        }

        # ----------------------------------------------------
        # Save path + score together
        # ----------------------------------------------------

        output[
            "categories"
        ][category] = {

            "samples": [
                {
                    "path": str(path),
                    "score": float(score),
                }
                for path, score in zip(
                    development_paths,
                    category_scores,
                )
            ],

            "scores": [
                float(x)
                for x in category_scores
            ],

            "statistics": stats,
        }

        print(
            f"Count:  {stats['count']}"
        )

        print(
            f"Min:    {stats['min']:.6f}"
        )

        print(
            f"Max:    {stats['max']:.6f}"
        )

        print(
            f"Mean:   {stats['mean']:.6f}"
        )

        print(
            f"Std:    {stats['std']:.6f}"
        )

        print(
            f"Median: {stats['median']:.6f}"
        )

        print(
            f"P95:    {stats['p95']:.6f}"
        )

        print(
            f"P97.5:  {stats['p97_5']:.6f}"
        )

        print(
            f"P99:    {stats['p99']:.6f}"
        )

        print(
            f"P99.5:  {stats['p99_5']:.6f}"
        )

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    print()
    print("=" * 72)
    print(
        "NORMAL DEVELOPMENT SCORING COMPLETE"
    )
    print(
        f"Saved to: {OUTPUT_PATH}"
    )
    print(
        "Distance metric: Euclidean (torch.cdist, p=2)"
    )
    print(
        "Aggregation: TOP5"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()