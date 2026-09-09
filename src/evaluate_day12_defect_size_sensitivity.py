import json
from pathlib import Path

import numpy as np
from PIL import Image


# ============================================================
# Configuration
# ============================================================

FUSION_PATH = Path(
    "artifacts/evaluation/day11_l4_l8_fusion.json"
)

OUTPUT_PATH = Path(
    "artifacts/evaluation/day12_defect_size_sensitivity.json"
)

CATEGORIES = [
    "bottle",
    "hazelnut",
    "cable",
    "capsule",
    "screw",
    "metal_nut",
]

AREA_BINS = [
    ("tiny", 0.0, 0.10),
    ("small", 0.10, 0.50),
    ("medium", 0.50, 2.00),
    ("large", 2.00, 100.01),
]


# ============================================================
# JSON
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# Path handling
# ============================================================

def resolve_image_path(path_string):
    """
    Day 11 stores paths such as:

        data/bottle/test/broken_large/019.png

    Therefore paths beginning with 'data' must be used
    directly rather than having 'data/' prepended again.
    """

    path = Path(path_string)

    if path.is_absolute():
        return path

    if path.parts and path.parts[0] == "data":
        return path

    return Path("data") / path


def find_mask(image_path):
    """
    Convert:

        data/bottle/test/broken_large/019.png

    into:

        data/bottle/ground_truth/broken_large/019_mask.png
    """

    image_path = Path(image_path)

    parts = image_path.parts

    if len(parts) < 5:
        return None

    category = parts[1]
    defect_type = parts[-2]
    stem = image_path.stem

    mask_path = (
        Path("data")
        / category
        / "ground_truth"
        / defect_type
        / f"{stem}_mask.png"
    )

    if mask_path.exists():
        return mask_path

    return None


# ============================================================
# Mask area
# ============================================================

def get_mask_area_percent(path_string):

    image_path = resolve_image_path(path_string)

    mask_path = find_mask(image_path)

    if mask_path is None:
        return None

    mask = np.asarray(
        Image.open(mask_path)
    )

    if mask.ndim == 3:
        mask = mask[..., 0]

    positive_pixels = np.count_nonzero(
        mask > 0
    )

    total_pixels = (
        mask.shape[0]
        * mask.shape[1]
    )

    if total_pixels == 0:
        return None

    return (
        positive_pixels
        / total_pixels
        * 100.0
    )


# ============================================================
# Area bins
# ============================================================

def assign_area_bin(area_percent):

    for name, lower, upper in AREA_BINS:

        if (
            area_percent >= lower
            and area_percent < upper
        ):
            return name

    return None


# ============================================================
# Extract defect scores
# ============================================================

def extract_defect_records(data):

    records = []

    for category in CATEGORIES:

        category_data = data[
            "categories"
        ][category]

        sample_scores = category_data[
            "sample_scores"
        ]

        defect = sample_scores[
            "defect"
        ]

        paths = defect["paths"]
        l4_scores = defect["L4"]
        l8_scores = defect["L8"]

        l4_normalized = defect[
            "L4_normalized"
        ]

        l8_normalized = defect[
            "L8_normalized"
        ]

        for (
            path,
            l4,
            l8,
            l4_norm,
            l8_norm,
        ) in zip(
            paths,
            l4_scores,
            l8_scores,
            l4_normalized,
            l8_normalized,
        ):

            path_obj = Path(path)

            defect_type = path_obj.parent.name

            fusion_score = (
                0.5 * float(l4_norm)
                + 0.5 * float(l8_norm)
            )

            records.append(
                {
                    "category": category,
                    "path": path,
                    "defect_type": defect_type,
                    "L4": float(l4),
                    "L8": float(l8),
                    "L4_normalized": float(
                        l4_norm
                    ),
                    "L8_normalized": float(
                        l8_norm
                    ),
                    "fusion_50_50": float(
                        fusion_score
                    ),
                }
            )

    return records


# ============================================================
# Spearman
# ============================================================

def rank_array(values):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    order = np.argsort(values)

    ranks = np.empty(
        len(values),
        dtype=np.float64,
    )

    ranks[order] = np.arange(
        len(values),
        dtype=np.float64,
    )

    return ranks


def spearman_correlation(x, y):

    if len(x) < 2:
        return float("nan")

    x_ranks = rank_array(x)
    y_ranks = rank_array(y)

    x_std = np.std(x_ranks)
    y_std = np.std(y_ranks)

    if x_std == 0 or y_std == 0:
        return float("nan")

    return float(
        np.corrcoef(
            x_ranks,
            y_ranks,
        )[0, 1]
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 78)
    print(
        "DAY 12 — DEFECT-SIZE SENSITIVITY"
    )
    print("=" * 78)

    data = load_json(FUSION_PATH)

    records = extract_defect_records(data)

    valid_records = []

    missing_masks = 0

    # --------------------------------------------------------
    # Validate paths and calculate mask areas
    # --------------------------------------------------------

    for record in records:

        area_percent = get_mask_area_percent(
            record["path"]
        )

        if area_percent is None:

            missing_masks += 1

            print(
                "WARNING: mask not found:",
                record["path"],
            )

            continue

        record["area_percent"] = float(
            area_percent
        )

        record["area_bin"] = assign_area_bin(
            area_percent
        )

        valid_records.append(record)

    print(
        f"\nTotal defect samples: "
        f"{len(records)}"
    )

    print(
        f"Valid masks: "
        f"{len(valid_records)}"
    )

    print(
        f"Missing masks: "
        f"{missing_masks}"
    )

    if not valid_records:

        raise RuntimeError(
            "No valid defect masks were found. "
            "Check the MVTec ground_truth paths before "
            "continuing the experiment."
        )

    # --------------------------------------------------------
    # Result structure
    # --------------------------------------------------------

    results = {
        "source": str(FUSION_PATH),
        "split": "development",
        "final_test_used": False,
        "analysis_scope": (
            "Relationship between defect mask area "
            "and anomaly score among defect samples."
        ),
        "area_bins": [
            {
                "name": name,
                "lower_percent": lower,
                "upper_percent": upper,
            }
            for name, lower, upper in AREA_BINS
        ],
        "categories": {},
    }

    # --------------------------------------------------------
    # Category analysis
    # --------------------------------------------------------

    for category in CATEGORIES:

        category_records = [
            record
            for record in valid_records
            if record["category"] == category
        ]

        print("\n" + "=" * 78)
        print(
            f"CATEGORY: {category}"
        )
        print("=" * 78)

        category_result = {
            "count": len(category_records),
            "area_percent": {},
            "bins": {},
        }

        if category_records:

            areas = np.asarray(
                [
                    record["area_percent"]
                    for record in category_records
                ],
                dtype=np.float64,
            )

            category_result[
                "area_percent"
            ] = {
                "mean": float(areas.mean()),
                "median": float(np.median(areas)),
                "min": float(areas.min()),
                "max": float(areas.max()),
            }

            print(
                f"Defects: {len(category_records)}"
            )

            print(
                f"Mask area %: "
                f"median={np.median(areas):.4f} "
                f"min={areas.min():.4f} "
                f"max={areas.max():.4f}"
            )

        # ----------------------------------------------------
        # Area bins
        # ----------------------------------------------------

        for (
            bin_name,
            _,
            _,
        ) in AREA_BINS:

            bin_records = [
                record
                for record in category_records
                if record["area_bin"] == bin_name
            ]

            bin_result = {
                "count": len(bin_records)
            }

            if bin_records:

                bin_areas = np.asarray(
                    [
                        record["area_percent"]
                        for record in bin_records
                    ],
                    dtype=np.float64,
                )

                bin_result[
                    "area_percent_mean"
                ] = float(
                    bin_areas.mean()
                )

                bin_result[
                    "area_percent_median"
                ] = float(
                    np.median(bin_areas)
                )

                for method in [
                    "L8",
                    "fusion_50_50",
                ]:

                    values = np.asarray(
                        [
                            record[method]
                            for record in bin_records
                        ],
                        dtype=np.float64,
                    )

                    bin_result[
                        f"{method}_mean"
                    ] = float(values.mean())

                    bin_result[
                        f"{method}_median"
                    ] = float(np.median(values))

                    bin_result[
                        f"{method}_min"
                    ] = float(values.min())

                    bin_result[
                        f"{method}_max"
                    ] = float(values.max())

                print(
                    f"  {bin_name:8s} "
                    f"n={len(bin_records):2d} "
                    f"area_median="
                    f"{np.median(bin_areas):.4f} "
                    f"L8="
                    f"{bin_result['L8_mean']:.4f} "
                    f"fusion="
                    f"{bin_result['fusion_50_50_mean']:.4f}"
                )

            else:

                print(
                    f"  {bin_name:8s} "
                    f"n=0"
                )

            category_result[
                "bins"
            ][bin_name] = bin_result

        results[
            "categories"
        ][category] = category_result

    # --------------------------------------------------------
    # Global relationship
    # --------------------------------------------------------

    print("\n" + "=" * 78)
    print(
        "GLOBAL AREA ↔ SCORE RELATIONSHIP"
    )
    print("=" * 78)

    areas = np.asarray(
        [
            record["area_percent"]
            for record in valid_records
        ],
        dtype=np.float64,
    )

    results["global"] = {
        "count": len(valid_records),
        "area_percent": {
            "mean": float(areas.mean()),
            "median": float(np.median(areas)),
            "min": float(areas.min()),
            "max": float(areas.max()),
        },
    }

    for method in [
        "L8",
        "fusion_50_50",
    ]:

        scores = np.asarray(
            [
                record[method]
                for record in valid_records
            ],
            dtype=np.float64,
        )

        correlation = spearman_correlation(
            areas,
            scores,
        )

        results[
            "global"
        ][
            f"{method}_area_spearman"
        ] = correlation

        print(
            f"{method:15s} "
            f"Spearman={correlation:+.4f}"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

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
            results,
            f,
            indent=2,
        )

    print("\nResults saved to:")
    print(OUTPUT_PATH)

    print(
        "\nDAY 12 EXPERIMENT 6 COMPLETE"
    )


if __name__ == "__main__":
    main()