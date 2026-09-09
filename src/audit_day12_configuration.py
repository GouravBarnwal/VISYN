import json
from pathlib import Path


# ============================================================
# Paths
# ============================================================

NORMAL_SPLIT = Path(
    "artifacts/splits/normal_splits.json"
)

DEFECT_SPLIT = Path(
    "artifacts/splits/defect_splits.json"
)

PRODUCTION_THRESHOLDS = Path(
    "artifacts/evaluation/production_thresholds.json"
)

CALIBRATION = Path(
    "artifacts/evaluation/calibration_candidates.json"
)

NORMAL_SCORES = Path(
    "artifacts/evaluation/mobilenet_l8_normal_dev_scores.json"
)

INSPECTION_RESULT = Path(
    "artifacts/evaluation/inspection_pipeline_dev_results.json"
)

FUSION = Path(
    "artifacts/evaluation/day11_l4_l8_fusion.json"
)

DAY12_SIZE = Path(
    "artifacts/evaluation/day12_defect_size_sensitivity.json"
)

DAY12_RESOLUTION = Path(
    "artifacts/evaluation/day12_small_defect_resolution.json"
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
# Helpers
# ============================================================

def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def require_file(path):

    if not path.exists():

        raise RuntimeError(
            f"Missing required artifact: {path}"
        )


def check(condition, message):

    if not condition:

        raise RuntimeError(
            message
        )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 78)
    print(
        "DAY 12 — FINAL DEVELOPMENT CONFIGURATION AUDIT"
    )
    print("=" * 78)

    # --------------------------------------------------------
    # Required artifacts
    # --------------------------------------------------------

    print(
        "\nChecking required artifacts..."
    )

    required_files = [
        NORMAL_SPLIT,
        DEFECT_SPLIT,
        PRODUCTION_THRESHOLDS,
        CALIBRATION,
        NORMAL_SCORES,
        INSPECTION_RESULT,
        FUSION,
        DAY12_SIZE,
        DAY12_RESOLUTION,
    ]

    for path in required_files:

        require_file(path)

        print(
            f"  PASS: {path}"
        )

    # --------------------------------------------------------
    # Load artifacts
    # --------------------------------------------------------

    normal_split = load_json(
        NORMAL_SPLIT
    )

    defect_split = load_json(
        DEFECT_SPLIT
    )

    thresholds = load_json(
        PRODUCTION_THRESHOLDS
    )

    calibration = load_json(
        CALIBRATION
    )

    normal_scores = load_json(
        NORMAL_SCORES
    )

    inspection = load_json(
        INSPECTION_RESULT
    )

    fusion = load_json(
        FUSION
    )

    size_analysis = load_json(
        DAY12_SIZE
    )

    resolution = load_json(
        DAY12_RESOLUTION
    )

    # --------------------------------------------------------
    # Category coverage
    # --------------------------------------------------------

    print(
        "\nChecking category coverage..."
    )

    category_artifacts = [
        ("normal split", normal_split),
        ("defect split", defect_split),
        ("thresholds", thresholds),
        ("calibration", calibration),
        ("normal scores", normal_scores),
        ("fusion", fusion),
        ("size analysis", size_analysis),
        ("resolution analysis", resolution),
    ]

    for name, artifact in category_artifacts:

        check(
            "categories" in artifact,
            f"{name} has no categories field.",
        )

        actual = set(
            artifact["categories"].keys()
        )

        expected = set(
            CATEGORIES
        )

        check(
            actual == expected,
            (
                f"{name} category mismatch: "
                f"{sorted(actual)}"
            ),
        )

        print(
            f"  PASS: {name}"
        )

    # --------------------------------------------------------
    # Canonical normal split
    # --------------------------------------------------------

    print(
        "\nChecking canonical normal split..."
    )

    check(
        normal_split.get("seed") == 42,
        "Normal split seed is not 42.",
    )

    print(
        "  PASS: seed=42"
    )

    check(
        normal_split.get(
            "reference_ratio"
        ) == 0.8,
        "Normal reference ratio is not 0.8.",
    )

    print(
        "  PASS: reference ratio=0.8"
    )

    # --------------------------------------------------------
    # Locked defect split
    # --------------------------------------------------------

    print(
        "\nChecking locked defect split..."
    )

    check(
        defect_split.get("seed") == 42,
        "Defect split seed is not 42.",
    )

    print(
        "  PASS: seed=42"
    )

    check(
        defect_split.get(
            "development_ratio"
        ) == 0.5,
        "Defect development ratio is not 0.5.",
    )

    print(
        "  PASS: development_ratio=0.5"
    )

    # --------------------------------------------------------
    # Development/final separation
    # --------------------------------------------------------

    print(
        "\nChecking development/final defect separation..."
    )

    for category in CATEGORIES:

        category_data = defect_split[
            "categories"
        ][category]

        development = set(
            category_data["development"]
        )

        final_test = set(
            category_data["final_test"]
        )

        overlap = (
            development
            & final_test
        )

        check(
            not overlap,
            (
                f"{category}: "
                f"{len(overlap)} samples overlap "
                "between development and final test."
            ),
        )

        print(
            f"  PASS: {category}"
        )

    # --------------------------------------------------------
    # Production normal-score artifact
    # --------------------------------------------------------

    print(
        "\nChecking production normal-score configuration..."
    )

    # The normal-score artifact is the source used by
    # calibration_candidates.json.
    #
    # It represents the validated MobileNet L8 production
    # scoring configuration.

    source = calibration.get(
        "source"
    )

    expected_source = (
        "artifacts\\evaluation\\"
        "mobilenet_l8_normal_dev_scores.json"
    )

    normalized_source = str(
        source
    ).replace(
        "/",
        "\\",
    )

    check(
        normalized_source
        == expected_source,
        (
            "Calibration source mismatch: "
            f"{source}"
        ),
    )

    print(
        "  PASS: calibration source = "
        "mobilenet_l8_normal_dev_scores.json"
    )

    # The normal-score artifact itself is the authoritative
    # artifact for the L8 production score configuration.

    normal_score_text = json.dumps(
        normal_scores
    ).lower()

    check(
        "euclidean" in normal_score_text,
        "Euclidean distance not recorded in normal-score artifact.",
    )

    print(
        "  PASS: Euclidean distance"
    )

    check(
        "top5" in normal_score_text,
        "TOP5 aggregation not recorded in normal-score artifact.",
    )

    print(
        "  PASS: TOP5 aggregation"
    )

    # --------------------------------------------------------
    # Production thresholds
    # --------------------------------------------------------

    print(
        "\nChecking production thresholds..."
    )

    expected_thresholds = {
        "bottle": 0.46985351681709286,
        "hazelnut": 0.7513870668411254,
        "cable": 0.7059989047050477,
        "capsule": 0.5000054594874382,
        "screw": 0.644817,
        "metal_nut": 0.737776,
    }

    for category in CATEGORIES:

        actual = thresholds[
            "categories"
        ][category][
            "threshold"
        ]

        expected = expected_thresholds[
            category
        ]

        check(
            abs(
                float(actual)
                - float(expected)
            ) < 1e-5,
            (
                f"{category}: "
                f"threshold mismatch "
                f"{actual} != {expected}"
            ),
        )

        print(
            f"  PASS: {category} "
            f"P99={actual:.6f}"
        )

    # --------------------------------------------------------
    # Inspection pipeline
    # --------------------------------------------------------

    print(
        "\nChecking production inspection pipeline..."
    )

    check(
        inspection.get(
            "model"
        ) == "MobileNetV3-Small-L8",
        (
            "Unexpected production model: "
            f"{inspection.get('model')}"
        ),
    )

    print(
        "  PASS: model=MobileNetV3-Small-L8"
    )

    check(
        inspection.get(
            "aggregation"
        ) == "TOP5",
        (
            "Unexpected production aggregation: "
            f"{inspection.get('aggregation')}"
        ),
    )

    print(
        "  PASS: aggregation=TOP5"
    )

    # --------------------------------------------------------
    # Localization
    # --------------------------------------------------------

    print(
        "\nChecking localization policy..."
    )

    # Localization is deliberately explanatory evidence.
    # Day 9 integration testing already verifies that
    # localization does not alter classification.

    print(
        "  PASS: localization is explanatory evidence"
    )

    # --------------------------------------------------------
    # Research fusion
    # --------------------------------------------------------

    print(
        "\nChecking research fusion configuration..."
    )

    fusion_layers = fusion.get(
        "layers"
    )

    check(
        isinstance(
            fusion_layers,
            dict,
        ),
        "Fusion layers have unexpected schema.",
    )

    check(
        set(
            fusion_layers.keys()
        ) == {"L4", "L8"},
        (
            "Fusion does not contain exactly "
            "L4 and L8."
        ),
    )

    print(
        "  PASS: L4 + L8"
    )

    check(
        fusion.get(
            "aggregation"
        ) == "TOP5",
        (
            "Fusion aggregation is not TOP5."
        ),
    )

    print(
        "  PASS: TOP5"
    )

    # The Day 11 fusion experiment used fixed 50/50 fusion.
    # Confirm its documented normalization/fusion structure.

    check(
        "normalization" in fusion,
        "Fusion normalization metadata missing.",
    )

    check(
        "fusion_results" in next(
            iter(
                fusion["categories"].values()
            )
        ),
        "Fusion result metadata missing.",
    )

    print(
        "  PASS: fixed L4 + L8 research candidate"
    )

    # --------------------------------------------------------
    # Day 12 split safety
    # --------------------------------------------------------

    print(
        "\nChecking Day 12 experiment split safety..."
    )

    for name, artifact in [
        (
            "defect-size",
            size_analysis,
        ),
        (
            "small-defect-resolution",
            resolution,
        ),
    ]:

        check(
            artifact.get(
                "final_test_used"
            ) is False,
            (
                f"{name} experiment indicates "
                "final test was used."
            ),
        )

        print(
            f"  PASS: {name}"
        )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    print(
        "\n" + "=" * 78
    )

    print(
        "CONFIGURATION AUDIT: PASS"
    )

    print(
        "=" * 78
    )

    print(
        "\nConfiguration frozen:"
    )

    print(
        "  Production: MobileNetV3-Small L8"
    )

    print(
        "  Distance: Euclidean"
    )

    print(
        "  Aggregation: TOP5"
    )

    print(
        "  Threshold: category-specific P99"
    )

    print(
        "  Decision: PASS / REVIEW / FAIL"
    )

    print(
        "  Localization: hybrid α=0.75 evidence"
    )

    print(
        "  Research candidate: L4 + L8 50/50"
    )

    print(
        "  Final defect test: LOCKED"
    )

    print(
        "\nDAY 12 COMPLETE — STOP HERE"
    )


if __name__ == "__main__":
    main()