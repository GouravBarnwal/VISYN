from pathlib import Path


# ============================================================
# VISIONFORGE — PRODUCTION CONFIGURATION
# ============================================================


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = PROJECT_ROOT / "data"

ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"

SPLITS_PATH = (
    ARTIFACTS_ROOT
    / "splits"
    / "normal_splits.json"
)

CALIBRATION_PATH = (
    ARTIFACTS_ROOT
    / "evaluation"
    / "production_thresholds.json"
)


# ------------------------------------------------------------
# Categories
# ------------------------------------------------------------

CATEGORIES = (
    "bottle",
    "hazelnut",
    "cable",
    "capsule",
    "screw",
    "metal_nut",
)


# ------------------------------------------------------------
# Image preprocessing
# ------------------------------------------------------------

IMAGE_SIZE = 224

IMAGE_MEAN = (
    0.485,
    0.456,
    0.406,
)

IMAGE_STD = (
    0.229,
    0.224,
    0.225,
)


# ------------------------------------------------------------
# Feature architecture
# ------------------------------------------------------------

MODEL_NAME = "MobileNetV3-Small"

FEATURE_LAYERS = {
    "L4": 4,
    "L8": 8,
}

L4_CHANNELS = 40

L8_CHANNELS = 48

PATCHES_PER_LAYER = 196


# ------------------------------------------------------------
# Anomaly scoring
# ------------------------------------------------------------

DISTANCE = "euclidean_equivalent"

AGGREGATION = "TOP5"

TOP_K = 5

CHUNK_SIZE = 8192


# ------------------------------------------------------------
# Fusion
# ------------------------------------------------------------

FUSION_WEIGHT_L4 = 0.5

FUSION_WEIGHT_L8 = 0.5


# ------------------------------------------------------------
# Decision policy
# ------------------------------------------------------------

THRESHOLD_PERCENTILE = 99.0

REVIEW_MULTIPLIER = 1.10


# ------------------------------------------------------------
# Localization
# ------------------------------------------------------------

LOCALIZATION_METHOD = (
    "l4_l8_patch_distance_p95_largest_component"
)

LOCALIZATION_ALPHA = None


# ------------------------------------------------------------
# Reference bank
# ------------------------------------------------------------

REFERENCE_RATIO = 0.80

RANDOM_SEED = 42


# ------------------------------------------------------------
# Runtime
# ------------------------------------------------------------

DEFAULT_BATCH_SIZE = 1

BULK_BATCH_SIZE = 8


# ------------------------------------------------------------
# Validation tolerances
# ------------------------------------------------------------

FEATURE_EQUIVALENCE_TOLERANCE = 1e-5

SCORE_EQUIVALENCE_TOLERANCE = 1e-5