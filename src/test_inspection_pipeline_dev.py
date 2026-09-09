from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from torchvision import models, transforms

from build_inspection_result import (
    build_inspection_result,
)


# ============================================================
# CONFIG
# ============================================================

DATA_ROOT = Path("data")

NORMAL_SCORES_PATH = Path(
    "artifacts/evaluation/mobilenet_l8_normal_dev_scores.json"
)

DEFECT_SCORES_PATH = Path(
    "artifacts/evaluation/mobilenet_l8_dev_scores.json"
)

CALIBRATION_PATH = Path(
    "artifacts/evaluation/calibration_candidates.json"
)

GLOBAL_BANK_ROOT = Path(
    "artifacts/patch_banks/mobilenet_l8"
)

OUTPUT_PATH = Path(
    "artifacts/evaluation/inspection_pipeline_dev_results.json"
)

CATEGORIES = [
    "bottle",
    "hazelnut",
    "cable",
    "capsule",
    "screw",
    "metal_nut",
]

REVIEW_MULTIPLIER = 1.10

SCORE_TOLERANCE = 1e-5


# ============================================================
# MODEL
# ============================================================

class MobileNetL8(nn.Module):

    def __init__(self):

        super().__init__()

        backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT
        )

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

    if feature_map.shape != (
        14,
        14,
        48,
    ):

        raise RuntimeError(
            f"Unexpected feature shape "
            f"for {image_path}: "
            f"{feature_map.shape}"
        )

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
# SCORE
# ============================================================

def calculate_top5_score(
    feature_map,
    reference_bank,
):

    h, w, d = feature_map.shape

    query = feature_map.reshape(
        -1,
        d,
    )

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

    nearest = (
        distances
        .min(dim=1)
        .values
        .cpu()
        .numpy()
    )

    nearest = np.sort(
        nearest
    )

    return float(
        np.mean(
            nearest[-5:]
        )
    )


# ============================================================
# JSON LOADING
# ============================================================

def load_json(path):

    if not path.exists():

        raise FileNotFoundError(
            f"Missing artifact: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


# ============================================================
# GET P99 THRESHOLD
# ============================================================

def get_p99_threshold(
    calibration,
    category,
):

    candidates = (
        calibration[
            "categories"
        ][category][
            "percentile_candidates"
        ]
    )

    for candidate in candidates:

        if candidate[
            "percentile"
        ] == 99.0:

            return float(
                candidate[
                    "threshold"
                ]
            )

    raise ValueError(
        f"P99 threshold not found for "
        f"{category}"
    )


# ============================================================
# EXPECTED POLICY DECISION
# ============================================================

def expected_decision(
    category,
    score,
    calibration,
):

    threshold = get_p99_threshold(
        calibration,
        category,
    )

    review_threshold = (
        threshold
        * REVIEW_MULTIPLIER
    )

    if score < threshold:

        return "PASS"

    if score < review_threshold:

        return "REVIEW"

    return "FAIL"


# ============================================================
# INSPECTION RESULT
# ============================================================

def run_inspection_result(
    category,
    score,
    evidence=None,
):

    return build_inspection_result(
        category,
        score,
        evidence=evidence,
    )


# ============================================================
# TEST ONE SAMPLE
# ============================================================

def test_sample(
    model,
    category,
    relative_path,
    reference_bank,
    calibration,
    expected_score,
):

    image_path = (
        DATA_ROOT
        / relative_path
    )

    if not image_path.exists():

        raise FileNotFoundError(
            f"Missing image: {image_path}"
        )

    feature_map = extract_feature_map(
        model,
        image_path,
    )

    actual_score = calculate_top5_score(
        feature_map,
        reference_bank,
    )

    score_difference = (
        actual_score
        -
        expected_score
    )

    score_ok = (
        abs(score_difference)
        <= SCORE_TOLERANCE
    )

    expected = expected_decision(
        category,
        actual_score,
        calibration,
    )

    evidence = None

    if expected in (
        "REVIEW",
        "FAIL",
    ):

        evidence = {
            "integration_test": True,
            "method": (
                "hybrid_alpha_0.75_"
                "p95_largest_component"
            ),
        }

    result = run_inspection_result(
        category=category,
        score=actual_score,
        evidence=evidence,
    )

    actual_decision = result[
        "decision"
    ]

    decision_ok = (
        actual_decision
        ==
        expected
    )

    return {
        "path": relative_path,
        "score": float(actual_score),
        "expected_score": float(
            expected_score
        ),
        "score_difference": float(
            score_difference
        ),
        "score_ok": bool(
            score_ok
        ),
        "expected_decision": (
            expected
        ),
        "actual_decision": (
            actual_decision
        ),
        "decision_ok": bool(
            decision_ok
        ),
        "result": result,
    }


# ============================================================
# PRINT SAMPLE RESULT
# ============================================================

def print_sample_result(
    result,
):

    filename = Path(
        result["path"]
    ).name

    print(
        f"{filename:<20} "
        f"score={result['score']:.6f} "
        f"decision={result['actual_decision']}"
    )

    print(
        f"expected="
        f"{result['expected_score']:.6f} "
        f"difference="
        f"{result['score_difference']:+.6f} "
        f"["
        f"{'PASS' if result['score_ok'] else 'FAIL'}"
        f"]"
    )

    print(
        f"policy expected="
        f"{result['expected_decision']} "
        f"actual="
        f"{result['actual_decision']} "
        f"["
        f"{'PASS' if result['decision_ok'] else 'FAIL'}"
        f"]"
    )

    localization = result[
        "result"
    ].get(
        "localization"
    )

    if localization:

        if localization.get(
            "available"
        ):

            print(
                "    localization evidence attached"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print(
        "VISIONFORGE — END-TO-END DEVELOPMENT "
        "INSPECTION TEST"
    )
    print("=" * 72)

    print()
    print(
        "Model: MobileNetV3-Small L8"
    )

    print(
        "Aggregation: TOP5"
    )

    print(
        "Distance: Euclidean (torch.cdist)"
    )

    print(
        "Decision: P99 + 1.10x REVIEW boundary"
    )

    print(
        "Localization: Hybrid alpha=0.75 + "
        "P95 + largest component"
    )

    print(
        "Defect set: DEVELOPMENT ONLY"
    )

    print(
        "Integration criterion: "
        "score reproduction + policy consistency"
    )

    # ========================================================
    # LOAD ARTIFACTS
    # ========================================================

    normal_scores = load_json(
        NORMAL_SCORES_PATH
    )

    defect_scores = load_json(
        DEFECT_SCORES_PATH
    )

    calibration = load_json(
        CALIBRATION_PATH
    )

    # ========================================================
    # SAFETY CHECKS
    # ========================================================

    if normal_scores.get(
        "distance_metric"
    ) != "euclidean_cdist_p2":

        raise RuntimeError(
            "Normal-development score artifact "
            "does not use Euclidean cdist."
        )

    if normal_scores.get(
        "aggregation"
    ) != "TOP5":

        raise RuntimeError(
            "Normal-development score artifact "
            "does not use TOP5."
        )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    print()
    print(
        "Loading MobileNetV3-Small L8..."
    )

    model = MobileNetL8()
    model.eval()

    print(
        "Model loaded."
    )

    # ========================================================
    # RESULT STORAGE
    # ========================================================

    normal_results = []
    defect_results = []

    score_errors = []
    decision_errors = []

    # ========================================================
    # CATEGORY LOOP
    # ========================================================

    for category in CATEGORIES:

        print()
        print(
            f"CATEGORY: {category}"
        )

        print(
            "-" * 34
        )

        # ----------------------------------------------------
        # GLOBAL REFERENCE BANK
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

        if reference_bank.ndim != 2:

            raise RuntimeError(
                f"Unexpected bank shape "
                f"for {category}: "
                f"{reference_bank.shape}"
            )

        if reference_bank.shape[1] != 48:

            raise RuntimeError(
                f"Unexpected descriptor dimension "
                f"for {category}: "
                f"{reference_bank.shape}"
            )

        # ====================================================
        # NORMAL DEVELOPMENT
        # ====================================================

        print()
        print(
            "[NORMAL]"
        )

        normal_samples = (
            normal_scores[
                "categories"
            ][category][
                "samples"
            ]
        )

        normal_samples = normal_samples[:2]

        for sample in normal_samples:

            relative_path = sample[
                "path"
            ]

            expected_score = float(
                sample["score"]
            )

            result = test_sample(
                model=model,
                category=category,
                relative_path=relative_path,
                reference_bank=reference_bank,
                calibration=calibration,
                expected_score=expected_score,
            )

            normal_results.append(
                result
            )

            if not result["score_ok"]:

                score_errors.append(
                    result
                )

            if not result["decision_ok"]:

                decision_errors.append(
                    result
                )

            print_sample_result(
                result
            )

        # ====================================================
        # DEFECT DEVELOPMENT
        # ====================================================

        print()
        print(
            "[DEFECT]"
        )

        category_defect_data = (
            defect_scores[
                category
            ]
        )

        if not isinstance(
            category_defect_data,
            dict,
        ):

            raise RuntimeError(
                f"Unexpected defect score structure "
                f"for {category}: "
                f"{type(category_defect_data)}"
            )

        if "defect_dev" not in category_defect_data:

            raise RuntimeError(
                f"'defect_dev' not found for {category}. "
                f"Available keys: "
                f"{list(category_defect_data.keys())}"
            )

        defect_samples = (
            category_defect_data[
                "defect_dev"
            ]
        )

        if not isinstance(
            defect_samples,
            list,
        ):

            raise RuntimeError(
                f"Expected defect_dev to be a list "
                f"for {category}, got "
                f"{type(defect_samples)}"
            )

        defect_samples = defect_samples[:2]

        for sample in defect_samples:

            relative_path = sample[
                "path"
            ]

            expected_score = float(
                sample["top5"]
            )

            result = test_sample(
                model=model,
                category=category,
                relative_path=relative_path,
                reference_bank=reference_bank,
                calibration=calibration,
                expected_score=expected_score,
            )

            defect_results.append(
                result
            )

            if not result["score_ok"]:

                score_errors.append(
                    result
                )

            if not result["decision_ok"]:

                decision_errors.append(
                    result
                )

            print_sample_result(
                result
            )

    # ========================================================
    # VALIDATION
    # ========================================================

    score_consistency_pass = (
        len(score_errors) == 0
    )

    normal_decision_pass = all(
        item["decision_ok"]
        for item in normal_results
    )

    defect_policy_consistency_pass = all(
        item["decision_ok"]
        for item in defect_results
    )

    overall_pass = (
        score_consistency_pass
        and normal_decision_pass
        and defect_policy_consistency_pass
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 72)
    print(
        "DAY 9 INTEGRATION VALIDATION"
    )
    print("=" * 72)

    print(
        "Score consistency:      "
        f"{'PASS' if score_consistency_pass else 'FAIL'}"
    )

    print(
        "Normal decision test:   "
        f"{'PASS' if normal_decision_pass else 'FAIL'}"
    )

    print(
        "Defect policy test:     "
        f"{'PASS' if defect_policy_consistency_pass else 'FAIL'}"
    )

    print()
    print(
        "Integration verifies:"
    )

    print(
        "1. Production scoring reproduces the "
        "validated MobileNet L8 TOP5 score."
    )

    print(
        "2. Production classification reproduces "
        "the calibrated P99 + 1.10x policy."
    )

    print(
        "3. Localization remains explanatory evidence "
        "and does not modify classification."
    )

    print()
    print(
        "Known defect misses are NOT treated as "
        "integration errors."
    )

    print(
        "They are measured separately by the "
        "full development-set evaluation."
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    output = {

        "model": (
            "MobileNetV3-Small-L8"
        ),

        "distance": (
            "Euclidean torch.cdist p=2"
        ),

        "aggregation": "TOP5",

        "review_multiplier": (
            REVIEW_MULTIPLIER
        ),

        "calibration_source": (
            str(CALIBRATION_PATH)
        ),

        "normal_score_source": (
            str(NORMAL_SCORES_PATH)
        ),

        "defect_score_source": (
            str(DEFECT_SCORES_PATH)
        ),

        "validation": {

            "score_consistency": (
                score_consistency_pass
            ),

            "normal_decision_test": (
                normal_decision_pass
            ),

            "defect_policy_test": (
                defect_policy_consistency_pass
            ),

            "overall": (
                overall_pass
            ),
        },

        "normal_samples": (
            normal_results
        ),

        "defect_samples": (
            defect_results
        ),

        "score_errors": (
            score_errors
        ),

        "decision_errors": (
            decision_errors
        ),
    }

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
    print(
        f"Results saved to: "
        f"{OUTPUT_PATH}"
    )

    print()

    # ========================================================
    # FINAL DAY 9 STATUS
    # ========================================================

    if overall_pass:

        print(
            "DAY 9 VALIDATION: PASS"
        )

        print()
        print(
            "DAY 9 COMPLETE — STOP HERE"
        )

    else:

        print(
            "DAY 9 VALIDATION: FAIL"
        )

        print()
        print(
            f"Score errors: "
            f"{len(score_errors)}"
        )

        print(
            f"Decision errors: "
            f"{len(decision_errors)}"
        )

        print()
        print(
            "Day 9 is NOT complete."
        )


if __name__ == "__main__":

    main()