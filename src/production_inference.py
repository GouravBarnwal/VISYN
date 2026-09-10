from pathlib import Path
import gc
import json

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    mobilenet_v3_small,
)

from src.mobilenet_batch_extractor import load_batch

from src.production_config import (
    ARTIFACTS_ROOT,
    CATEGORIES,
    CHUNK_SIZE,
    FUSION_WEIGHT_L4,
    FUSION_WEIGHT_L8,
    REVIEW_MULTIPLIER,
    SCORE_EQUIVALENCE_TOLERANCE,
    TOP_K,
)

REFERENCE_BANK_ROOT = ARTIFACTS_ROOT / "production" / "reference_banks"

REFERENCE_BANK_MANIFEST = (
    ARTIFACTS_ROOT / "production" / "reference_bank_manifest.json"
)


class ProductionInferenceEngine:
    """
    VISYN production inference engine.

    Memory-optimized deployment architecture:

        - MobileNet model loaded once.
        - Thresholds loaded once.
        - Reference-bank metadata loaded once.
        - Only the currently requested category's L4/L8
          reference banks are kept in memory.

    Production scoring remains unchanged:

        MobileNetV3-Small L4 + L8
        normalized patch features
        Euclidean-equivalent distance
        TOP5 aggregation
        raw 50/50 fusion
        calibrated PASS/REVIEW/FAIL policy
    """

    def __init__(
        self,
        device=None,
    ):
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cpu")
        )

        # Only the currently loaded category lives here.
        self.reference_banks = {}

        # Paths and shapes for all categories remain lightweight metadata.
        self.reference_bank_paths = {}

        self.reference_bank_shapes = {}

        self.thresholds = {}

        self.active_category = None

        self.model = self._load_model()

        self._load_reference_bank_metadata()
        self._load_thresholds()

        self._validate_configuration()

    # ========================================================
    # MODEL
    # ========================================================

    def _load_model(self):
        weights = MobileNet_V3_Small_Weights.DEFAULT

        model = mobilenet_v3_small(
            weights=weights,
        )

        model.eval()
        model.to(self.device)

        return model

    # ========================================================
    # JSON
    # ========================================================

    @staticmethod
    def _load_json(path):
        if not path.exists():
            raise FileNotFoundError(
                f"Required artifact not found: {path}"
            )

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:
            return json.load(f)

    # ========================================================
    # REFERENCE-BANK METADATA
    # ========================================================

    def _load_reference_bank_metadata(self):
        """
        Validate and register all reference-bank files without
        loading their contents into RAM.
        """

        manifest = self._load_json(
            REFERENCE_BANK_MANIFEST
        )

        if manifest.get("model") != "MobileNetV3-Small":
            raise RuntimeError(
                "Reference-bank model does not match "
                "the production model."
            )

        feature_layers = manifest.get(
            "feature_layers",
            {},
        )

        if feature_layers.get("L4", {}).get("layer_index") != 4:
            raise RuntimeError(
                "Reference-bank L4 configuration mismatch."
            )

        if feature_layers.get("L8", {}).get("layer_index") != 8:
            raise RuntimeError(
                "Reference-bank L8 configuration mismatch."
            )

        categories = manifest.get(
            "categories",
            {},
        )

        for category in CATEGORIES:
            category_data = categories.get(category)

            if category_data is None:
                raise RuntimeError(
                    f"Missing manifest entry for {category}."
                )

            l4_path = (
                REFERENCE_BANK_ROOT
                / category
                / "L4.npy"
            )

            l8_path = (
                REFERENCE_BANK_ROOT
                / category
                / "L8.npy"
            )

            if not l4_path.exists():
                raise FileNotFoundError(
                    f"Missing L4 bank: {l4_path}"
                )

            if not l8_path.exists():
                raise FileNotFoundError(
                    f"Missing L8 bank: {l8_path}"
                )

            expected_l4_shape = tuple(
                category_data["l4_shape"]
            )

            expected_l8_shape = tuple(
                category_data["l8_shape"]
            )

            if len(expected_l4_shape) != 2:
                raise RuntimeError(
                    f"{category}: invalid L4 shape "
                    f"{expected_l4_shape}."
                )

            if len(expected_l8_shape) != 2:
                raise RuntimeError(
                    f"{category}: invalid L8 shape "
                    f"{expected_l8_shape}."
                )

            if expected_l4_shape[1] != 40:
                raise RuntimeError(
                    f"{category}: invalid L4 "
                    "feature dimension."
                )

            if expected_l8_shape[1] != 48:
                raise RuntimeError(
                    f"{category}: invalid L8 "
                    "feature dimension."
                )

            self.reference_bank_paths[category] = {
                "L4": l4_path,
                "L8": l8_path,
            }

            self.reference_bank_shapes[category] = {
                "L4": expected_l4_shape,
                "L8": expected_l8_shape,
            }

    # ========================================================
    # LAZY REFERENCE-BANK LOADING
    # ========================================================

    def _release_reference_bank(self):
        """
        Release the currently loaded category from memory.
        """

        self.reference_banks.clear()
        self.active_category = None

        gc.collect()

    def _load_category_bank(self, category):
        """
        Load only one category's L4/L8 reference banks.

        If another category is currently loaded, release it first.
        """

        if category not in CATEGORIES:
            raise ValueError(
                f"Unsupported category: {category}"
            )

        if self.active_category == category:
            return

        self._release_reference_bank()

        paths = self.reference_bank_paths[category]

        expected_shapes = self.reference_bank_shapes[
            category
        ]

        # Load L4.
        l4 = np.load(
            paths["L4"],
        )

        if l4.shape != expected_shapes["L4"]:
            raise RuntimeError(
                f"{category}: L4 bank shape mismatch. "
                f"Expected {expected_shapes['L4']}, "
                f"got {l4.shape}."
            )

        # Load L8.
        l8 = np.load(
            paths["L8"],
        )

        if l8.shape != expected_shapes["L8"]:
            raise RuntimeError(
                f"{category}: L8 bank shape mismatch. "
                f"Expected {expected_shapes['L8']}, "
                f"got {l8.shape}."
            )

        # Convert the selected category only.
        l4_tensor = torch.from_numpy(
            np.asarray(l4)
        ).to(self.device)

        l8_tensor = torch.from_numpy(
            np.asarray(l8)
        ).to(self.device)

        self.reference_banks[category] = {
            "L4": l4_tensor,
            "L8": l8_tensor,
        }

        self.active_category = category

    def get_reference_bank(self, category):
        """
        Return the selected category's production bank.

        This is the public access point used by inference
        and localization.
        """

        self._load_category_bank(category)

        return self.reference_banks[category]

    # ========================================================
    # THRESHOLDS
    # ========================================================

    def _load_thresholds(self):
        calibration_path = (
            ARTIFACTS_ROOT
            / "evaluation"
            / "production_thresholds.json"
        )

        calibration = self._load_json(
            calibration_path
        )

        categories = calibration.get(
            "categories",
            {},
        )

        for category in CATEGORIES:
            category_data = categories.get(
                category
            )

            if category_data is None:
                raise RuntimeError(
                    f"Missing production threshold "
                    f"for {category}."
                )

            if "threshold" in category_data:
                threshold = category_data["threshold"]

            elif "p99_threshold" in category_data:
                threshold = category_data[
                    "p99_threshold"
                ]

            elif "percentile_candidates" in category_data:
                threshold = None

                for candidate in category_data[
                    "percentile_candidates"
                ]:
                    if (
                        float(candidate["percentile"])
                        == 99.0
                    ):
                        threshold = candidate[
                            "threshold"
                        ]
                        break

                if threshold is None:
                    raise RuntimeError(
                        f"P99 threshold not found "
                        f"for {category}."
                    )

            else:
                raise RuntimeError(
                    f"No usable threshold found "
                    f"for {category}."
                )

            self.thresholds[category] = float(
                threshold
            )

    # ========================================================
    # CONFIGURATION VALIDATION
    # ========================================================

    def _validate_configuration(self):
        missing_paths = [
            category
            for category in CATEGORIES
            if category not in self.reference_bank_paths
        ]

        if missing_paths:
            raise RuntimeError(
                "Missing reference banks for: "
                + ", ".join(missing_paths)
            )

        missing_thresholds = [
            category
            for category in CATEGORIES
            if category not in self.thresholds
        ]

        if missing_thresholds:
            raise RuntimeError(
                "Missing thresholds for: "
                + ", ".join(missing_thresholds)
            )

        if not 0.0 <= FUSION_WEIGHT_L4 <= 1.0:
            raise RuntimeError(
                "Invalid L4 fusion weight."
            )

        if not 0.0 <= FUSION_WEIGHT_L8 <= 1.0:
            raise RuntimeError(
                "Invalid L8 fusion weight."
            )

        if (
            abs(
                FUSION_WEIGHT_L4
                + FUSION_WEIGHT_L8
                - 1.0
            )
            > SCORE_EQUIVALENCE_TOLERANCE
        ):
            raise RuntimeError(
                "Fusion weights must sum to 1."
            )

    # ========================================================
    # FEATURE EXTRACTION
    # ========================================================

    @torch.no_grad()
    def extract_features(
        self,
        image_paths,
    ):
        batch = load_batch(image_paths)

        batch = batch.to(self.device)

        x = batch

        l4 = None
        l8 = None

        for index, layer in enumerate(
            self.model.features
        ):
            x = layer(x)

            if index == 4:
                l4 = x

            elif index == 8:
                l8 = x

        if l4 is None or l8 is None:
            raise RuntimeError(
                "Failed to capture L4/L8 features."
            )

        batch_size = l4.shape[0]

        l4 = l4.permute(
            0,
            2,
            3,
            1,
        ).reshape(
            batch_size,
            -1,
            l4.shape[1],
        )

        l8 = l8.permute(
            0,
            2,
            3,
            1,
        ).reshape(
            batch_size,
            -1,
            l8.shape[1],
        )

        l4 = F.normalize(
            l4,
            p=2,
            dim=2,
        )

        l8 = F.normalize(
            l8,
            p=2,
            dim=2,
        )

        return l4, l8

    # ========================================================
    # SCORING
    # ========================================================

    @torch.no_grad()
    def _score_chunked(
        self,
        query,
        reference_bank,
    ):
        best_similarity = torch.full(
            (query.shape[0],),
            -float("inf"),
            dtype=query.dtype,
            device=query.device,
        )

        num_references = reference_bank.shape[0]

        for start in range(
            0,
            num_references,
            CHUNK_SIZE,
        ):
            end = min(
                start + CHUNK_SIZE,
                num_references,
            )

            reference_chunk = reference_bank[
                start:end
            ]

            similarity = (
                query @ reference_chunk.T
            )

            chunk_best = similarity.max(
                dim=1
            ).values

            best_similarity = torch.maximum(
                best_similarity,
                chunk_best,
            )

        squared_distance = (
            2.0 - 2.0 * best_similarity
        )

        squared_distance = torch.clamp(
            squared_distance,
            min=0.0,
        )

        distances = torch.sqrt(
            squared_distance
        )

        if distances.shape[0] < TOP_K:
            raise RuntimeError(
                "Reference bank contains fewer "
                "than TOP_K distances."
            )

        topk = torch.topk(
            distances,
            k=TOP_K,
            largest=True,
        ).values

        return topk.mean()

    # ========================================================
    # DECISION
    # ========================================================

    @staticmethod
    def _classify(
        score,
        threshold,
    ):
        review_threshold = (
            threshold * REVIEW_MULTIPLIER
        )

        if score < threshold:
            decision = "PASS"

        elif score < review_threshold:
            decision = "REVIEW"

        else:
            decision = "FAIL"

        return decision, review_threshold

    # ========================================================
    # BATCH INSPECTION
    # ========================================================

    @torch.no_grad()
    def inspect_batch(
        self,
        image_paths,
        category,
    ):
        if category not in CATEGORIES:
            raise ValueError(
                f"Unsupported category: {category}"
            )

        if not image_paths:
            raise ValueError(
                "image_paths cannot be empty."
            )

        # Load ONLY the requested category.
        reference_bank = self.get_reference_bank(
            category
        )

        l4_queries, l8_queries = (
            self.extract_features(image_paths)
        )

        l4_bank = reference_bank["L4"]
        l8_bank = reference_bank["L8"]

        threshold = self.thresholds[category]

        results = []

        for index, image_path in enumerate(
            image_paths
        ):
            l4_score = self._score_chunked(
                l4_queries[index],
                l4_bank,
            )

            l8_score = self._score_chunked(
                l8_queries[index],
                l8_bank,
            )

            fusion_score = (
                FUSION_WEIGHT_L4 * l4_score
                + FUSION_WEIGHT_L8 * l8_score
            )

            score = float(
                fusion_score.item()
            )

            decision, review_threshold = (
                self._classify(
                    score,
                    threshold,
                )
            )

            results.append(
                {
                    "image_path": str(
                        Path(image_path)
                    ),
                    "category": category,
                    "l4_score": float(
                        l4_score.item()
                    ),
                    "l8_score": float(
                        l8_score.item()
                    ),
                    "anomaly_score": score,
                    "threshold": float(
                        threshold
                    ),
                    "review_threshold": float(
                        review_threshold
                    ),
                    "decision": decision,
                }
            )

        return results

    # ========================================================
    # SINGLE IMAGE
    # ========================================================

    def inspect(
        self,
        image_path,
        category,
    ):
        return self.inspect_batch(
            [image_path],
            category,
        )[0]