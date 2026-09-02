"""Versioned, fail-closed confidence calibration.

The runtime deliberately has no dependency on scikit-learn or pickle.  Training
tools export the small logistic-regression artifact described by
``CalibrationArtifact`` and this module evaluates it with the Python standard
library only.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bursa.features import FEATURES_VERSION
from bursa.models import RecommendedAction


CALIBRATION_SCHEMA_VERSION = 1
FEATURE_KEYS = (
    "name_alias_similarity",
    "guardian_relationship",
    "amount_to_balance_agreement",
    "historical_payer_consistency",
    "candidate_separation",
    "llm_ranking_consistency",
    "constraint_validation_result",
    "budget_shed",
    "candidate_count",
)
_LEGACY_SCORE_KEYS = FEATURE_KEYS[:7]


class CalibrationError(ValueError):
    """The artifact is malformed, incompatible, or belongs to another model."""


@dataclass(frozen=True)
class CalibrationArtifact:
    schema_version: int
    features_version: int
    model_sha256: str
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    intercept: float
    coefficients: tuple[float, ...]
    auto_threshold: float
    review_threshold: float
    auto_enabled: bool
    fit_positive_count: int
    fit_negative_count: int
    seed: int
    source_manifest_sha256: str
    threshold_case_count: int = 0
    threshold_negative_count: int = 0
    threshold_false_positive_count: int = 0

    @classmethod
    def from_dict(
        cls, value: dict[str, Any], *, expected_model_sha256: str | None = None
    ) -> "CalibrationArtifact":
        if not isinstance(value.get("auto_enabled"), bool):
            raise CalibrationError("auto_enabled must be a boolean")
        try:
            artifact = cls(
                schema_version=int(value["schema_version"]),
                features_version=int(value["features_version"]),
                model_sha256=str(value["model_sha256"]).lower(),
                feature_names=tuple(value["feature_names"]),
                means=tuple(float(v) for v in value["means"]),
                scales=tuple(float(v) for v in value["scales"]),
                intercept=float(value["intercept"]),
                coefficients=tuple(float(v) for v in value["coefficients"]),
                auto_threshold=float(value["auto_threshold"]),
                review_threshold=float(value["review_threshold"]),
                auto_enabled=bool(value["auto_enabled"]),
                fit_positive_count=int(value["fit_positive_count"]),
                fit_negative_count=int(value["fit_negative_count"]),
                seed=int(value["seed"]),
                source_manifest_sha256=str(value["source_manifest_sha256"]),
                threshold_case_count=int(value.get("threshold_case_count", 0)),
                threshold_negative_count=int(
                    value.get("threshold_negative_count", 0)
                ),
                threshold_false_positive_count=int(
                    value.get("threshold_false_positive_count", 0)
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CalibrationError(f"invalid calibration artifact: {exc}") from exc
        artifact.validate(expected_model_sha256=expected_model_sha256)
        return artifact

    def validate(self, *, expected_model_sha256: str | None = None) -> None:
        if self.schema_version != CALIBRATION_SCHEMA_VERSION:
            raise CalibrationError("unsupported calibration schema version")
        if self.features_version != FEATURES_VERSION:
            raise CalibrationError("calibration feature version does not match runtime")
        if self.feature_names != FEATURE_KEYS:
            raise CalibrationError("calibration feature order does not match runtime")
        width = len(FEATURE_KEYS)
        if len(self.means) != width or len(self.scales) != width:
            raise CalibrationError("calibration standardization width is invalid")
        if len(self.coefficients) != width:
            raise CalibrationError("calibration coefficient width is invalid")
        numeric = (
            *self.means,
            *self.scales,
            self.intercept,
            *self.coefficients,
            self.auto_threshold,
            self.review_threshold,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise CalibrationError("calibration contains a non-finite number")
        if any(scale <= 0 for scale in self.scales):
            raise CalibrationError("calibration scales must be positive")
        if not 0 <= self.review_threshold <= self.auto_threshold <= 1:
            raise CalibrationError("calibration thresholds are invalid")
        if self.auto_threshold < 0.90:
            raise CalibrationError("automatic threshold may not be below 0.90")
        if self.review_threshold != 0.65:
            raise CalibrationError("review threshold must remain 0.65")
        if self.fit_positive_count < 0 or self.fit_negative_count < 0:
            raise CalibrationError("calibration sample counts are invalid")
        if self.auto_enabled and (
            self.fit_positive_count < 25 or self.fit_negative_count < 25
        ):
            raise CalibrationError(
                "automatic posting requires at least 25 positive and 25 negative fits"
            )
        if self.auto_enabled and (
            self.threshold_case_count < 200
            or self.threshold_false_positive_count != 0
        ):
            raise CalibrationError(
                "automatic posting requires at least 200 held-out threshold cases "
                "and zero false positives"
            )
        if not (
            0
            <= self.threshold_negative_count
            <= self.threshold_case_count
        ):
            raise CalibrationError("threshold sample counts are invalid")
        if len(self.model_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.model_sha256
        ):
            raise CalibrationError("model_sha256 must be a lowercase SHA-256 digest")
        if expected_model_sha256 and self.model_sha256 != expected_model_sha256.lower():
            raise CalibrationError("calibration artifact belongs to a different model")
        if len(self.source_manifest_sha256) != 64 or any(
            char not in "0123456789abcdef"
            for char in self.source_manifest_sha256.lower()
        ):
            raise CalibrationError("source_manifest_sha256 must be a SHA-256 digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "features_version": self.features_version,
            "model_sha256": self.model_sha256,
            "feature_names": list(self.feature_names),
            "means": list(self.means),
            "scales": list(self.scales),
            "intercept": self.intercept,
            "coefficients": list(self.coefficients),
            "auto_threshold": self.auto_threshold,
            "review_threshold": self.review_threshold,
            "auto_enabled": self.auto_enabled,
            "fit_positive_count": self.fit_positive_count,
            "fit_negative_count": self.fit_negative_count,
            "seed": self.seed,
            "source_manifest_sha256": self.source_manifest_sha256,
            "threshold_case_count": self.threshold_case_count,
            "threshold_negative_count": self.threshold_negative_count,
            "threshold_false_positive_count": (
                self.threshold_false_positive_count
            ),
        }


class ModelConfidencePolicy:
    """Evaluate a fitted artifact, or fail closed to the review queue.

    The historical unweighted score remains available for diagnostics when no
    artifact is loaded.  It never grants automatic-post authority.
    """

    def __init__(self, artifact: CalibrationArtifact | None = None):
        self.artifact = artifact

    @classmethod
    def from_file(
        cls, path: str | Path, *, expected_model_sha256: str | None = None
    ) -> "ModelConfidencePolicy":
        try:
            with Path(path).open(encoding="utf-8") as handle:
                value = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise CalibrationError(f"cannot load calibration artifact: {exc}") from exc
        return cls(
            CalibrationArtifact.from_dict(
                value, expected_model_sha256=expected_model_sha256
            )
        )

    @property
    def calibrated(self) -> bool:
        return self.artifact is not None

    @property
    def automatic_posting_enabled(self) -> bool:
        return bool(self.artifact and self.artifact.auto_enabled)

    def score(self, features: dict) -> float:
        if self.artifact is None:
            values = [float(features.get(key, 0)) for key in _LEGACY_SCORE_KEYS]
            return round(sum(values) / len(values), 4) if values else 0.0
        if int(features.get("features_version", -1)) != self.artifact.features_version:
            return 0.0
        logit = self.artifact.intercept
        for key, mean, scale, coefficient in zip(
            self.artifact.feature_names,
            self.artifact.means,
            self.artifact.scales,
            self.artifact.coefficients,
        ):
            logit += coefficient * ((float(features.get(key, 0)) - mean) / scale)
        if logit >= 0:
            probability = 1 / (1 + math.exp(-logit))
        else:
            exponent = math.exp(logit)
            probability = exponent / (1 + exponent)
        return round(probability, 8)

    def route(self, features: dict) -> RecommendedAction:
        if self.artifact is None:
            return RecommendedAction.REVIEW
        if int(features.get("features_version", -1)) != self.artifact.features_version:
            return RecommendedAction.REVIEW
        if not self.artifact.auto_enabled:
            return RecommendedAction.REVIEW
        score = self.score(features)
        if score >= self.artifact.auto_threshold:
            return RecommendedAction.AUTO
        if score < self.artifact.review_threshold:
            return RecommendedAction.UNMATCHED
        return RecommendedAction.REVIEW
