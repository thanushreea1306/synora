from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

try:
    from ml.src.inference import rf_feature_builder as feature_builder
except ModuleNotFoundError as exc:
    if exc.name not in ("ml", "ml.src"):
        raise
    SOURCE_ROOT = str(Path(__file__).resolve().parents[1])
    if SOURCE_ROOT not in sys.path:
        sys.path.insert(0, SOURCE_ROOT)
    import inference.rf_feature_builder as feature_builder


SERVICE_VERSION = "synora_rf_service_v1"
EXPERIMENT_ID = "synora_baseline_v1_20261121"
EXPECTED_FEATURE_COUNT = feature_builder.EXPECTED_FEATURE_COUNT
EXPECTED_CLASS_COUNT = 21
EXPECTED_MODEL_CLASSES = tuple(range(EXPECTED_CLASS_COUNT))
STATUS_OK = "ok"
STATUS_INVALID_RECORDING = "invalid_recording"
STATUS_FEATURE_ERROR = "feature_error"
STATUS_MODEL_ERROR = "model_error"
CONFIDENCE_NOTE = (
    "predict_proba confidence is raw model output; it is not calibrated and is not "
    "a validated rejection threshold"
)
DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "synora_baseline_v1_20261121"
    / "random_forest"
    / "random_forest.joblib"
)
DEFAULT_MODEL_CONFIG_PATH = DEFAULT_MODEL_PATH.with_name("config.json")
DEFAULT_FEATURE_SCHEMA_PATH = feature_builder.DEFAULT_RF_SCHEMA_PATH
DEFAULT_PRIMARY_CLASSES_PATH = (
    Path(__file__).resolve().parents[2]
    / "dataset"
    / "processed"
    / "synora_include21_video_features_v1"
    / "config.json"
)
DEFAULT_LABEL_MAPPING_PATH = (
    Path(__file__).resolve().parents[2]
    / "reports"
    / "synora_baseline_v1_20261121"
    / "experiment_config.json"
)
EXPECTED_MODEL_SHA256 = (
    "cdfb6673a6499f999c263c8503e2ae1d48ff888165d1bfa478db73c44bab65c5"
)
EXPECTED_FEATURE_SCHEMA_SHA256 = (
    "dc68d8b4d3f1096d41c83ad2b0a4db7b04432b35fbd68e7d85e1e6aeeb17db63"
)


class RFServiceConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class LabelMapping:
    model_index: int
    original_label_id: int
    class_label: str
    class_folder: str
    folder_id: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_index": self.model_index,
            "original_label_id": self.original_label_id,
            "class_label": self.class_label,
            "class_folder": self.class_folder,
            "folder_id": self.folder_id,
        }


class RFInferenceService:
    def __init__(
        self,
        model_path: str | Path | None = None,
        label_mapping_path: str | Path | None = None,
        feature_schema_path: str | Path | None = None,
        primary_classes_path: str | Path | None = None,
        model_loader: Callable[[Path], Any] | None = None,
        feature_builder_fn: Callable[..., Any] | None = None,
        expected_model_sha256: str | None = EXPECTED_MODEL_SHA256,
        expected_feature_schema_sha256: str | None = EXPECTED_FEATURE_SCHEMA_SHA256,
    ) -> None:
        self.model_path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
        self.label_mapping_path = (
            Path(label_mapping_path) if label_mapping_path is not None else DEFAULT_LABEL_MAPPING_PATH
        )
        self.feature_schema_path = (
            Path(feature_schema_path) if feature_schema_path is not None else DEFAULT_FEATURE_SCHEMA_PATH
        )
        self.primary_classes_path = (
            Path(primary_classes_path)
            if primary_classes_path is not None
            else DEFAULT_PRIMARY_CLASSES_PATH
        )
        self._model_loader = model_loader if model_loader is not None else _default_model_loader
        self._feature_builder_fn = (
            feature_builder_fn if feature_builder_fn is not None else feature_builder.build_rf_feature_row
        )

        self.model_sha256 = _sha256_file(self.model_path, "RF model")
        if expected_model_sha256 is not None and self.model_sha256 != expected_model_sha256:
            raise RFServiceConfigurationError(
                f"RF model SHA-256 mismatch: {self.model_sha256} != {expected_model_sha256}"
            )
        self.feature_schema_sha256 = _sha256_file(self.feature_schema_path, "RF feature schema")
        if (
            expected_feature_schema_sha256 is not None
            and self.feature_schema_sha256 != expected_feature_schema_sha256
        ):
            raise RFServiceConfigurationError(
                "RF feature schema SHA-256 mismatch: "
                f"{self.feature_schema_sha256} != {expected_feature_schema_sha256}"
            )
        self.label_mapping_sha256 = _sha256_file(self.label_mapping_path, "label mapping")
        self.primary_classes_sha256 = _sha256_file(self.primary_classes_path, "primary classes config")

        self.model = self._model_loader(self.model_path)
        self._validate_model(self.model)
        self.feature_names = feature_builder.load_authoritative_feature_names(
            self.feature_schema_path
        )
        self.label_mapping = self._load_label_mapping()
        self.class_positions = {
            int(value): position for position, value in enumerate(self._model_classes(self.model))
        }
        self.identity = self._build_identity()

    def predict(self, records: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        try:
            feature_row = self._feature_builder_fn(
                records,
                metadata=metadata,
                expected_feature_names=self.feature_names,
            )
        except Exception as exc:
            return self._error_result(
                STATUS_INVALID_RECORDING,
                "FEATURE_BUILD_FAILED",
                f"{type(exc).__name__}: {exc}",
            )
        return self._predict_feature_row(feature_row, {})

    def predict_capture(self, capture_result: Any) -> dict[str, Any]:
        diagnostics = {
            "capture_status": getattr(capture_result, "status", None),
            "capture_reason": getattr(capture_result, "reason", None),
            "capture_frame_count": getattr(capture_result, "frame_count", None),
        }
        warnings = [str(item) for item in getattr(capture_result, "warnings", [])]
        if getattr(capture_result, "status", None) != STATUS_OK:
            return self._error_result(
                STATUS_INVALID_RECORDING,
                "CAPTURE_NOT_OK",
                "capture result is not a successful recording",
                diagnostics=diagnostics,
                warnings=warnings,
            )
        try:
            feature_row = self._feature_builder_fn(
                capture_result.records,
                metadata=capture_result.metadata,
                expected_feature_names=self.feature_names,
            )
        except Exception as exc:
            return self._error_result(
                STATUS_INVALID_RECORDING,
                "FEATURE_BUILD_FAILED",
                f"{type(exc).__name__}: {exc}",
                diagnostics=diagnostics,
                warnings=warnings,
            )
        return self._predict_feature_row(feature_row, diagnostics, warnings)

    def _predict_feature_row(
        self,
        feature_row: Any,
        capture_diagnostics: Mapping[str, Any],
        warnings: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        try:
            values = self._validate_feature_row(feature_row)
        except Exception as exc:
            return self._error_result(
                STATUS_FEATURE_ERROR,
                "FEATURE_VALIDATION_FAILED",
                f"{type(exc).__name__}: {exc}",
                diagnostics=dict(capture_diagnostics),
                warnings=warnings,
            )

        try:
            prediction = np.asarray(self.model.predict(values))
            probabilities = np.asarray(self.model.predict_proba(values))
        except Exception as exc:
            return self._error_result(
                STATUS_MODEL_ERROR,
                "MODEL_PREDICTION_FAILED",
                f"{type(exc).__name__}: {exc}",
                diagnostics=dict(capture_diagnostics),
                warnings=warnings,
            )

        try:
            predicted_index = self._validate_prediction(prediction)
            probability_row = self._validate_probabilities(probabilities)
            raw_confidence = float(probability_row[self.class_positions[predicted_index]])
            mapping = self.label_mapping[predicted_index]
        except Exception as exc:
            return self._error_result(
                STATUS_MODEL_ERROR,
                "MODEL_OUTPUT_INVALID",
                f"{type(exc).__name__}: {exc}",
                diagnostics=dict(capture_diagnostics),
                warnings=warnings,
            )

        result_warnings = [str(item) for item in (warnings or [])]
        probability_total = float(probability_row.sum())
        if abs(probability_total - 1.0) > 1e-6:
            result_warnings.append(f"probability_total_not_one:{probability_total:.9f}")

        diagnostics = dict(capture_diagnostics)
        diagnostics.update(getattr(feature_row, "diagnostics", {}) or {})
        return {
            "status": STATUS_OK,
            "predicted_sign": mapping.class_label,
            "original_label_id": mapping.original_label_id,
            "model_class_index": predicted_index,
            "raw_confidence": raw_confidence,
            "raw_probabilities": self._probability_output(probability_row),
            "recording_diagnostics": diagnostics,
            "model": self.identity,
            "confidence_note": CONFIDENCE_NOTE,
            "warnings": result_warnings,
            "errors": [],
        }

    def _validate_feature_row(self, feature_row: Any) -> np.ndarray:
        values = getattr(feature_row, "values", None)
        if not isinstance(values, np.ndarray):
            raise ValueError("feature values must be a numpy.ndarray")
        if values.shape != (1, EXPECTED_FEATURE_COUNT):
            raise ValueError(
                f"feature shape {values.shape} != (1, {EXPECTED_FEATURE_COUNT})"
            )
        if not np.issubdtype(values.dtype, np.number):
            raise ValueError(f"feature dtype {values.dtype} is not numeric")
        if not bool(np.isfinite(values).all()):
            bad_indices = np.where(~np.isfinite(values))[0].tolist()
            raise ValueError(f"non-finite feature values at indices {bad_indices[:5]}")
        names = tuple(getattr(feature_row, "feature_names", ()))
        if names != self.feature_names:
            first_mismatch = next(
                (
                    index
                    for index, pair in enumerate(zip(names, self.feature_names))
                    if pair[0] != pair[1]
                ),
                min(len(names), len(self.feature_names)),
            )
            raise ValueError(f"feature order mismatch at index {first_mismatch}")
        return values

    def _validate_prediction(self, prediction: np.ndarray) -> int:
        if prediction.shape != (1,):
            raise ValueError(f"predict output shape {prediction.shape} != (1,)")
        try:
            predicted_index = int(prediction[0])
        except (TypeError, ValueError) as exc:
            raise ValueError("predict output is not an integer class index") from exc
        if predicted_index not in self.class_positions:
            raise ValueError(f"predict returned unknown class index {predicted_index}")
        return predicted_index

    def _validate_probabilities(self, probabilities: np.ndarray) -> np.ndarray:
        if probabilities.shape != (1, EXPECTED_CLASS_COUNT):
            raise ValueError(
                f"predict_proba shape {probabilities.shape} != (1, {EXPECTED_CLASS_COUNT})"
            )
        if not np.issubdtype(probabilities.dtype, np.number):
            raise ValueError(f"predict_proba dtype {probabilities.dtype} is not numeric")
        if not bool(np.isfinite(probabilities).all()):
            raise ValueError("predict_proba returned non-finite values")
        if bool((probabilities < 0.0).any()) or bool((probabilities > 1.0).any()):
            raise ValueError("predict_proba values must be between 0 and 1")
        return probabilities[0]

    def _probability_output(self, probability_row: np.ndarray) -> list[dict[str, Any]]:
        output = []
        for model_index in range(EXPECTED_CLASS_COUNT):
            mapping = self.label_mapping[model_index]
            output.append(
                {
                    "model_class_index": model_index,
                    "class_label": mapping.class_label,
                    "original_label_id": mapping.original_label_id,
                    "probability": float(probability_row[self.class_positions[model_index]]),
                }
            )
        return output

    def _load_label_mapping(self) -> tuple[LabelMapping, ...]:
        payload = _read_json(self.label_mapping_path, "label mapping")
        entries = payload.get("label_mapping")
        if not isinstance(entries, list) or len(entries) != EXPECTED_CLASS_COUNT:
            raise RFServiceConfigurationError(
                f"label_mapping must contain {EXPECTED_CLASS_COUNT} entries"
            )
        mapping = {}
        for position, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                raise RFServiceConfigurationError(f"label_mapping[{position}] is not an object")
            try:
                item = LabelMapping(
                    model_index=int(entry["model_index"]),
                    original_label_id=int(entry["original_label_id"]),
                    class_label=str(entry["class_label"]),
                    class_folder=str(entry["class_folder"]),
                    folder_id=int(entry["folder_id"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise RFServiceConfigurationError(
                    f"label_mapping[{position}] has invalid fields: {exc}"
                ) from exc
            if item.model_index != position:
                raise RFServiceConfigurationError(
                    f"label_mapping[{position}] has model_index {item.model_index}"
                )
            mapping[item.model_index] = item
        ordered = tuple(mapping[index] for index in range(EXPECTED_CLASS_COUNT))
        self._cross_check_primary_classes(ordered)
        return ordered

    def _cross_check_primary_classes(self, mapping: Sequence[LabelMapping]) -> None:
        payload = _read_json(self.primary_classes_path, "primary classes config")
        primary = payload.get("primary_classes")
        if not isinstance(primary, list) or len(primary) != EXPECTED_CLASS_COUNT:
            raise RFServiceConfigurationError(
                f"primary_classes must contain {EXPECTED_CLASS_COUNT} entries"
            )
        for index, item in enumerate(mapping):
            expected = primary[index]
            if not isinstance(expected, Mapping):
                raise RFServiceConfigurationError(f"primary_classes[{index}] is not an object")
            try:
                matches = (
                    int(expected["folder_id"]) == item.folder_id
                    and int(expected["label_id"]) == item.original_label_id
                    and str(expected["class_label"]) == item.class_label
                    and str(expected["class_folder"]) == item.class_folder
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise RFServiceConfigurationError(
                    f"primary_classes[{index}] has invalid fields: {exc}"
                ) from exc
            if not matches:
                raise RFServiceConfigurationError(
                    f"label mapping mismatch at model index {index} against primary_classes"
                )

    def _validate_model(self, model: Any) -> None:
        n_features = getattr(model, "n_features_in_", None)
        if n_features is None:
            raise RFServiceConfigurationError("loaded model does not expose n_features_in_")
        try:
            n_features = int(n_features)
        except (TypeError, ValueError) as exc:
            raise RFServiceConfigurationError("n_features_in_ is not an integer") from exc
        if n_features != EXPECTED_FEATURE_COUNT:
            raise RFServiceConfigurationError(
                f"loaded model expects {n_features} features; expected {EXPECTED_FEATURE_COUNT}"
            )
        model_classes = self._model_classes(model)
        if len(model_classes) != EXPECTED_CLASS_COUNT:
            raise RFServiceConfigurationError(
                f"loaded model has {len(model_classes)} classes; expected {EXPECTED_CLASS_COUNT}"
            )
        if tuple(model_classes) != EXPECTED_MODEL_CLASSES:
            raise RFServiceConfigurationError(
                f"loaded model classes {model_classes} != {list(EXPECTED_MODEL_CLASSES)}"
            )
        if not callable(getattr(model, "predict", None)):
            raise RFServiceConfigurationError("loaded model does not provide predict()")
        if not callable(getattr(model, "predict_proba", None)):
            raise RFServiceConfigurationError("loaded model does not provide predict_proba()")

    @staticmethod
    def _model_classes(model: Any) -> tuple[int, ...]:
        classes = getattr(model, "classes_", None)
        if classes is None:
            raise RFServiceConfigurationError("loaded model does not expose classes_")
        array = np.asarray(classes)
        if array.ndim != 1:
            raise RFServiceConfigurationError("model classes_ must be one-dimensional")
        try:
            return tuple(int(value) for value in array.tolist())
        except (TypeError, ValueError) as exc:
            raise RFServiceConfigurationError("model classes_ contains non-integer values") from exc

    def _build_identity(self) -> dict[str, Any]:
        model_config_path = DEFAULT_MODEL_CONFIG_PATH
        model_config_sha256 = (
            _sha256_file(model_config_path, "RF model config")
            if model_config_path.is_file()
            else None
        )
        return {
            "service_version": SERVICE_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "estimator_type": (
                type(self.model).__module__ + "." + type(self.model).__name__
            ),
            "model_path": str(self.model_path),
            "model_sha256": self.model_sha256,
            "model_config_path": str(model_config_path),
            "model_config_sha256": model_config_sha256,
            "feature_schema_path": str(self.feature_schema_path),
            "feature_schema_sha256": self.feature_schema_sha256,
            "label_mapping_path": str(self.label_mapping_path),
            "label_mapping_sha256": self.label_mapping_sha256,
            "primary_classes_path": str(self.primary_classes_path),
            "primary_classes_sha256": self.primary_classes_sha256,
            "expected_feature_count": EXPECTED_FEATURE_COUNT,
            "model_class_count": EXPECTED_CLASS_COUNT,
        }

    def _error_result(
        self,
        status: str,
        code: str,
        message: str,
        diagnostics: Mapping[str, Any] | None = None,
        warnings: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "status": status,
            "predicted_sign": None,
            "original_label_id": None,
            "model_class_index": None,
            "raw_confidence": None,
            "raw_probabilities": None,
            "recording_diagnostics": dict(diagnostics or {}),
            "model": self.identity,
            "confidence_note": CONFIDENCE_NOTE,
            "warnings": [str(item) for item in (warnings or [])],
            "errors": [{"code": code, "message": message}],
        }


def _default_model_loader(path: Path) -> Any:
    import joblib

    return joblib.load(path)


def _sha256_file(path: Path, label: str) -> str:
    if not path.is_file():
        raise RFServiceConfigurationError(f"{label} not found: {path}")
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RFServiceConfigurationError(f"cannot read {label} at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RFServiceConfigurationError(f"{label} must contain a JSON object")
    return payload
