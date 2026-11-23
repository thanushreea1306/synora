from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

try:
    from ml.src import prepare_video_features as feature_source
    from ml.src import train_baselines_v1 as v1
    from ml.src.experiments.rf_v2_720p30 import build_features_v2
    from ml.src.experiments.rf_v2_720p30 import capture as v2_capture
    from ml.src.experiments.rf_v2_720p30 import dataset as v2_dataset
    from ml.src.experiments.rf_v3_generalisation import calibration, cv, model
except ModuleNotFoundError as exc:
    if exc.name not in ("ml", "ml.src"):
        raise
    SOURCE_ROOT = str(Path(__file__).resolve().parents[2])
    if SOURCE_ROOT not in sys.path:
        sys.path.insert(0, SOURCE_ROOT)
    import prepare_video_features as feature_source
    import train_baselines_v1 as v1
    from experiments.rf_v2_720p30 import build_features_v2
    from experiments.rf_v2_720p30 import capture as v2_capture
    from experiments.rf_v2_720p30 import dataset as v2_dataset
    from experiments.rf_v3_generalisation import calibration, cv, model


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_FEATURES_ROOT = REPO_ROOT / "ml/dataset/processed/experiments/rf_v2_720p30/features"
DEFAULT_EXTRACTION_ROOT = REPO_ROOT / "ml/dataset/processed/experiments/rf_v2_720p30"
DEFAULT_MODEL_ROOT = REPO_ROOT / "ml/models/experiments/rf_v3_generalisation"
DEFAULT_REPORT_ROOT = REPO_ROOT / "ml/reports/experiments/rf_v3_generalisation"
DEFAULT_LABEL_MAPPING_PATH = REPO_ROOT / "ml/reports/synora_baseline_v1_20261121/experiment_config.json"
SERVICE_VERSION = "synora_rf_v3_service_v1"
EXPERIMENT_ID = cv.EXPERIMENT_ID
STATUS_OK = "ok"
STATUS_INVALID_RECORDING = "invalid_recording"
STATUS_FEATURE_ERROR = "feature_error"
STATUS_MODEL_ERROR = "model_error"
UNKNOWN_SIGN = "unknown"
CLASS_COUNT = model.CLASS_COUNT
MIN_FRAMES = v2_dataset.MIN_FRAMES
MAX_FRAMES = v2_dataset.MAX_FRAMES
CONFIDENCE_NOTE = (
    "calibrated confidence uses temperature scaling fitted on out-of-fold train/val predictions; "
    "no real-world accuracy claim is made"
)
LIMITATION_NOTE = (
    "Validation is clip-disjoint CV over train+val only; signer identity is unavailable and this "
    "is not a signer-independent evaluation"
)


class V3ServiceConfigurationError(ValueError):
    pass


class V3RecordingError(ValueError):
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


class V3RecognitionService:
    def __init__(
        self,
        features_root: str | Path | None = None,
        extraction_root: str | Path | None = None,
        model_root: str | Path | None = None,
        report_root: str | Path | None = None,
        label_mapping_path: str | Path | None = None,
        model_loader: Callable[[Path, dict[str, Any]], Any] | None = None,
    ) -> None:
        self.features_root = Path(features_root) if features_root is not None else DEFAULT_FEATURES_ROOT
        self.extraction_root = Path(extraction_root) if extraction_root is not None else DEFAULT_EXTRACTION_ROOT
        self.model_root = Path(model_root) if model_root is not None else DEFAULT_MODEL_ROOT
        self.report_root = Path(report_root) if report_root is not None else DEFAULT_REPORT_ROOT
        self.label_mapping_path = (
            Path(label_mapping_path) if label_mapping_path is not None else DEFAULT_LABEL_MAPPING_PATH
        )
        self.model_loader = model_loader or self._default_model_loader

        self.extraction_config = v2_dataset.read_json(self.extraction_root / "config.json")
        if self.extraction_config.get("split_policy", {}).get("test_split_read") is not False:
            raise V3ServiceConfigurationError("extraction config does not certify no test-split access")
        self.selection = v2_dataset.read_json(self.report_root / "selection.json")
        self.calibration_report = v2_dataset.read_json(self.report_root / "calibration.json")
        self.training_manifest = v2_dataset.read_json(self.report_root / "training_manifest.json")
        self.schema = v2_dataset.read_json(self.features_root / "rf_feature_schema.json")
        self.label_config = v2_dataset.read_json(self.label_mapping_path)
        self.selected_variant = str(self.selection["selected_variant"])
        if self.selected_variant not in cv.VARIANTS:
            raise V3ServiceConfigurationError(f"unsupported selected variant: {self.selected_variant}")
        if self.selection.get("selection_uses_test") is not False:
            raise V3ServiceConfigurationError("model selection must not use the test split")
        self.threshold = float(self.calibration_report["threshold"]["chosen"]["threshold"])
        self.temperature = float(self.calibration_report["temperature"]["temperature"])
        if not 0.0 <= self.threshold <= 1.0:
            raise V3ServiceConfigurationError("invalid rejection threshold")
        if not np.isfinite(self.temperature) or self.temperature <= 0.0:
            raise V3ServiceConfigurationError("invalid calibration temperature")

        self._verify_hashes()
        self.feature_names = [
            str(item) for item in self.schema["feature_columns"]
        ]
        if len(self.feature_names) != 831:
            raise V3ServiceConfigurationError("RF feature schema mismatch")
        self.label_mapping = self._load_label_mapping()
        with np.load(self.features_root / "mlp_preprocessing.npz") as preprocessing:
            self.mean = preprocessing["mean"].copy()
            self.std = preprocessing["std"].copy()
        self.channel_names = v2_dataset.read_json(self.features_root / "mlp_channel_names.json")
        if self.channel_names != v1.TemporalMLP.expected_channels():
            raise V3ServiceConfigurationError("MLP channel order mismatch")
        self.model = self.model_loader(self._model_path(), self._model_config())
        self.model_sha256 = v2_dataset.sha256_file(self._model_path())
        self.identity = {
            "service_version": SERVICE_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "selected_variant": self.selected_variant,
            "model_sha256": self.model_sha256,
            "validation_protocol": self.training_manifest["cv_protocol"],
            "validation_macro_f1": float(self.selection["selected_metrics"]["macro_f1"]),
            "rejection_threshold": self.threshold,
            "calibration_temperature": self.temperature,
            "confidence_note": CONFIDENCE_NOTE,
            "limitation_note": LIMITATION_NOTE,
        }

    def predict(self, records: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return self._predict_records(records, metadata, {})

    def predict_capture(self, capture_result: Any) -> dict[str, Any]:
        diagnostics = {
            "capture_status": getattr(capture_result, "status", None),
            "capture_reason": getattr(capture_result, "reason", None),
            "capture_frame_count": getattr(capture_result, "frame_count", None),
        }
        warnings = [str(item) for item in getattr(capture_result, "warnings", []) or []]
        if getattr(capture_result, "status", None) != v2_capture.STATUS_OK:
            return self._error_result(
                STATUS_INVALID_RECORDING,
                "CAPTURE_NOT_OK",
                "capture result is not a successful recording",
                diagnostics=diagnostics,
                warnings=warnings,
            )
        return self._predict_records(
            getattr(capture_result, "records", []),
            getattr(capture_result, "metadata", None),
            diagnostics,
            warnings,
        )

    def _predict_records(
        self,
        records: Sequence[Mapping[str, Any]],
        metadata: Mapping[str, Any] | None,
        diagnostics: Mapping[str, Any],
        warnings: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        frame_count = len(records)
        duration_diagnostics = dict(diagnostics)
        duration_diagnostics["record_frame_count"] = frame_count
        duration_diagnostics["minimum_frame_count"] = MIN_FRAMES
        duration_diagnostics["maximum_frame_count"] = MAX_FRAMES
        if frame_count < MIN_FRAMES:
            return self._error_result(
                STATUS_INVALID_RECORDING,
                "INSUFFICIENT_DURATION",
                f"recording has {frame_count} frames; minimum is {MIN_FRAMES}",
                diagnostics=duration_diagnostics,
                warnings=warnings,
            )
        if frame_count > MAX_FRAMES:
            return self._error_result(
                STATUS_INVALID_RECORDING,
                "EXCESSIVE_DURATION",
                f"recording has {frame_count} frames; maximum is {MAX_FRAMES}",
                diagnostics=duration_diagnostics,
                warnings=warnings,
            )
        try:
            raw_probabilities, calibrated_probabilities = self._probabilities(records, metadata)
        except V3RecordingError as exc:
            return self._error_result(
                STATUS_INVALID_RECORDING,
                "INVALID_RECORDING",
                f"{type(exc).__name__}: {exc}",
                diagnostics=duration_diagnostics,
                warnings=warnings,
            )
        except Exception as exc:
            return self._error_result(
                STATUS_FEATURE_ERROR,
                "FEATURE_BUILD_FAILED",
                f"{type(exc).__name__}: {exc}",
                diagnostics=duration_diagnostics,
                warnings=warnings,
            )
        try:
            raw_probabilities = self._validate_probabilities(raw_probabilities)
            calibrated_probabilities = self._validate_probabilities(calibrated_probabilities)
            candidate_index = int(np.argmax(calibrated_probabilities))
            calibrated_confidence = float(calibrated_probabilities[candidate_index])
            raw_confidence = float(raw_probabilities[candidate_index])
            mapping = self.label_mapping[candidate_index]
        except Exception as exc:
            return self._error_result(
                STATUS_MODEL_ERROR,
                "MODEL_OUTPUT_INVALID",
                f"{type(exc).__name__}: {exc}",
                diagnostics=duration_diagnostics,
                warnings=warnings,
            )
        accepted = calibrated_confidence >= self.threshold
        result_diagnostics = dict(duration_diagnostics)
        result = {
            "status": STATUS_OK,
            "accepted": accepted,
            "predicted_sign": mapping.class_label if accepted else UNKNOWN_SIGN,
            "original_label_id": mapping.original_label_id if accepted else None,
            "model_class_index": candidate_index if accepted else None,
            "candidate_sign": mapping.class_label,
            "candidate_model_class_index": candidate_index,
            "raw_confidence": raw_confidence,
            "calibrated_confidence": calibrated_confidence,
            "rejection_threshold": self.threshold,
            "calibration_temperature": self.temperature,
            "raw_probabilities": {
                str(item.model_index): float(raw_probabilities[item.model_index]) for item in self.label_mapping
            },
            "calibrated_probabilities": {
                str(item.model_index): float(calibrated_probabilities[item.model_index])
                for item in self.label_mapping
            },
            "recording_diagnostics": result_diagnostics,
            "model": self.identity,
            "confidence_note": CONFIDENCE_NOTE,
            "limitation_note": LIMITATION_NOTE,
            "warnings": [str(item) for item in (warnings or [])],
            "errors": [],
        }
        if not accepted:
            result["warnings"].append(
                f"LOW_CONFIDENCE_REJECTED: {calibrated_confidence:.6f} < {self.threshold:.6f}"
            )
        return result

    def _logits(self, records: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any] | None) -> np.ndarray:
        if metadata is None:
            metadata = {}
        fps = float(metadata.get("fps", 30.0))
        width = int(float(metadata.get("width", 1280)))
        height = int(float(metadata.get("height", 720)))
        if fps != 30.0 or width != 1280 or height != 720:
            raise V3RecordingError("live recording must be 1280x720 at 30 FPS")
        frame_count = len(records)
        row = {
            "video_id": "live_capture_v3_720p30",
            "split": "live",
            "official_rel_path_posix": None,
            "source_video_abs": None,
            "output_file": None,
            "class_folder": None,
            "class_label": None,
            "label_id": -1,
            "duration_s": float(metadata.get("duration_s", frame_count / 30.0)),
            "source_duration_s": float(metadata.get("duration_s", frame_count / 30.0)),
        }
        sample = build_features_v2.build_sample_from_records(row, list(records))
        tensor = feature_source.build_mlp_tensor(sample, self.mean, self.std, self.channel_names)
        prepared, real = self.model.prepare_inputs(tensor[None, :, :])
        return np.asarray(self.model.predict_logits(prepared, real), dtype=np.float64)

    def _probabilities(
        self, records: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any] | None
    ) -> tuple[np.ndarray, np.ndarray]:
        logits = self._logits(records, metadata)
        raw = calibration.softmax(logits, 1.0)
        calibrated = calibration.softmax(logits, self.temperature)
        return raw, calibrated

    def _default_model_loader(self, path: Path, config: dict[str, Any]) -> Any:
        result = model.RegularizedTemporalMLP(config)
        with np.load(path) as weights:
            result.weights = {key: weights[key].copy() for key in weights.files}
        return result

    def _model_config_path(self) -> Path:
        return self.model_root / "temporal_mlp" / "config.json"

    def _model_config(self) -> dict[str, Any]:
        return v2_dataset.read_json(self._model_config_path())

    def _model_path(self) -> Path:
        return self.model_root / "temporal_mlp" / "weights.npz"

    def _load_label_mapping(self) -> list[LabelMapping]:
        mapping = []
        for record in self.label_config["label_mapping"]:
            mapping.append(
                LabelMapping(
                    model_index=int(record["model_index"]),
                    original_label_id=int(record["original_label_id"]),
                    class_label=str(record["class_label"]),
                    class_folder=str(record["class_folder"]),
                    folder_id=int(record["folder_id"]),
                )
            )
        if len(mapping) != CLASS_COUNT or {item.model_index for item in mapping} != set(range(CLASS_COUNT)):
            raise V3ServiceConfigurationError("invalid label mapping")
        return mapping

    def _verify_hashes(self) -> None:
        expected_inputs = self.training_manifest["input_hashes"]
        checked_inputs = {
            "features/rf_features.npy": self.features_root / "rf_features.npy",
            "features/mlp_tensor.npy": self.features_root / "mlp_tensor.npy",
            "features/mlp_preprocessing.npz": self.features_root / "mlp_preprocessing.npz",
            "features/rf_feature_schema.json": self.features_root / "rf_feature_schema.json",
            "features/video_manifest.csv": self.features_root / "video_manifest.csv",
            "extraction/config.json": self.extraction_root / "config.json",
        }
        for name, path in checked_inputs.items():
            if v2_dataset.sha256_file(path) != expected_inputs[name]:
                raise V3ServiceConfigurationError(f"hash mismatch for {name}")
        if v2_dataset.sha256_file(self.label_mapping_path) != expected_inputs["v1_experiment_config"]:
            raise V3ServiceConfigurationError("hash mismatch for v1 experiment config")
        expected_outputs = self.training_manifest["output_hashes"]
        checked_outputs = {
            "temporal_mlp/weights.npz": self._model_path(),
            "temporal_mlp/config.json": self._model_config_path(),
            "calibration.json": self.report_root / "calibration.json",
            "selection.json": self.report_root / "selection.json",
        }
        for name, path in checked_outputs.items():
            if v2_dataset.sha256_file(path) != expected_outputs[name]:
                raise V3ServiceConfigurationError(f"hash mismatch for {name}")

    @staticmethod
    def _validate_probabilities(proba: np.ndarray) -> np.ndarray:
        values = np.asarray(proba, dtype=np.float64)
        if values.shape != (1, CLASS_COUNT):
            raise ValueError(f"unexpected probability shape: {values.shape}")
        if not np.isfinite(values).all() or (values < 0.0).any():
            raise ValueError("invalid probabilities")
        total = float(values.sum())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"probability total is {total:.9f}")
        return values.reshape(CLASS_COUNT)

    @staticmethod
    def _error_result(
        status: str,
        code: str,
        message: str,
        diagnostics: Mapping[str, Any] | None = None,
        warnings: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "status": status,
            "accepted": False,
            "predicted_sign": UNKNOWN_SIGN,
            "raw_confidence": None,
            "calibrated_confidence": None,
            "rejection_threshold": None,
            "recording_diagnostics": dict(diagnostics or {}),
            "warnings": [str(item) for item in (warnings or [])],
            "errors": [{"code": code, "message": message}],
            "confidence_note": CONFIDENCE_NOTE,
            "limitation_note": LIMITATION_NOTE,
        }
