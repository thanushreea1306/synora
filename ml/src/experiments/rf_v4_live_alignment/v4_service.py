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
    from ml.src.experiments.rf_v2_720p30 import dataset as v2_dataset
    from ml.src.experiments.rf_v3_generalisation import calibration, model as v3_model
    from ml.src.experiments.rf_v4_live_alignment import capture as v4_capture
    from ml.src.experiments.rf_v4_live_alignment import features as v4_features
    from ml.src.experiments.rf_v4_live_alignment import models as v4_models
    from ml.src.experiments.rf_v4_live_alignment import preprocessing
except ModuleNotFoundError as exc:
    if exc.name not in ("ml", "ml.src"):
        raise
    SOURCE_ROOT = str(Path(__file__).resolve().parents[2])
    if SOURCE_ROOT not in sys.path:
        sys.path.insert(0, SOURCE_ROOT)
    import prepare_video_features as feature_source
    import train_baselines_v1 as v1
    from experiments.rf_v2_720p30 import build_features_v2
    from experiments.rf_v2_720p30 import dataset as v2_dataset
    from experiments.rf_v3_generalisation import calibration, model as v3_model
    from experiments.rf_v4_live_alignment import capture as v4_capture
    from experiments.rf_v4_live_alignment import features as v4_features
    from experiments.rf_v4_live_alignment import models as v4_models
    from experiments.rf_v4_live_alignment import preprocessing


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_FEATURES_ROOT = REPO_ROOT / "ml/dataset/processed/experiments/rf_v4_live_alignment/features"
DEFAULT_EXTRACTION_ROOT = REPO_ROOT / "ml/dataset/processed/experiments/rf_v4_live_alignment"
DEFAULT_V2_FEATURES_ROOT = REPO_ROOT / "ml/dataset/processed/experiments/rf_v2_720p30/features"
DEFAULT_MODEL_ROOT = REPO_ROOT / "ml/models/experiments/rf_v4_live_alignment"
DEFAULT_REPORT_ROOT = REPO_ROOT / "ml/reports/experiments/rf_v4_live_alignment"
DEFAULT_V1_CONFIG = REPO_ROOT / "ml/reports/synora_baseline_v1_20261121/experiment_config.json"
SERVICE_VERSION = "synora_rf_v4_service_v1"
EXPERIMENT_ID = "synora_rf_v4_live_alignment_20261123"
STATUS_OK = "ok"
STATUS_INVALID_RECORDING = "invalid_recording"
STATUS_FEATURE_ERROR = "feature_error"
STATUS_MODEL_ERROR = "model_error"
UNKNOWN_SIGN = "unknown"
CLASS_COUNT = v3_model.CLASS_COUNT
MIN_FRAMES = preprocessing.MIN_FRAMES
MAX_FRAMES = preprocessing.MAX_FRAMES
ALLOWED_VARIANTS = {
    "v3_regularized_cv",
    "v4_base_mlp_cv",
    "v4_enhanced_tcn_cv",
    "v4_enhanced_attention_cv",
}
CONFIDENCE_NOTE = (
    "calibrated confidence uses temperature scaling fitted on out-of-fold predictions from the "
    "333 common train+val clips; it is not a real-world accuracy claim"
)
LIMITATION_NOTE = (
    "V4 capture and offline extraction share resize, color, timestamp, and tracking-reset logic; "
    "validation is clip-disjoint CV over train+val only, signer identity is unavailable, and no "
    "live-webcam accuracy improvement is claimed without live evaluation"
)


class V4ServiceConfigurationError(ValueError):
    pass


class V4RecordingError(ValueError):
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


class V4RecognitionService:
    def __init__(
        self,
        features_root: str | Path | None = None,
        extraction_root: str | Path | None = None,
        model_root: str | Path | None = None,
        report_root: str | Path | None = None,
        v2_features_root: str | Path | None = None,
        label_mapping_path: str | Path | None = None,
        model_loader: Callable[[Path, dict[str, Any]], Any] | None = None,
    ) -> None:
        self.features_root = Path(features_root) if features_root is not None else DEFAULT_FEATURES_ROOT
        self.extraction_root = Path(extraction_root) if extraction_root is not None else DEFAULT_EXTRACTION_ROOT
        self.model_root = Path(model_root) if model_root is not None else DEFAULT_MODEL_ROOT
        self.report_root = Path(report_root) if report_root is not None else DEFAULT_REPORT_ROOT
        self.v2_features_root = (
            Path(v2_features_root) if v2_features_root is not None else DEFAULT_V2_FEATURES_ROOT
        )
        self.label_mapping_path = (
            Path(label_mapping_path) if label_mapping_path is not None else DEFAULT_V1_CONFIG
        )
        self.model_loader = model_loader or self._default_model_loader
        self.extraction_config = v2_dataset.read_json(self.extraction_root / "config.json")
        if self.extraction_config.get("split_policy", {}).get("test_split_read") is not False:
            raise V4ServiceConfigurationError("extraction config does not certify no test-split access")
        self.selection = v2_dataset.read_json(self.report_root / "selection.json")
        self.calibration_report = v2_dataset.read_json(self.report_root / "calibration.json")
        self.training_manifest = v2_dataset.read_json(self.report_root / "training_manifest.json")
        self.schema = v2_dataset.read_json(self.features_root / "feature_schema.json")
        self.label_config = v2_dataset.read_json(self.label_mapping_path)
        self.model_config = self._model_config()
        self.selected_variant = str(self.selection["selected_variant"])
        if self.selected_variant not in ALLOWED_VARIANTS:
            raise V4ServiceConfigurationError(f"unsupported selected variant: {self.selected_variant}")
        if self.selection.get("selection_uses_test") is not False:
            raise V4ServiceConfigurationError("model selection must not use the test split")
        if self.training_manifest.get("test_split_read") is not False:
            raise V4ServiceConfigurationError("training manifest does not certify no test-split access")
        self.kind = str(self.model_config["kind"])
        self.feature_mode = str(self.model_config["feature_mode"])
        if self.kind not in {"v3", "v4"} or self.feature_mode not in {"base", "enhanced"}:
            raise V4ServiceConfigurationError("invalid selected model configuration")
        if self.kind == "v3" and self.feature_mode != "base":
            raise V4ServiceConfigurationError("v3 model requires base features")
        self.threshold = float(self.calibration_report["threshold"]["chosen"]["threshold"])
        self.temperature = float(self.calibration_report["temperature"]["temperature"])
        if not 0.0 <= self.threshold <= 1.0:
            raise V4ServiceConfigurationError("invalid rejection threshold")
        if not np.isfinite(self.temperature) or self.temperature <= 0.0:
            raise V4ServiceConfigurationError("invalid calibration temperature")
        self._verify_hashes()
        self.label_mapping = self._load_label_mapping()
        if self.feature_mode == "base":
            with np.load(self.v2_features_root / "mlp_preprocessing.npz") as preprocessing_array:
                self.mean = preprocessing_array["mean"].copy()
                self.std = preprocessing_array["std"].copy()
            self.channel_names = v2_dataset.read_json(self.v2_features_root / "mlp_channel_names.json")
            if self.channel_names != v1.TemporalMLP.expected_channels():
                raise V4ServiceConfigurationError("base channel order mismatch")
            self.base_preparer = v3_model.RegularizedTemporalMLP(
                v3_model.build_config("v4_service_base", label_smoothing=0.0, class_weight_power=0.0)
            )
        else:
            self.mean = None
            self.std = None
            self.channel_names = None
            self.base_preparer = None
            if self.schema["enhanced_channel_names"] != v4_features.ENHANCED_CHANNEL_NAMES:
                raise V4ServiceConfigurationError("enhanced channel order mismatch")
        self.model = self.model_loader(self._model_path(), self.model_config)
        self.model_sha256 = v2_dataset.sha256_file(self._model_path())
        self.identity = {
            "service_version": SERVICE_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "selected_variant": self.selected_variant,
            "model_kind": self.kind,
            "feature_mode": self.feature_mode,
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
        if getattr(capture_result, "status", None) != v4_capture.STATUS_OK:
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
        except V4RecordingError as exc:
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
            "recording_diagnostics": dict(duration_diagnostics),
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
        fps = float(metadata.get("fps", preprocessing.TARGET_FPS))
        width = int(float(metadata.get("width", preprocessing.TARGET_WIDTH)))
        height = int(float(metadata.get("height", preprocessing.TARGET_HEIGHT)))
        if fps != preprocessing.TARGET_FPS or width != preprocessing.TARGET_WIDTH or height != preprocessing.TARGET_HEIGHT:
            raise V4RecordingError("live recording must be 1280x720 at 30 FPS")
        frame_count = len(records)
        row = {
            "video_id": "live_capture_v4_720p30",
            "split": "live",
            "official_rel_path_posix": None,
            "source_video_abs": None,
            "output_file": None,
            "class_folder": None,
            "class_label": None,
            "label_id": -1,
            "duration_s": float(metadata.get("duration_s", frame_count / preprocessing.TARGET_FPS)),
            "source_duration_s": float(metadata.get("duration_s", frame_count / preprocessing.TARGET_FPS)),
        }
        sample = build_features_v2.build_sample_from_records(row, list(records))
        if self.feature_mode == "enhanced":
            tensor = v4_features.build_enhanced_tensor(sample)[None, :, :]
            prepared, real = v4_models.extract_inputs(tensor, "enhanced", None)
        else:
            tensor = v4_features.build_base_tensor(sample, self.mean, self.std, self.channel_names)[None, :, :]
            prepared, real = v4_models.extract_inputs(tensor, "base", self.base_preparer)
        if self.kind == "v3":
            prepared, real = self.model.prepare_inputs(tensor)
        return np.asarray(self.model.predict_logits(prepared, real), dtype=np.float64)

    def _probabilities(
        self, records: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any] | None
    ) -> tuple[np.ndarray, np.ndarray]:
        logits = self._logits(records, metadata)
        return calibration.softmax(logits, 1.0), calibration.softmax(logits, self.temperature)

    def _default_model_loader(self, path: Path, config: Mapping[str, Any]) -> Any:
        if str(config["kind"]) == "v3":
            result = v3_model.RegularizedTemporalMLP(dict(config))
        else:
            result = v4_models.TemporalNetwork(config)
        with np.load(path) as weights:
            result.weights = {key: weights[key].copy() for key in weights.files}
        return result

    def _model_config_path(self) -> Path:
        return self.model_root / "selected" / "config.json"

    def _model_config(self) -> dict[str, Any]:
        return v2_dataset.read_json(self._model_config_path())

    def _model_path(self) -> Path:
        return self.model_root / "selected" / "weights.npz"

    def _load_label_mapping(self) -> list[LabelMapping]:
        mapping = [
            LabelMapping(
                model_index=int(record["model_index"]),
                original_label_id=int(record["original_label_id"]),
                class_label=str(record["class_label"]),
                class_folder=str(record["class_folder"]),
                folder_id=int(record["folder_id"]),
            )
            for record in self.label_config["label_mapping"]
        ]
        if len(mapping) != CLASS_COUNT or {item.model_index for item in mapping} != set(range(CLASS_COUNT)):
            raise V4ServiceConfigurationError("invalid label mapping")
        return mapping

    def _verify_hashes(self) -> None:
        expected_inputs = self.training_manifest["input_hashes"]
        checked_inputs = {
            "v2_features/mlp_tensor.npy": self.v2_features_root / "mlp_tensor.npy",
            "v2_features/mlp_preprocessing.npz": self.v2_features_root / "mlp_preprocessing.npz",
            "v2_features/mlp_channel_names.json": self.v2_features_root / "mlp_channel_names.json",
            "v2_features/video_manifest.csv": self.v2_features_root / "video_manifest.csv",
            "v4_features/base_tensor.npy": self.features_root / "base_tensor.npy",
            "v4_features/enhanced_tensor.npy": self.features_root / "enhanced_tensor.npy",
            "v4_features/labels.npy": self.features_root / "labels.npy",
            "v4_features/feature_schema.json": self.features_root / "feature_schema.json",
            "v4_features/video_manifest.csv": self.features_root / "video_manifest.csv",
            "v4_features/feature_build_summary.json": self.features_root / "feature_build_summary.json",
            "v4_extraction/config.json": self.extraction_root / "config.json",
            "v4_extraction/manifest.csv": self.extraction_root / "manifest.csv",
        }
        for name, path in checked_inputs.items():
            if v2_dataset.sha256_file(path) != expected_inputs[name]:
                raise V4ServiceConfigurationError(f"hash mismatch for {name}")
        if v2_dataset.sha256_file(self.label_mapping_path) != expected_inputs["v1_experiment_config"]:
            raise V4ServiceConfigurationError("hash mismatch for v1 experiment config")
        expected_outputs = self.training_manifest["output_hashes"]
        checked_outputs = {
            "selected/weights.npz": self._model_path(),
            "selected/config.json": self._model_config_path(),
            "calibration.json": self.report_root / "calibration.json",
            "selection.json": self.report_root / "selection.json",
        }
        for name, path in checked_outputs.items():
            if v2_dataset.sha256_file(path) != expected_outputs[name]:
                raise V4ServiceConfigurationError(f"hash mismatch for {name}")

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
