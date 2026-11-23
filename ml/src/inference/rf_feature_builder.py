from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    from ml.src import prepare_video_features as feature_source
except ModuleNotFoundError as exc:
    if exc.name not in ("ml", "ml.src"):
        raise
    SOURCE_ROOT = str(Path(__file__).resolve().parents[1])
    if SOURCE_ROOT not in sys.path:
        sys.path.insert(0, SOURCE_ROOT)
    import prepare_video_features as feature_source


EXPECTED_FEATURE_COUNT = 831
DEFAULT_RF_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "dataset"
    / "processed"
    / "synora_include21_video_features_v1"
    / "rf_feature_schema.json"
)
REQUIRED_METADATA_FIELDS = ("duration_s", "fps", "width", "height")


class FeatureBuildError(ValueError):
    pass


@dataclass(frozen=True)
class RFFeatureRow:
    feature_names: tuple[str, ...]
    values: np.ndarray
    diagnostics: dict[str, Any]


@lru_cache(maxsize=None)
def load_authoritative_feature_names(path: str | Path | None = None) -> tuple[str, ...]:
    schema_path = DEFAULT_RF_SCHEMA_PATH if path is None else Path(path)
    if not schema_path.is_file():
        raise FeatureBuildError(f"RF feature schema not found: {schema_path}")
    payload = json.loads(schema_path.read_text(encoding="utf-8"))
    names = tuple(payload.get("feature_columns", ()))
    if len(names) != EXPECTED_FEATURE_COUNT:
        raise FeatureBuildError(
            f"schema has {len(names)} feature columns; expected {EXPECTED_FEATURE_COUNT}"
        )
    if len(set(names)) != len(names):
        raise FeatureBuildError("schema contains duplicate feature names")
    if payload.get("scaling") != "none":
        raise FeatureBuildError("schema scaling must be 'none'")
    source_names = tuple(feature_source.base_feature_names())
    if names != source_names:
        first_mismatch = next(
            (index for index, pair in enumerate(zip(names, source_names)) if pair[0] != pair[1]),
            min(len(names), len(source_names)),
        )
        raise FeatureBuildError(
            "schema order does not match prepare_video_features.base_feature_names() "
            f"at index {first_mismatch}: schema={names[first_mismatch] if first_mismatch < len(names) else '<missing>'}, "
            f"source={source_names[first_mismatch] if first_mismatch < len(source_names) else '<missing>'}"
        )
    return names


def build_rf_feature_vector(
    records: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any] | None = None,
    expected_feature_names: Sequence[str] | None = None,
) -> np.ndarray:
    return build_rf_feature_row(records, metadata, expected_feature_names).values


def build_rf_feature_row(
    records: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any] | None = None,
    expected_feature_names: Sequence[str] | None = None,
) -> RFFeatureRow:
    ordered_records = _validate_records(records)
    frame_count = len(ordered_records)
    resolved_metadata = _resolve_metadata(ordered_records, metadata)
    names = (
        tuple(expected_feature_names)
        if expected_feature_names is not None
        else load_authoritative_feature_names()
    )
    _validate_feature_names(names)

    coords = np.zeros(
        (feature_source.T_VALUE, 2, feature_source.LANDMARK_COUNT, 3),
        dtype=feature_source.RF_DTYPE,
    )
    presence = np.zeros((feature_source.T_VALUE, 2), dtype=bool)
    confidence = np.zeros((feature_source.T_VALUE, 2), dtype=feature_source.RF_DTYPE)
    frame_real = np.zeros(feature_source.T_VALUE, dtype=bool)
    no_hand = np.zeros(feature_source.T_VALUE, dtype=bool)
    padded = np.zeros(feature_source.T_VALUE, dtype=bool)
    duplicate_count = 0

    for frame_index, record in enumerate(ordered_records):
        try:
            frame_coords, frame_presence, frame_confidence, frame_no_hand, frame_duplicates = (
                feature_source.canonical_frame(record)
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise FeatureBuildError(f"invalid frame at frame_index={frame_index}: {exc}") from exc
        coords[frame_index] = frame_coords
        presence[frame_index] = frame_presence
        confidence[frame_index] = frame_confidence
        no_hand[frame_index] = frame_no_hand
        frame_real[frame_index] = True
        duplicate_count += int(frame_duplicates)

    for frame_index in range(frame_count, feature_source.T_VALUE):
        padded[frame_index] = True

    sample = {
        "meta": resolved_metadata,
        "coords": coords,
        "presence": presence,
        "confidence": confidence,
        "frame_real": frame_real,
        "no_hand": no_hand,
        "padded": padded,
        "frame_count": frame_count,
        "duplicate_count": duplicate_count,
    }
    features = feature_source.compute_rf_features(sample)
    if tuple(features.keys()) != names:
        raise FeatureBuildError("computed feature order does not match expected feature names")

    values = np.asarray(
        [float(features[name]) for name in names],
        dtype=feature_source.RF_DTYPE,
    ).reshape(1, EXPECTED_FEATURE_COUNT)
    if values.shape != (1, EXPECTED_FEATURE_COUNT):
        raise FeatureBuildError(f"unexpected feature shape: {values.shape}")
    if not bool(np.isfinite(values).all()):
        bad_indices = np.where(~np.isfinite(values))[0].tolist()
        bad_names = [names[index] for index in bad_indices[:5]]
        raise FeatureBuildError(f"non-finite feature values: {bad_names}")

    both = presence[:, 0] & presence[:, 1] & frame_real
    diagnostics = {
        "frame_count": frame_count,
        "duration_s": resolved_metadata["duration_s"],
        "fps": resolved_metadata["fps"],
        "width": resolved_metadata["width"],
        "height": resolved_metadata["height"],
        "padded_frame_count": int(padded.sum()),
        "padded_frame_ratio": float(padded.sum() / feature_source.T_VALUE),
        "no_hand_frame_count": int((no_hand & frame_real).sum()),
        "both_hands_frame_count": int(both.sum()),
        "duplicate_handedness_frames": int(duplicate_count),
        "left_observed_frames": int((presence[:, 0] & frame_real).sum()),
        "right_observed_frames": int((presence[:, 1] & frame_real).sum()),
    }
    return RFFeatureRow(feature_names=names, values=values, diagnostics=diagnostics)


def _validate_records(records: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    if records is None:
        raise FeatureBuildError("records must not be None")
    if isinstance(records, (str, bytes)):
        raise FeatureBuildError("records must be a sequence of frame records, not a path string")
    try:
        ordered = list(records)
    except TypeError as exc:
        raise FeatureBuildError("records must be iterable") from exc
    if not ordered:
        raise FeatureBuildError("records must contain at least one frame")
    if len(ordered) > feature_source.T_VALUE:
        raise FeatureBuildError(
            f"record has {len(ordered)} frames; maximum supported is {feature_source.T_VALUE}"
        )
    if not all(isinstance(record, Mapping) for record in ordered):
        raise FeatureBuildError("every record must be a mapping")

    try:
        ordered.sort(key=lambda item: int(item["frame_index"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise FeatureBuildError("every record must have an integer frame_index") from exc

    for expected_index, record in enumerate(ordered):
        actual_index = int(record["frame_index"])
        if actual_index != expected_index:
            raise FeatureBuildError(
                f"non-sequential frame_index at position {expected_index}: {actual_index}"
            )
        schema_version = record.get("schema_version")
        if schema_version != feature_source.FRAME_SCHEMA_VERSION:
            raise FeatureBuildError(
                f"invalid schema_version at frame_index={expected_index}: {schema_version!r}"
            )
        if record.get("decode_ok") is not True:
            raise FeatureBuildError(f"decode_ok is not true at frame_index={expected_index}")
        if record.get("infer_ok") is not True:
            raise FeatureBuildError(f"infer_ok is not true at frame_index={expected_index}")
    return ordered


def _resolve_metadata(
    records: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any] | None,
) -> dict[str, float]:
    if metadata is None:
        first = records[0]
        fps = _finite_float(first.get("fps"), "fps", allow_none=False)
        width = _finite_float(first.get("width"), "width", allow_none=False)
        height = _finite_float(first.get("height"), "height", allow_none=False)
        duration_s = len(records) / fps
    else:
        if not isinstance(metadata, Mapping):
            raise FeatureBuildError("metadata must be a mapping or None")
        missing = [field for field in REQUIRED_METADATA_FIELDS if field not in metadata]
        if missing:
            raise FeatureBuildError(f"metadata missing required fields: {missing}")
        duration_s = _finite_float(metadata["duration_s"], "duration_s", allow_none=False)
        fps = _finite_float(metadata["fps"], "fps", allow_none=False)
        width = _finite_float(metadata["width"], "width", allow_none=False)
        height = _finite_float(metadata["height"], "height", allow_none=False)

    if fps <= 0.0:
        raise FeatureBuildError("fps must be positive")
    if width <= 0.0 or height <= 0.0:
        raise FeatureBuildError("width and height must be positive")
    if not math.isfinite(duration_s):
        raise FeatureBuildError("duration_s must be finite")
    return {
        "duration_s": duration_s,
        "fps": fps,
        "width": width,
        "height": height,
    }


def _finite_float(value: Any, name: str, allow_none: bool) -> float:
    if value is None and allow_none:
        return 0.0
    if value is None:
        raise FeatureBuildError(f"{name} is missing")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise FeatureBuildError(f"{name} must be numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise FeatureBuildError(f"{name} must be finite: {value!r}")
    return number


def _validate_feature_names(names: tuple[str, ...]) -> None:
    if len(names) != EXPECTED_FEATURE_COUNT:
        raise FeatureBuildError(
            f"expected {EXPECTED_FEATURE_COUNT} feature names; got {len(names)}"
        )
    if len(set(names)) != len(names):
        raise FeatureBuildError("expected feature names contain duplicates")
