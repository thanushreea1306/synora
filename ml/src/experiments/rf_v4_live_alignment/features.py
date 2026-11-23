from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from ml.src import prepare_video_features as feature_source
except ModuleNotFoundError as exc:
    if exc.name not in ("ml", "ml.src"):
        raise
    SOURCE_ROOT = str(Path(__file__).resolve().parents[2])
    if SOURCE_ROOT not in sys.path:
        sys.path.insert(0, SOURCE_ROOT)
    import prepare_video_features as feature_source


HAND_LABELS = ("left", "right")
AXES = ("x", "y", "z")
LANDMARK_COUNT = feature_source.LANDMARK_COUNT
T_VALUE = feature_source.T_VALUE
ABS_SCALE = 2.0
REL_SCALE = 2.0
MOTION_SCALE = 8.0
SPEED_SCALE = 10.0
DISTANCE_SCALE = 2.0


def _clip(values: np.ndarray, low: float = -1.0, high: float = 1.0) -> np.ndarray:
    return np.clip(np.nan_to_num(values, nan=0.0, posinf=high, neginf=low), low, high).astype(np.float32)


def enhanced_channel_names() -> list[str]:
    names: list[str] = []
    for side in HAND_LABELS:
        for landmark_index in range(LANDMARK_COUNT):
            for axis in AXES:
                names.append(f"abs_{side}_lm{landmark_index:02d}_{axis}")
    for side in HAND_LABELS:
        for landmark_index in range(LANDMARK_COUNT):
            for axis in AXES:
                names.append(f"rel_{side}_lm{landmark_index:02d}_{axis}")
    for side in HAND_LABELS:
        names.append(f"{side}_palm_scale")
        names.append(f"{side}_bbox_scale")
    for side in HAND_LABELS:
        names.append(f"{side}_orient_wrist_middle_x")
        names.append(f"{side}_orient_wrist_middle_y")
        names.append(f"{side}_orient_index_pinky_x")
        names.append(f"{side}_orient_index_pinky_y")
    names.extend(
        [
            "left_present",
            "right_present",
            "left_confidence",
            "right_confidence",
            "left_missing_real",
            "right_missing_real",
            "frame_real",
            "no_hand",
            "padded",
        ]
    )
    for landmark_index in range(LANDMARK_COUNT):
        names.append(f"interhand_lm{landmark_index:02d}_distance")
    names.extend(["interhand_centroid_dx", "interhand_centroid_dy", "interhand_centroid_dz", "interhand_wrist_distance"])
    for side in HAND_LABELS:
        for landmark_index in range(LANDMARK_COUNT):
            for axis in AXES:
                names.append(f"motion_abs_{side}_lm{landmark_index:02d}_{axis}")
    for side in HAND_LABELS:
        for landmark_index in range(LANDMARK_COUNT):
            names.append(f"motion_speed_{side}_lm{landmark_index:02d}")
    names.append("interhand_motion_distance")
    for side in HAND_LABELS:
        for landmark_index in range(LANDMARK_COUNT):
            names.append(f"{side}_lm{landmark_index:02d}_valid")
    return names


ENHANCED_CHANNEL_NAMES = enhanced_channel_names()


def build_base_tensor(
    sample: Mapping[str, Any],
    mean: np.ndarray,
    std: np.ndarray,
    channel_names: Sequence[str],
) -> np.ndarray:
    return feature_source.build_mlp_tensor(sample, mean, std, channel_names)


def _hand_scale(coords: np.ndarray) -> tuple[float, float]:
    palm = float(np.linalg.norm(coords[9] - coords[0]))
    bbox = float(np.linalg.norm(coords.max(axis=0) - coords.min(axis=0)))
    return palm, bbox


def _hand_orientation(coords: np.ndarray) -> tuple[float, float, float, float]:
    wrist_to_middle = coords[9] - coords[0]
    index_to_pinky = coords[17] - coords[5]
    return (
        float(wrist_to_middle[0]),
        float(wrist_to_middle[1]),
        float(index_to_pinky[0]),
        float(index_to_pinky[1]),
    )


def build_enhanced_tensor(sample: Mapping[str, Any]) -> np.ndarray:
    coords = np.asarray(sample["coords"], dtype=np.float32)
    presence = np.asarray(sample["presence"], dtype=bool)
    confidence = np.asarray(sample["confidence"], dtype=np.float32)
    frame_real = np.asarray(sample["frame_real"], dtype=bool)
    no_hand = np.asarray(sample["no_hand"], dtype=bool)
    padded = np.asarray(sample["padded"], dtype=bool)
    if coords.shape != (T_VALUE, 2, LANDMARK_COUNT, 3):
        raise ValueError(f"unexpected coordinate shape: {coords.shape}")

    tensor = np.zeros((T_VALUE, len(ENHANCED_CHANNEL_NAMES)), dtype=np.float32)
    abs_start = 0
    rel_start = abs_start + 126
    scale_start = rel_start + 126
    orientation_start = scale_start + 4
    state_start = orientation_start + 8
    interhand_start = state_start + 9
    motion_abs_start = interhand_start + 25
    motion_speed_start = motion_abs_start + 126
    motion_distance_index = motion_speed_start + 42
    validity_start = motion_distance_index + 1

    previous_coords = np.zeros((2, LANDMARK_COUNT, 3), dtype=np.float32)
    previous_real = False
    previous_interhand = 0.0
    for frame_index in range(T_VALUE):
        real = bool(frame_real[frame_index])
        if not real:
            continue
        frame_coords = coords[frame_index]
        valid = presence[frame_index] & real
        for side_index in range(2):
            side_coords = frame_coords[side_index]
            side_valid = bool(valid[side_index])
            if side_valid:
                absolute = _clip((side_coords - 0.5) * ABS_SCALE)
                relative = _clip((side_coords - side_coords[0]) * REL_SCALE)
                palm_scale, bbox_scale = _hand_scale(side_coords)
                orientation = _hand_orientation(side_coords)
            else:
                absolute = np.zeros((LANDMARK_COUNT, 3), dtype=np.float32)
                relative = np.zeros((LANDMARK_COUNT, 3), dtype=np.float32)
                palm_scale = 0.0
                bbox_scale = 0.0
                orientation = (0.0, 0.0, 0.0, 0.0)
            abs_offset = abs_start + side_index * 63
            rel_offset = rel_start + side_index * 63
            tensor[frame_index, abs_offset : abs_offset + 63] = absolute.reshape(-1)
            tensor[frame_index, rel_offset : rel_offset + 63] = relative.reshape(-1)
            scale_offset = scale_start + side_index * 2
            tensor[frame_index, scale_offset] = _clip(np.asarray(palm_scale * DISTANCE_SCALE), 0.0, 1.0)
            tensor[frame_index, scale_offset + 1] = _clip(np.asarray(bbox_scale * DISTANCE_SCALE), 0.0, 1.0)
            orientation_offset = orientation_start + side_index * 4
            tensor[frame_index, orientation_offset : orientation_offset + 4] = _clip(
                np.asarray(orientation, dtype=np.float32) * REL_SCALE
            )
            motion_offset = motion_abs_start + side_index * 63
            speed_offset = motion_speed_start + side_index * LANDMARK_COUNT
            if side_valid and previous_real and presence[frame_index - 1, side_index]:
                delta = (side_coords - previous_coords[side_index]) * MOTION_SCALE
                tensor[frame_index, motion_offset : motion_offset + 63] = _clip(delta).reshape(-1)
                tensor[frame_index, speed_offset : speed_offset + LANDMARK_COUNT] = _clip(
                    np.linalg.norm(delta.reshape(LANDMARK_COUNT, 3), axis=1) * (SPEED_SCALE / MOTION_SCALE),
                    0.0,
                    1.0,
                )
            for landmark_index in range(LANDMARK_COUNT):
                tensor[frame_index, validity_start + side_index * LANDMARK_COUNT + landmark_index] = float(side_valid)

        tensor[frame_index, state_start] = float(presence[frame_index, 0])
        tensor[frame_index, state_start + 1] = float(presence[frame_index, 1])
        tensor[frame_index, state_start + 2] = float(confidence[frame_index, 0])
        tensor[frame_index, state_start + 3] = float(confidence[frame_index, 1])
        tensor[frame_index, state_start + 4] = float(not presence[frame_index, 0])
        tensor[frame_index, state_start + 5] = float(not presence[frame_index, 1])
        tensor[frame_index, state_start + 6] = 1.0
        tensor[frame_index, state_start + 7] = float(no_hand[frame_index])
        tensor[frame_index, state_start + 8] = float(padded[frame_index])

        if valid.all():
            distances = np.linalg.norm(frame_coords[0] - frame_coords[1], axis=1)
            tensor[frame_index, interhand_start : interhand_start + LANDMARK_COUNT] = _clip(
                distances * DISTANCE_SCALE, 0.0, 1.0
            )
            centroid_delta = frame_coords[0].mean(axis=0) - frame_coords[1].mean(axis=0)
            tensor[frame_index, interhand_start + 21 : interhand_start + 24] = _clip(centroid_delta * REL_SCALE)
            wrist_distance = float(np.linalg.norm(frame_coords[0, 0] - frame_coords[1, 0]))
            tensor[frame_index, interhand_start + 24] = _clip(np.asarray(wrist_distance * DISTANCE_SCALE), 0.0, 1.0)
            tensor[frame_index, motion_distance_index] = _clip(
                np.asarray((wrist_distance - previous_interhand) * MOTION_SCALE)
            )
            previous_interhand = wrist_distance
        previous_coords = frame_coords
        previous_real = True
    return tensor
