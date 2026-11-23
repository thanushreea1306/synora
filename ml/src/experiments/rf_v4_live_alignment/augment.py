from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ml.src.experiments.rf_v4_live_alignment import features


ABS_START = 0
REL_START = 126
MOTION_ABS_START = 126 + 126 + 4 + 8 + 9 + 25
MOTION_SPEED_START = MOTION_ABS_START + 126
VALIDITY_START = len(features.ENHANCED_CHANNEL_NAMES) - 42
STATE_START = 126 + 126 + 4 + 8


def default_augmentation_config() -> dict[str, Any]:
    return {
        "coordinate_noise_std": 0.01,
        "scale_range": [0.98, 1.02],
        "translation_range": [-0.01, 0.01],
        "speed_range": [0.96, 1.04],
        "landmark_drop_probability": 0.02,
        "hand_drop_probability": 0.01,
        "mirror": False,
    }


def _time_warp(x: np.ndarray, real: np.ndarray, rng: np.random.Generator, settings: Mapping[str, Any]) -> None:
    low, high = (float(value) for value in settings["speed_range"])
    for sample_index in range(x.shape[0]):
        length = int(real[sample_index].sum())
        if length < 2:
            continue
        speed = float(rng.uniform(low, high))
        target_length = int(np.clip(round(length * speed), 2, x.shape[1]))
        source_frames = np.flatnonzero(real[sample_index])
        source_indices = np.linspace(0, length - 1, target_length).round().astype(np.int64)
        warped = np.zeros_like(x[sample_index])
        warped[:target_length] = x[sample_index][source_frames[source_indices]]
        x[sample_index] = warped
        real[sample_index] = np.arange(x.shape[1]) < target_length


def _apply_enhanced_dropout(x: np.ndarray, real: np.ndarray, rng: np.random.Generator, settings: Mapping[str, Any]) -> None:
    landmark_probability = float(settings["landmark_drop_probability"])
    hand_probability = float(settings["hand_drop_probability"])
    for sample_index in range(x.shape[0]):
        real_indices = np.flatnonzero(real[sample_index])
        if len(real_indices) == 0:
            continue
        for side_index in range(2):
            validity_offset = VALIDITY_START + side_index * 21
            valid = x[sample_index, real_indices, validity_offset : validity_offset + 21] > 0.5
            if rng.random() < hand_probability:
                valid = valid & True
                drop_all = True
            else:
                drop_all = False
            drop_mask = (rng.random(21) < landmark_probability) | (drop_all & valid)
            drop_mask &= valid
            if not drop_mask.any():
                continue
            abs_offset = ABS_START + side_index * 63
            rel_offset = REL_START + side_index * 63
            motion_offset = MOTION_ABS_START + side_index * 63
            speed_offset = MOTION_SPEED_START + side_index * 21
            for landmark_index in np.flatnonzero(drop_mask.any(axis=0)):
                affected_frames = real_indices[drop_mask[:, landmark_index]]
                coordinate_slice = slice(landmark_index * 3, landmark_index * 3 + 3)
                x[sample_index, affected_frames, abs_offset + coordinate_slice.start : abs_offset + coordinate_slice.stop] = 0.0
                x[sample_index, affected_frames, rel_offset + coordinate_slice.start : rel_offset + coordinate_slice.stop] = 0.0
                x[sample_index, affected_frames, motion_offset + coordinate_slice.start : motion_offset + coordinate_slice.stop] = 0.0
                x[sample_index, affected_frames, speed_offset + landmark_index] = 0.0
                x[sample_index, affected_frames, validity_offset + landmark_index] = 0.0


def augment_batch(
    x: np.ndarray,
    real: np.ndarray,
    config: Mapping[str, Any] | None,
    rng: np.random.Generator,
    *,
    feature_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    if not config:
        return x, real
    augmented_x = np.array(x, copy=True)
    augmented_real = np.array(real, copy=True)
    noise_std = float(config["coordinate_noise_std"])
    low, high = (float(value) for value in config["scale_range"])
    translate_low, translate_high = (float(value) for value in config["translation_range"])
    for sample_index in range(augmented_x.shape[0]):
        scale = float(rng.uniform(low, high))
        dx = float(rng.uniform(translate_low, translate_high))
        dy = float(rng.uniform(translate_low, translate_high))
        if feature_mode == "base":
            for side_index in range(2):
                presence = augmented_x[sample_index, :, 126 + side_index] > 0.5
                mask = presence & augmented_real[sample_index]
                if not mask.any():
                    continue
                offset = side_index * 63
                block = augmented_x[sample_index, mask, offset : offset + 63].reshape(-1, 21, 3)
                block[:, :, 0] = block[:, :, 0] * scale + dx
                block[:, :, 1] = block[:, :, 1] * scale + dy
                block += rng.normal(0.0, noise_std, size=block.shape).astype(np.float32)
                augmented_x[sample_index, mask, offset : offset + 63] = block.reshape(-1, 63).astype(np.float32)
        else:
            validity_start = VALIDITY_START
            for side_index in range(2):
                validity_offset = validity_start + side_index * 21
                valid = augmented_x[sample_index, augmented_real[sample_index], validity_offset : validity_offset + 21] > 0.5
                if not valid.any():
                    continue
                abs_offset = ABS_START + side_index * 63
                rel_offset = REL_START + side_index * 63
                motion_offset = MOTION_ABS_START + side_index * 63
                abs_block = augmented_x[sample_index, :, abs_offset : abs_offset + 63].reshape(-1, 21, 3)
                rel_block = augmented_x[sample_index, :, rel_offset : rel_offset + 63].reshape(-1, 21, 3)
                motion_block = augmented_x[sample_index, :, motion_offset : motion_offset + 63].reshape(-1, 21, 3)
                abs_block[:, :, 0] = (abs_block[:, :, 0] / features.ABS_SCALE + dx) * features.ABS_SCALE
                abs_block[:, :, 1] = (abs_block[:, :, 1] / features.ABS_SCALE + dy) * features.ABS_SCALE
                rel_block *= scale
                motion_block *= scale
                noise = rng.normal(0.0, noise_std, size=abs_block.shape).astype(np.float32)
                abs_block += noise
                rel_block += noise
                augmented_x[sample_index, :, abs_offset : abs_offset + 63] = abs_block.reshape(-1, 63)
                augmented_x[sample_index, :, rel_offset : rel_offset + 63] = rel_block.reshape(-1, 63)
                augmented_x[sample_index, :, motion_offset : motion_offset + 63] = motion_block.reshape(-1, 63)
    _time_warp(augmented_x, augmented_real, rng, config)
    if feature_mode == "enhanced":
        _apply_enhanced_dropout(augmented_x, augmented_real, rng, config)
    augmented_x[~augmented_real] = 0.0
    return augmented_x, augmented_real
