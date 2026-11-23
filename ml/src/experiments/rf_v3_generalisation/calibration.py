from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from ml.src.experiments.rf_v3_generalisation.model import CLASS_COUNT


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    scale = float(temperature)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("temperature must be finite and positive")
    shifted = values / scale
    shifted -= shifted.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def expected_calibration_error(y_true: Sequence[int], proba: np.ndarray, bins: int = 10) -> float:
    labels = np.asarray(y_true, dtype=np.int64)
    values = np.asarray(proba, dtype=np.float64)
    confidence = values.max(axis=1)
    predictions = values.argmax(axis=1)
    correct = (predictions == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        upper_mask = confidence < edges[index + 1]
        if index == bins - 1:
            upper_mask = upper_mask | (confidence <= 1.0)
        selected = (confidence >= edges[index]) & upper_mask
        if not selected.any():
            continue
        total += float(selected.mean() * abs(confidence[selected].mean() - correct[selected].mean()))
    return total


def negative_log_likelihood(y_true: Sequence[int], proba: np.ndarray) -> float:
    labels = np.asarray(y_true, dtype=np.int64)
    values = np.clip(np.asarray(proba, dtype=np.float64), 1e-12, 1.0)
    return float(-np.log(values[np.arange(len(labels)), labels]).mean())


def temperature_scale(
    y_true: Sequence[int],
    logits: np.ndarray,
    *,
    max_iterations: int = 300,
    learning_rate: float = 0.05,
    lower: float = 0.1,
    upper: float = 10.0,
) -> dict[str, Any]:
    labels = np.asarray(y_true, dtype=np.int64)
    values = np.asarray(logits, dtype=np.float64)
    if values.shape != (len(labels), CLASS_COUNT):
        raise ValueError(f"unexpected logit shape: {values.shape}")
    log_temperature = 0.0
    for _ in range(int(max_iterations)):
        temperature = math.exp(log_temperature)
        probabilities = softmax(values, temperature)
        expected_logits = np.sum(probabilities * values, axis=1)
        gradient = float(np.mean((values[np.arange(len(labels)), labels] - expected_logits) / temperature))
        log_temperature -= float(learning_rate) * gradient
        log_temperature = float(np.clip(log_temperature, math.log(lower), math.log(upper)))
    temperature = math.exp(log_temperature)
    before = softmax(values, 1.0)
    after = softmax(values, temperature)
    return {
        "temperature": temperature,
        "nll_before": negative_log_likelihood(labels, before),
        "nll_after": negative_log_likelihood(labels, after),
        "ece_before": expected_calibration_error(labels, before),
        "ece_after": expected_calibration_error(labels, after),
        "fit_data": "out_of_fold_train_val_predictions",
        "test_split_used": False,
    }


def threshold_sweep(
    y_true: Sequence[int],
    proba: np.ndarray,
    *,
    min_coverage: float = 0.90,
    threshold_step: float = 0.01,
) -> dict[str, Any]:
    labels = np.asarray(y_true, dtype=np.int64)
    values = np.asarray(proba, dtype=np.float64)
    predictions = values.argmax(axis=1)
    confidence = values.max(axis=1)
    baseline_accuracy = float(accuracy_score(labels, predictions))
    rows: list[dict[str, Any]] = []
    for threshold in np.round(np.arange(0.0, 1.0 + threshold_step, threshold_step), 4):
        accepted = confidence >= threshold
        coverage = float(accepted.mean())
        accepted_accuracy = (
            float(accuracy_score(labels[accepted], predictions[accepted])) if accepted.any() else 0.0
        )
        accepted_macro_f1 = (
            float(
                f1_score(
                    labels[accepted],
                    predictions[accepted],
                    labels=list(range(CLASS_COUNT)),
                    average="macro",
                    zero_division=0,
                )
            )
            if accepted.any()
            else 0.0
        )
        rows.append(
            {
                "threshold": float(threshold),
                "coverage": coverage,
                "accepted_count": int(accepted.sum()),
                "rejected_count": int((~accepted).sum()),
                "accepted_accuracy": accepted_accuracy,
                "accepted_macro_f1": accepted_macro_f1,
            }
        )
    eligible = [row for row in rows if row["coverage"] >= min_coverage]
    improved = [row for row in eligible if row["accepted_accuracy"] > baseline_accuracy]
    if improved:
        chosen = sorted(
            improved,
            key=lambda row: (row["accepted_accuracy"], row["coverage"], -row["threshold"]),
            reverse=True,
        )[0]
    else:
        chosen = next(row for row in rows if row["threshold"] == 0.0)
    return {
        "selection_metric": "accepted_accuracy",
        "selection_data": "out_of_fold_train_val_predictions",
        "min_coverage": float(min_coverage),
        "baseline_accuracy": baseline_accuracy,
        "chosen": chosen,
        "candidates": rows,
        "test_split_used": False,
    }


def calibration_report(y_true: Sequence[int], logits: np.ndarray, **kwargs: Any) -> dict[str, Any]:
    options = dict(kwargs)
    min_coverage = float(options.pop("min_coverage", 0.90))
    scaled = temperature_scale(y_true, logits, **options)
    probabilities = softmax(logits, scaled["temperature"])
    threshold = threshold_sweep(y_true, probabilities, min_coverage=min_coverage)
    return {
        "temperature": scaled,
        "threshold": threshold,
        "probabilities": probabilities,
    }


def summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "temperature": report["temperature"]["temperature"],
        "threshold": report["threshold"]["chosen"]["threshold"],
        "coverage": report["threshold"]["chosen"]["coverage"],
        "accepted_accuracy": report["threshold"]["chosen"]["accepted_accuracy"],
        "nll_before": report["temperature"]["nll_before"],
        "nll_after": report["temperature"]["nll_after"],
        "ece_before": report["temperature"]["ece_before"],
        "ece_after": report["temperature"]["ece_after"],
    }
