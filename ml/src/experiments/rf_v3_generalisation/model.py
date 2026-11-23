from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

try:
    from ml.src import train_baselines_v1 as v1
except ModuleNotFoundError as exc:
    if exc.name not in ("ml", "ml.src"):
        raise
    SOURCE_ROOT = str(Path(__file__).resolve().parents[2])
    if SOURCE_ROOT not in sys.path:
        sys.path.insert(0, SOURCE_ROOT)
    import train_baselines_v1 as v1


CLASS_COUNT = 21
SEED = 20261122


def metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, float]:
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def class_weight_vector(y: np.ndarray, power: float) -> np.ndarray:
    labels = np.asarray(y, dtype=np.int64)
    if power == 0.0:
        return np.ones(CLASS_COUNT, dtype=np.float32)
    counts = np.bincount(labels, minlength=CLASS_COUNT).astype(np.float64)
    frequencies = np.clip(counts / max(counts.sum(), 1.0), 1e-9, None)
    values = frequencies ** (-float(power))
    return (values / values.mean()).astype(np.float32)


def build_config(name: str, *, label_smoothing: float, class_weight_power: float) -> dict[str, Any]:
    config = deepcopy(v1.MLP_CONFIG)
    config["input_shape"] = [346, 89, 135]
    config["training_input_shape"] = [346, 89, 135]
    config["frame_input_shape"] = [89, 135]
    config["architecture"] = "framewise_shared_mlp_masked_mean_max_pooling_regularized"
    config["early_stopping_data"] = "inner_train_val"
    config["regularization"] = {
        "label_smoothing": float(label_smoothing),
        "class_weight_power": float(class_weight_power),
        "class_weight_rule": "inverse_frequency_raised_to_power_normalized_to_mean_one",
    }
    config["selection_split"] = "train_val_group_cv"
    config["variant"] = name
    return config


class RegularizedTemporalMLP(v1.TemporalMLP):
    def __init__(self, config: Mapping[str, Any]):
        super().__init__(dict(config))
        self.class_weights = np.ones(CLASS_COUNT, dtype=np.float32)
        self.label_smoothing = 0.0

    def fit_regularized(
        self,
        features: np.ndarray,
        real: np.ndarray,
        y: np.ndarray,
        config: Mapping[str, Any],
        val_features: np.ndarray | None = None,
        val_real: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict[str, Any]:
        settings = dict(config)
        self.label_smoothing = float(settings.get("regularization", {}).get("label_smoothing", 0.0))
        power = float(settings.get("regularization", {}).get("class_weight_power", 0.0))
        self.class_weights = class_weight_vector(y, power)
        self.initialize(int(settings["seed"]))
        rng = np.random.default_rng(int(settings["seed"]))
        first_moment = {key: np.zeros_like(value) for key, value in self.weights.items()}
        second_moment = {key: np.zeros_like(value) for key, value in self.weights.items()}
        step = 0
        best_score = -np.inf
        best_epoch = 0
        best_weights = None
        history: list[dict[str, float | int]] = []
        n_samples = len(y)
        batch_size = int(settings["batch_size"])
        lr = float(settings["learning_rate"])
        beta1 = float(settings["beta1"])
        beta2 = float(settings["beta2"])
        epsilon = float(settings["epsilon"])
        weight_decay = float(settings["weight_decay"])
        patience = int(settings["early_stopping_patience"])
        min_delta = float(settings["min_delta"])
        has_validation = val_features is not None and val_real is not None and y_val is not None
        max_epochs = int(settings["max_epochs"])
        stale = 0
        target_identity = np.eye(CLASS_COUNT, dtype=np.float32)
        smoothing = float(np.clip(self.label_smoothing, 0.0, 0.5))
        for epoch in range(1, max_epochs + 1):
            order = rng.permutation(n_samples)
            total_loss = 0.0
            total_batches = 0
            for start in range(0, n_samples, batch_size):
                batch_index = order[start : start + batch_size]
                logits, cache = self.forward(features[batch_index], real[batch_index])
                shifted = logits - logits.max(axis=1, keepdims=True)
                exponent = np.exp(shifted)
                probabilities = exponent / exponent.sum(axis=1, keepdims=True)
                target = (1.0 - smoothing) * target_identity[y[batch_index]] + smoothing / CLASS_COUNT
                sample_weights = self.class_weights[y[batch_index]].astype(np.float32)
                weight_total = float(sample_weights.sum())
                log_probabilities = np.log(np.clip(probabilities, 1e-12, 1.0))
                data_loss = -float(np.sum(target * log_probabilities * sample_weights[:, None]) / weight_total)
                regularization = sum(
                    np.sum(value * value) for key, value in self.weights.items() if key.startswith("W")
                )
                loss = data_loss + weight_decay * regularization
                dlogits = (
                    (probabilities - target) * sample_weights[:, None] / weight_total
                ).astype(np.float32)
                gradients = self.backward(cache, dlogits, weight_decay)
                step += 1
                correction1 = 1.0 - beta1**step
                correction2 = 1.0 - beta2**step
                for key in self.weights:
                    first_moment[key] = beta1 * first_moment[key] + (1.0 - beta1) * gradients[key]
                    second_moment[key] = beta2 * second_moment[key] + (1.0 - beta2) * (
                        gradients[key] * gradients[key]
                    )
                    adjusted_first = first_moment[key] / correction1
                    adjusted_second = second_moment[key] / correction2
                    self.weights[key] -= (
                        lr * adjusted_first / (np.sqrt(adjusted_second) + epsilon)
                    ).astype(np.float32)
                total_loss += float(loss)
                total_batches += 1
            epoch_record: dict[str, float | int] = {
                "epoch": epoch,
                "train_loss": float(total_loss / max(total_batches, 1)),
            }
            if has_validation:
                val_prediction = self.predict(val_features, val_real)
                val_metrics = metrics(y_val, val_prediction)
                score = val_metrics["macro_f1"]
                epoch_record.update(
                    {
                        "val_macro_f1": float(score),
                        "val_accuracy": float(val_metrics["accuracy"]),
                        "val_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
                    }
                )
                history.append(epoch_record)
                if score > best_score + min_delta:
                    best_score = score
                    best_epoch = epoch
                    best_weights = deepcopy(self.weights)
                    stale = 0
                else:
                    stale += 1
                if stale >= patience:
                    break
            else:
                history.append(epoch_record)
                best_epoch = epoch
        if has_validation:
            self.weights = best_weights
        return {
            "best_epoch": best_epoch,
            "best_validation_macro_f1": float(best_score) if has_validation else None,
            "epochs_completed": len(history),
            "stopped_early": has_validation and len(history) < max_epochs,
            "early_stopping_data": "inner_train_val" if has_validation else None,
            "class_weights": [float(value) for value in self.class_weights],
            "label_smoothing": self.label_smoothing,
            "history": history,
        }
