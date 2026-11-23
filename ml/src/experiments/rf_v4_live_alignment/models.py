from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    from ml.src import train_baselines_v1 as v1
    from ml.src.experiments.rf_v3_generalisation import model as v3_model
    from ml.src.experiments.rf_v4_live_alignment import augment, features
except ModuleNotFoundError as exc:
    if exc.name not in ("ml", "ml.src"):
        raise
    SOURCE_ROOT = str(Path(__file__).resolve().parents[2])
    if SOURCE_ROOT not in sys.path:
        sys.path.insert(0, SOURCE_ROOT)
    import train_baselines_v1 as v1
    from experiments.rf_v3_generalisation import model as v3_model
    from experiments.rf_v4_live_alignment import augment, features


CLASS_COUNT = v3_model.CLASS_COUNT
SEED = 20261123
BASE_FRAME_CHANNEL_COUNT = 135
ENHANCED_FRAME_CHANNEL_COUNT = len(features.ENHANCED_CHANNEL_NAMES)
ENHANCED_REAL_INDEX = features.ENHANCED_CHANNEL_NAMES.index("frame_real")


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return v3_model.metrics(y_true, y_pred)


def class_weight_vector(y: np.ndarray, power: float) -> np.ndarray:
    return v3_model.class_weight_vector(y, power)


def build_config(
    name: str,
    *,
    feature_mode: str,
    architecture: str,
    input_dim: int,
    hidden1: int = 64,
    hidden2: int = 48,
    hidden3: int = 48,
    kernel_size: int = 1,
    attention: bool = False,
    label_smoothing: float = 0.10,
    class_weight_power: float = 1.0,
    augmentation: dict[str, Any] | None = None,
    max_epochs: int = 48,
    patience: int = 8,
) -> dict[str, Any]:
    return {
        "name": name,
        "feature_mode": feature_mode,
        "architecture": architecture,
        "input_dim": int(input_dim),
        "hidden1": int(hidden1),
        "hidden2": int(hidden2),
        "hidden3": int(hidden3),
        "kernel_size": int(kernel_size),
        "attention": bool(attention),
        "label_smoothing": float(label_smoothing),
        "class_weight_power": float(class_weight_power),
        "augmentation": copy.deepcopy(augmentation) if augmentation else None,
        "max_epochs": int(max_epochs),
        "batch_size": 32,
        "learning_rate": 0.0015,
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-8,
        "weight_decay": 1e-5,
        "patience": int(patience),
        "min_delta": 1e-6,
        "seed": SEED,
        "selection_split": "train_val_group_cv",
        "signer_independent": False,
    }


def extract_inputs(
    tensor: np.ndarray,
    feature_mode: str,
    base_preparer: Any,
) -> tuple[np.ndarray, np.ndarray]:
    if feature_mode == "base":
        return base_preparer.prepare_inputs(tensor)
    if feature_mode != "enhanced":
        raise ValueError(f"unknown feature mode: {feature_mode}")
    real = tensor[:, :, ENHANCED_REAL_INDEX] > 0.5
    prepared = np.array(tensor, dtype=np.float32, copy=True)
    prepared[~real] = 0.0
    return prepared, real


def _windows(u: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size == 1:
        return u
    n_samples, time_steps, channels = u.shape
    output = np.zeros((n_samples, time_steps, channels * kernel_size), dtype=np.float32)
    for shift in range(kernel_size):
        lag = kernel_size - 1 - shift
        if lag == 0:
            output[:, :, shift * channels : (shift + 1) * channels] = u
        else:
            output[:, : time_steps - lag, shift * channels : (shift + 1) * channels] = u[:, lag:]
    return output


def _window_gradient(dwindows: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size == 1:
        return dwindows
    n_samples, time_steps, total_channels = dwindows.shape
    channels = total_channels // kernel_size
    output = np.zeros((n_samples, time_steps, channels), dtype=np.float32)
    for shift in range(kernel_size):
        lag = kernel_size - 1 - shift
        block = dwindows[:, :, shift * channels : (shift + 1) * channels]
        if lag == 0:
            output += block
        else:
            output[:, : time_steps - lag] += block[:, lag:]
    return output


class TemporalNetwork:
    def __init__(self, config: Mapping[str, Any]):
        self.config = dict(config)
        self.weights: dict[str, np.ndarray] = {}
        self.class_weights = np.ones(CLASS_COUNT, dtype=np.float32)
        self.label_smoothing = 0.0
        self.initialize(int(self.config["seed"]))

    def initialize(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        input_dim = int(self.config["input_dim"])
        hidden1 = int(self.config["hidden1"])
        hidden2 = int(self.config["hidden2"])
        hidden3 = int(self.config["hidden3"])
        kernel_size = int(self.config["kernel_size"])
        pool_dim = hidden2 * (3 if self.config["attention"] else 2)
        shapes = {
            "W1": (input_dim * kernel_size, hidden1),
            "b1": (hidden1,),
            "W2": (hidden1 * kernel_size, hidden2),
            "b2": (hidden2,),
            "Wa": (hidden2,),
            "ba": (1,),
            "Wp": (pool_dim, hidden3),
            "bp": (hidden3,),
            "Wo": (hidden3, CLASS_COUNT),
            "bo": (CLASS_COUNT,),
        }
        self.weights = {}
        for key, shape in shapes.items():
            if key.startswith("W") and len(shape) == 2:
                scale = np.sqrt(2.0 / max(shape[0], 1))
                self.weights[key] = (rng.normal(0.0, scale, size=shape) * 0.8).astype(np.float32)
            else:
                self.weights[key] = np.zeros(shape, dtype=np.float32)

    def parameter_count(self) -> int:
        return int(sum(int(value.size) for value in self.weights.values()))

    def prepare_inputs(self, tensor: np.ndarray, base_preparer: Any) -> tuple[np.ndarray, np.ndarray]:
        return extract_inputs(tensor, self.config["feature_mode"], base_preparer)

    def forward(self, features_array: np.ndarray, real: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        kernel_size = int(self.config["kernel_size"])
        weights = self.weights
        window1 = _windows(features_array, kernel_size)
        z1 = window1 @ weights["W1"] + weights["b1"]
        a1 = np.maximum(z1, 0.0) * real[:, :, None]
        window2 = _windows(a1, kernel_size)
        z2 = window2 @ weights["W2"] + weights["b2"]
        a2 = np.maximum(z2, 0.0) * real[:, :, None]
        counts = np.maximum(real.sum(axis=1, keepdims=True).astype(np.float32), 1.0)
        mean = a2.sum(axis=1) / counts
        maximum = a2.max(axis=1)
        pooled_parts = [mean, maximum]
        attention_weights = None
        attention_pre = None
        if self.config["attention"]:
            attention_pre = np.tanh(a2 @ weights["Wa"] + weights["ba"][0])
            masked_pre = np.where(real, attention_pre, -1e9)
            shifted = masked_pre - masked_pre.max(axis=1, keepdims=True)
            exponent = np.exp(shifted)
            attention_weights = exponent / exponent.sum(axis=1, keepdims=True)
            pooled_parts.append(np.einsum("nt,ntc->nc", attention_weights, a2))
        pooled = np.concatenate(pooled_parts, axis=1)
        z3 = pooled @ weights["Wp"] + weights["bp"]
        h3 = np.maximum(z3, 0.0)
        logits = h3 @ weights["Wo"] + weights["bo"]
        cache = {
            "features": features_array,
            "real": real,
            "window1": window1,
            "z1": z1,
            "a1": a1,
            "window2": window2,
            "z2": z2,
            "a2": a2,
            "pooled": pooled,
            "z3": z3,
            "h3": h3,
            "counts": counts,
            "maximum": maximum,
            "attention_weights": attention_weights,
            "attention_pre": attention_pre,
        }
        return logits, cache

    def backward(self, cache: dict[str, Any], dlogits: np.ndarray) -> dict[str, np.ndarray]:
        weights = self.weights
        real = cache["real"]
        gradients = {key: np.zeros_like(value) for key, value in weights.items()}
        gradients["Wo"] = cache["h3"].T @ dlogits
        gradients["bo"] = dlogits.sum(axis=0)
        dh3 = dlogits @ weights["Wo"].T
        dz3 = dh3 * (cache["z3"] > 0)
        gradients["Wp"] = cache["pooled"].T @ dz3
        gradients["bp"] = dz3.sum(axis=0)
        dpooled = dz3 @ weights["Wp"].T
        hidden2 = int(self.config["hidden2"])
        split = np.split(dpooled, 3 if self.config["attention"] else 2, axis=1)
        da2 = split[0][:, None, :] / cache["counts"][:, :, None]
        max_mask = (cache["a2"] >= cache["maximum"][:, None, :]) & real[:, :, None]
        max_count = np.maximum(max_mask.sum(axis=1, keepdims=True), 1)
        da2 = da2 + split[1][:, None, :] * max_mask / max_count
        if self.config["attention"]:
            attention_weights = cache["attention_weights"]
            context_gradient = np.einsum("nc,ntc->nt", split[2], cache["a2"])
            weighted = attention_weights * (
                context_gradient - (attention_weights * context_gradient).sum(axis=1, keepdims=True)
            )
            attention_pre_gradient = weighted * (1.0 - cache["attention_pre"] ** 2)
            gradients["Wa"] = np.einsum("ntc,nt->c", cache["a2"], attention_pre_gradient)
            gradients["ba"] = np.asarray([attention_pre_gradient.sum()], dtype=np.float32)
            da2 = da2 + attention_weights[:, :, None] * split[2][:, None, :]
        gradients["W2"] = cache["window2"].reshape(-1, cache["window2"].shape[2]).T @ da2.reshape(
            -1, da2.shape[2]
        )
        gradients["b2"] = da2.sum(axis=(0, 1))
        dwindow2 = da2 @ weights["W2"].T
        du1 = _window_gradient(dwindow2, int(self.config["kernel_size"]))
        da1 = du1 * (cache["z1"] > 0) * real[:, :, None]
        gradients["W1"] = cache["window1"].reshape(-1, cache["window1"].shape[2]).T @ da1.reshape(
            -1, da1.shape[2]
        )
        gradients["b1"] = da1.sum(axis=(0, 1))
        return gradients

    def predict_logits(self, features_array: np.ndarray, real: np.ndarray) -> np.ndarray:
        logits, _ = self.forward(features_array, real)
        return logits

    def predict(self, features_array: np.ndarray, real: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_logits(features_array, real), axis=1).astype(np.int64)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        config: Mapping[str, Any],
        real: np.ndarray | None = None,
        val_x: np.ndarray | None = None,
        val_real: np.ndarray | None = None,
        val_y: np.ndarray | None = None,
        *,
        base_preparer: Any = None,
        preextracted: bool = False,
    ) -> dict[str, Any]:
        settings = dict(config)
        self.label_smoothing = float(settings["label_smoothing"])
        self.class_weights = class_weight_vector(y, float(settings["class_weight_power"]))
        self.initialize(int(settings["seed"]))
        if preextracted:
            if real is None:
                raise ValueError("preextracted training requires a real-frame mask")
            prepared = np.asarray(x)
            real = np.asarray(real, dtype=bool)
        else:
            prepared, real = extract_inputs(x, self.config["feature_mode"], base_preparer)
        val_prepared = val_x
        if val_x is not None and not preextracted:
            val_prepared, val_real = extract_inputs(val_x, self.config["feature_mode"], base_preparer)
        rng = np.random.default_rng(int(settings["seed"]))
        first = {key: np.zeros_like(value) for key, value in self.weights.items()}
        second = {key: np.zeros_like(value) for key, value in self.weights.items()}
        step = 0
        best_score = -np.inf
        best_epoch = 0
        best_weights: dict[str, np.ndarray] | None = None
        history: list[dict[str, float | int]] = []
        target_identity = np.eye(CLASS_COUNT, dtype=np.float32)
        smoothing = float(np.clip(self.label_smoothing, 0.0, 0.5))
        has_validation = val_x is not None and val_y is not None
        for epoch in range(1, int(settings["max_epochs"]) + 1):
            order = rng.permutation(len(y))
            total_loss = 0.0
            batch_count = 0
            for start in range(0, len(y), int(settings["batch_size"])):
                batch_index = order[start : start + int(settings["batch_size"])]
                batch_x, batch_real = augment.augment_batch(
                    prepared[batch_index],
                    real[batch_index],
                    settings["augmentation"],
                    rng,
                    feature_mode=self.config["feature_mode"],
                )
                logits, cache = self.forward(batch_x, batch_real)
                shifted = logits - logits.max(axis=1, keepdims=True)
                exponent = np.exp(shifted)
                probabilities = exponent / exponent.sum(axis=1, keepdims=True)
                target = (1.0 - smoothing) * target_identity[y[batch_index]] + smoothing / CLASS_COUNT
                sample_weights = self.class_weights[y[batch_index]].astype(np.float32)
                weight_total = float(sample_weights.sum())
                log_probabilities = np.log(np.clip(probabilities, 1e-12, 1.0))
                data_loss = -float(np.sum(target * log_probabilities * sample_weights[:, None]) / weight_total)
                regularization = sum(
                    float(np.sum(value * value)) for key, value in self.weights.items() if key.startswith("W")
                )
                loss = data_loss + float(settings["weight_decay"]) * regularization
                dlogits = ((probabilities - target) * sample_weights[:, None] / weight_total).astype(np.float32)
                gradients = self.backward(cache, dlogits)
                for key in self.weights:
                    if key.startswith("W"):
                        gradients[key] += float(settings["weight_decay"]) * 2.0 * self.weights[key]
                    first[key] = float(settings["beta1"]) * first[key] + (1.0 - float(settings["beta1"])) * gradients[key]
                    second[key] = float(settings["beta2"]) * second[key] + (1.0 - float(settings["beta2"])) * (
                        gradients[key] * gradients[key]
                    )
                    step += 1 if key == "W1" else 0
                    correction1 = 1.0 - float(settings["beta1"]) ** step
                    correction2 = 1.0 - float(settings["beta2"]) ** step
                    adjusted_first = first[key] / correction1
                    adjusted_second = second[key] / correction2
                    self.weights[key] -= (
                        float(settings["learning_rate"]) * adjusted_first / (np.sqrt(adjusted_second) + float(settings["epsilon"]))
                    ).astype(np.float32)
                total_loss += loss
                batch_count += 1
            record: dict[str, float | int] = {"epoch": epoch, "train_loss": float(total_loss / max(batch_count, 1))}
            if has_validation:
                prediction = self.predict(val_prepared, val_real)
                val_metrics = metrics(np.asarray(val_y), prediction)
                score = float(val_metrics["macro_f1"])
                record.update(
                    {
                        "val_macro_f1": score,
                        "val_accuracy": float(val_metrics["accuracy"]),
                        "val_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
                    }
                )
                history.append(record)
                if score > best_score + float(settings["min_delta"]):
                    best_score = score
                    best_epoch = epoch
                    best_weights = copy.deepcopy(self.weights)
                    stale = 0
                else:
                    stale = stale + 1 if epoch > 1 else 1
                if stale >= int(settings["patience"]):
                    break
            else:
                history.append(record)
                best_epoch = epoch
                stale = 0
        if has_validation and best_weights is not None:
            self.weights = best_weights
        return {
            "best_epoch": best_epoch,
            "best_validation_macro_f1": float(best_score) if has_validation else None,
            "epochs_completed": len(history),
            "stopped_early": bool(has_validation and len(history) < int(settings["max_epochs"])),
            "class_weights": [float(value) for value in self.class_weights],
            "label_smoothing": self.label_smoothing,
            "history": history,
        }
