from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURES_DIR = REPO_ROOT / "ml/dataset/processed/synora_include21_video_features_v1"
DEFAULT_MODEL_ROOT = REPO_ROOT / "ml/models/synora_baseline_v1_20261121"
DEFAULT_REPORT_ROOT = REPO_ROOT / "ml/reports/synora_baseline_v1_20261121"

SEED = 20261121
SPLITS = ("train", "val", "test")
DUPLICATE_HANDEDNESS_LIMITATION = (
    "886 real frames contained duplicate handedness labels; the highest handedness-score hand "
    "was retained by the v1 feature preparation. This is a known representation limitation."
)
CLAIM_BOUNDARY = (
    "Controlled isolated-sign recognition on the official INCLUDE split. Signer identity is "
    "unverified; no signer-independent generalization claim is made."
)

RF_CONFIG = {
    "estimator": "sklearn.ensemble.RandomForestClassifier",
    "n_estimators": 300,
    "criterion": "gini",
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "min_weight_fraction_leaf": 0.0,
    "max_features": "sqrt",
    "max_leaf_nodes": None,
    "min_impurity_decrease": 0.0,
    "bootstrap": True,
    "oob_score": False,
    "class_weight": None,
    "n_jobs": 1,
    "random_state": SEED,
    "verbose": 0,
    "warm_start": False,
    "ccp_alpha": 0.0,
    "scaling": "none",
}
RF_ESTIMATOR_CONFIG = {key: value for key, value in RF_CONFIG.items() if key not in {"estimator", "scaling"}}

DUMMY_CONFIG = {
    "estimator": "sklearn.dummy.DummyClassifier",
    "strategy": "most_frequent",
    "random_state": SEED,
    "role": "reference_baseline_not_an_ml_model",
}
DUMMY_ESTIMATOR_CONFIG = {
    "strategy": DUMMY_CONFIG["strategy"],
    "random_state": DUMMY_CONFIG["random_state"],
}

MLP_CONFIG = {
    "architecture": "framewise_shared_mlp_masked_mean_max_pooling",
    "input_shape": [443, 89, 135],
    "frame_layers": [135, 96, 48],
    "pooled_features": 96,
    "head_layers": [96, 64, 21],
    "activation": "relu",
    "pooling": ["masked_mean", "masked_max"],
    "max_epochs": 120,
    "batch_size": 32,
    "learning_rate": 0.001,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-8,
    "weight_decay": 0.0001,
    "early_stopping_metric": "macro_f1",
    "early_stopping_mode": "max",
    "early_stopping_patience": 20,
    "min_delta": 0.000001,
    "seed": SEED,
    "initialization": "he_normal_frame_layers_he_uniform_head",
    "input_mask_policy": {
        "padding": "padded frame vectors zeroed and excluded from pooling",
        "presence": "left/right coordinate and confidence channels zeroed unless present",
        "missing": "left_hand_missing/right_hand_missing and no_hand_frame retained as channels",
        "frame_real": "only frame_real=1 frames contribute to masked mean and max",
    },
    "preprocessing": {
        "source": "existing mlp_preprocessing.npz",
        "fit_splits": ["train"],
        "refit": False,
        "scaled_channels": "observed normalized landmark coordinates only",
    },
}

EXPECTED_CHANNELS = [f"{side}_lm{lm:02d}_{axis}" for side in ("left", "right") for lm in range(21) for axis in ("x", "y", "z")]
EXPECTED_CHANNELS.extend(
    [
        "left_hand_present",
        "right_hand_present",
        "left_hand_missing",
        "right_hand_missing",
        "frame_real",
        "no_hand_frame",
        "frame_padded",
        "left_confidence",
        "right_confidence",
    ]
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def package_versions():
    names = ["numpy", "scikit-learn", "joblib", "matplotlib", "pandas", "python"]
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except Exception:
            result[name] = None
    result["python_runtime"] = sys.version.split()[0]
    return result


def compute_metrics(y_true, y_pred):
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def json_report(y_true, y_pred, target_names):
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(target_names))),
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    converted = {}
    for key, value in report.items():
        if isinstance(value, dict):
            converted[key] = {subkey: float(subvalue) for subkey, subvalue in value.items()}
        else:
            converted[key] = float(value)
    return converted


def plot_confusion_matrix(cm, target_names, title, path, normalize=False):
    values = cm.astype(float)
    if normalize:
        row_sums = values.sum(axis=1, keepdims=True)
        values = np.divide(values, row_sums, out=np.zeros_like(values), where=row_sums != 0)
    fig, ax = plt.subplots(figsize=(15, 13), dpi=120)
    image = ax.imshow(values, cmap="Blues", vmin=0.0, vmax=1.0 if normalize else None)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Predicted model index")
    ax.set_ylabel("True model index")
    ticks = list(range(len(target_names)))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([str(i) for i in ticks], fontsize=8)
    ax.set_yticklabels([f"{i}:{name}" for i, name in enumerate(target_names)], fontsize=7)
    plt.setp(ax.get_xticklabels(), rotation=90)
    threshold = 0.02 if normalize else 0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if value > threshold:
                text = f"{value:.2f}" if normalize else str(int(cm[i, j]))
                ax.text(j, i, text, ha="center", va="center", fontsize=6, color="black" if value < 0.6 else "white")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def input_hash_map(features_dir: Path, config):
    paths = {
        "config.json": True,
        "rf_feature_schema.json": True,
        "mlp_feature_schema.json": True,
        "validation_report.json": True,
        "video_manifest.csv": True,
        "rf_features.npy": True,
        "rf_features.csv": True,
        "mlp_tensor.npy": True,
        "mlp_flat.npy": False,
        "mlp_preprocessing.npz": True,
        "split_inventories.json": True,
    }
    result = {}
    for name, used in paths.items():
        path = features_dir / name
        result[f"features/{name}"] = {"path": str(path), "sha256": sha256_file(path), "used": used}
    preparation = Path(config["source_hashes"]["preparation_script"]["path"])
    result["preparation_script"] = {
        "path": str(preparation),
        "sha256": sha256_file(preparation),
        "used": True,
    }
    label_map = Path(config["source_hashes"]["label_map"]["path"])
    result["label_map"] = {"path": str(label_map), "sha256": sha256_file(label_map), "used": True}
    return result


def load_and_validate(features_dir: Path):
    config = read_json(features_dir / "config.json")
    validation = read_json(features_dir / "validation_report.json")
    rf_schema = read_json(features_dir / "rf_feature_schema.json")
    mlp_schema = read_json(features_dir / "mlp_feature_schema.json")
    if validation.get("passed") is not True:
        raise RuntimeError("v1 validation_report.json did not pass")
    if config.get("schema_version") != "synora_video_features_v1":
        raise RuntimeError("unexpected feature schema version")

    with open(features_dir / "video_manifest.csv", "r", encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    if len(manifest) != 443:
        raise RuntimeError("manifest row count is not 443")
    if len(set(row["video_id"] for row in manifest)) != 443:
        raise RuntimeError("duplicate video_id in manifest")
    if len(set(row["official_rel_path_posix"] for row in manifest)) != 443:
        raise RuntimeError("duplicate official path in manifest")

    class_rows = sorted(config["primary_classes"], key=lambda item: int(item["folder_id"]))
    if len(class_rows) != 21:
        raise RuntimeError("primary class count is not 21")
    label_mapping = []
    original_to_model = {}
    for model_index, item in enumerate(class_rows):
        original_id = int(item["label_id"])
        record = {
            "model_index": model_index,
            "original_label_id": original_id,
            "class_label": item["class_label"],
            "class_folder": item["class_folder"],
            "folder_id": int(item["folder_id"]),
        }
        label_mapping.append(record)
        original_to_model[original_id] = model_index

    label_map_path = Path(config["source_hashes"]["label_map"]["path"])
    official_label_map = read_json(label_map_path)
    for record in label_mapping:
        if int(official_label_map[record["class_label"]]) != record["original_label_id"]:
            raise RuntimeError("official label map mismatch")

    split_indices = {}
    split_video_ids = {}
    split_paths = {}
    for split in SPLITS:
        indices = [i for i, row in enumerate(manifest) if row["split"] == split]
        split_indices[split] = np.asarray(indices, dtype=np.int64)
        split_video_ids[split] = set(manifest[i]["video_id"] for i in indices)
        split_paths[split] = set(manifest[i]["official_rel_path_posix"] for i in indices)
    expected_counts = {"train": 317, "val": 42, "test": 84}
    for split in SPLITS:
        if len(split_indices[split]) != expected_counts[split]:
            raise RuntimeError(f"unexpected {split} count")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if split_video_ids[a] & split_video_ids[b]:
            raise RuntimeError(f"video overlap between {a} and {b}")
        if split_paths[a] & split_paths[b]:
            raise RuntimeError(f"official path overlap between {a} and {b}")

    official_dir = Path(config["source_hashes"]["official_splits"]["train"]["path"]).parent
    official_lines = {}
    for split, filename in (("train", "include_train.txt"), ("val", "include_val.txt"), ("test", "include_test.txt")):
        path = official_dir / filename
        if sha256_file(path) != config["source_hashes"]["official_splits"][split]["sha256"]:
            raise RuntimeError(f"official split hash mismatch for {split}")
        official_lines[split] = set(line.strip().replace("\\", "/") for line in path.read_text(encoding="utf-8").splitlines() if line.strip())

    y = np.empty(len(manifest), dtype=np.int64)
    for i, row in enumerate(manifest):
        original_id = int(row["label_id"])
        if original_id not in original_to_model:
            raise RuntimeError("manifest contains label outside primary mapping")
        expected_folder = next(item for item in label_mapping if item["original_label_id"] == original_id)
        if row["class_folder"] != expected_folder["class_folder"] or row["class_label"] != expected_folder["class_label"]:
            raise RuntimeError("manifest label mapping mismatch")
        if row["official_rel_path_posix"] not in official_lines[row["split"]]:
            raise RuntimeError("manifest row not in official split file")
        if hashlib.sha256(row["official_rel_path_posix"].encode("utf-8")).hexdigest() != row["video_id"]:
            raise RuntimeError("video_id does not match official path hash")
        y[i] = original_to_model[original_id]

    support_counts = {}
    for split in SPLITS:
        support_counts[split] = {
            str(record["model_index"]): int(np.sum(y[split_indices[split]] == record["model_index"]))
            for record in label_mapping
        }

    X_rf = np.load(features_dir / "rf_features.npy")
    feature_names = rf_schema["feature_columns"]
    if X_rf.shape != (443, 831) or len(feature_names) != 831:
        raise RuntimeError("RF dimensions do not match schema")
    id_columns = set(rf_schema["csv_id_columns"])
    if id_columns.intersection(feature_names):
        raise RuntimeError("RF feature columns contain ID metadata")
    if not np.isfinite(X_rf).all():
        raise RuntimeError("RF matrix contains non-finite values")
    with open(features_dir / "rf_features.csv", "r", encoding="utf-8", newline="") as handle:
        rf_rows = list(csv.DictReader(handle))
    if len(rf_rows) != 443:
        raise RuntimeError("RF CSV row count mismatch")
    rf_from_csv = np.asarray([[float(row[name]) for name in feature_names] for row in rf_rows], dtype=np.float32)
    if not np.array_equal(X_rf, rf_from_csv):
        raise RuntimeError("RF CSV does not match RF NPY")

    tensor = np.load(features_dir / "mlp_tensor.npy")
    flat = np.load(features_dir / "mlp_flat.npy")
    if tensor.shape != (443, 89, 135) or flat.shape != (443, 12015):
        raise RuntimeError("MLP dimensions do not match schema")
    if mlp_schema["channel_order"] != EXPECTED_CHANNELS:
        raise RuntimeError("MLP channel order mismatch")
    if not np.array_equal(flat, tensor.reshape(443, -1)):
        raise RuntimeError("mlp_flat.npy is not a reshape of mlp_tensor.npy")
    if not np.isfinite(tensor).all():
        raise RuntimeError("MLP tensor contains non-finite values")

    presence_left = tensor[:, :, 126] > 0.5
    presence_right = tensor[:, :, 127] > 0.5
    missing_left = tensor[:, :, 128] > 0.5
    missing_right = tensor[:, :, 129] > 0.5
    frame_real = tensor[:, :, 130] > 0.5
    no_hand = tensor[:, :, 131] > 0.5
    frame_padded = tensor[:, :, 132] > 0.5
    if not np.array_equal(frame_real, ~frame_padded):
        raise RuntimeError("frame_real and frame_padded are not complements")
    if not np.array_equal(missing_left, frame_real & ~presence_left):
        raise RuntimeError("left missing mask mismatch")
    if not np.array_equal(missing_right, frame_real & ~presence_right):
        raise RuntimeError("right missing mask mismatch")
    if not np.array_equal(no_hand, frame_real & ~presence_left & ~presence_right):
        raise RuntimeError("no-hand mask mismatch")
    for i in range(tensor.shape[0]):
        count = int(frame_real[i].sum())
        if not (np.all(frame_real[i, :count]) and np.all(~frame_real[i, count:])):
            raise RuntimeError("real frames are not a temporal prefix")
    if not np.all(tensor[frame_padded, 0:126] == 0):
        raise RuntimeError("padded coordinate slots are not zero")
    if not np.all(tensor[frame_padded, 133:135] == 0):
        raise RuntimeError("padded confidence slots are not zero")
    if not np.all(tensor[missing_left, 0:63] == 0):
        raise RuntimeError("missing left-hand coordinate slots are not zero")
    if not np.all(tensor[missing_right, 63:126] == 0):
        raise RuntimeError("missing right-hand coordinate slots are not zero")
    if not np.all(tensor[missing_left, 133] == 0) or not np.all(tensor[missing_right, 134] == 0):
        raise RuntimeError("missing-hand confidence slots are not zero")
    frame_counts = frame_real.sum(axis=1).astype(np.int64)
    if not np.array_equal(frame_counts, np.asarray([int(row["frame_count"]) for row in manifest], dtype=np.int64)):
        raise RuntimeError("tensor frame counts do not match manifest")
    if not np.array_equal(no_hand.sum(axis=1).astype(np.int64), np.asarray([int(row["no_hand_frames"]) for row in manifest], dtype=np.int64)):
        raise RuntimeError("tensor no-hand counts do not match manifest")

    with np.load(features_dir / "mlp_preprocessing.npz") as preprocessing:
        coordinate_mean = preprocessing["coordinate_mean"].copy()
        coordinate_std = preprocessing["coordinate_std"].copy()
    if coordinate_mean.shape != (2, 21, 3) or coordinate_std.shape != (2, 21, 3):
        raise RuntimeError("MLP preprocessing statistics shape mismatch")
    if not np.isfinite(coordinate_mean).all() or not np.isfinite(coordinate_std).all():
        raise RuntimeError("MLP preprocessing statistics contain non-finite values")

    checks = {
        "validation_report_passed": True,
        "row_count": len(manifest),
        "class_count": len(label_mapping),
        "train_val_video_overlap": 0,
        "train_test_video_overlap": 0,
        "val_test_video_overlap": 0,
        "official_path_cross_split_overlap": 0,
        "rf_shape": list(X_rf.shape),
        "rf_csv_matches_npy": True,
        "rf_non_finite_values": int((~np.isfinite(X_rf)).sum()),
        "mlp_tensor_shape": list(tensor.shape),
        "mlp_flat_shape": list(flat.shape),
        "mlp_flat_is_tensor_reshape": True,
        "mlp_non_finite_values": int((~np.isfinite(tensor)).sum()),
        "mlp_mask_invariants_passed": True,
        "mlp_frame_counts_match_manifest": True,
        "input_hashes_match_config": True,
        "duplicate_handedness_limitation_recorded": True,
    }
    return {
        "config": config,
        "manifest": manifest,
        "y": y,
        "label_mapping": label_mapping,
        "split_indices": split_indices,
        "support_counts": support_counts,
        "X_rf": X_rf,
        "feature_names": feature_names,
        "tensor": tensor,
        "coordinate_mean": coordinate_mean,
        "coordinate_std": coordinate_std,
        "checks": checks,
    }


class TemporalMLP:
    def __init__(self, config):
        self.config = config
        self.weights = None

    @staticmethod
    def expected_channels():
        return list(EXPECTED_CHANNELS)

    def initialize(self, seed):
        rng = np.random.default_rng(seed)
        self.weights = {
            "W1": rng.normal(0.0, np.sqrt(2.0 / 135.0), size=(135, 96)).astype(np.float32),
            "b1": np.zeros(96, dtype=np.float32),
            "W2": rng.normal(0.0, np.sqrt(2.0 / 96.0), size=(96, 48)).astype(np.float32),
            "b2": np.zeros(48, dtype=np.float32),
            "Wp": rng.normal(0.0, np.sqrt(2.0 / 96.0), size=(96, 64)).astype(np.float32),
            "bp": np.zeros(64, dtype=np.float32),
            "Wo": rng.normal(0.0, np.sqrt(1.0 / 64.0), size=(64, 21)).astype(np.float32),
            "bo": np.zeros(21, dtype=np.float32),
        }
        return self

    def prepare_inputs(self, tensor):
        features = np.array(tensor, dtype=np.float32, copy=True)
        present_left = features[:, :, 126:127]
        present_right = features[:, :, 127:128]
        features[:, :, 0:63] *= present_left
        features[:, :, 63:126] *= present_right
        features[:, :, 133:134] *= present_left
        features[:, :, 134:135] *= present_right
        padded = features[:, :, 132] > 0.5
        features[padded] = 0.0
        real = (tensor[:, :, 130] > 0.5).astype(np.float32)
        return features, real

    def forward(self, features, real):
        w = self.weights
        batch_size = features.shape[0]
        frames = features.shape[1]
        x = features.reshape(-1, 135)
        z1 = x @ w["W1"] + w["b1"]
        h1 = np.maximum(z1, 0.0)
        z2 = h1 @ w["W2"] + w["b2"]
        h2 = np.maximum(z2, 0.0)
        hidden = h2.reshape(batch_size, frames, 48)
        mask = real > 0.5
        counts = np.clip(mask.sum(axis=1, keepdims=True), 1, None).astype(np.float32)
        masked_sum = (hidden * mask[:, :, None]).sum(axis=1)
        mean_pool = masked_sum / counts
        masked_hidden = np.where(mask[:, :, None], hidden, np.float32(-np.inf))
        max_pool = np.max(masked_hidden, axis=1)
        pooled = np.concatenate([mean_pool, max_pool], axis=1)
        z3 = pooled @ w["Wp"] + w["bp"]
        h3 = np.maximum(z3, 0.0)
        logits = h3 @ w["Wo"] + w["bo"]
        cache = {
            "features": features,
            "real": real,
            "mask": mask,
            "counts": counts,
            "z1": z1.reshape(batch_size, frames, 96),
            "h1": h1.reshape(batch_size, frames, 96),
            "z2": z2.reshape(batch_size, frames, 48),
            "h2": hidden,
            "pooled": pooled,
            "z3": z3,
            "h3": h3,
        }
        return logits, cache

    def backward(self, cache, dlogits, weight_decay):
        w = self.weights
        batch_size = cache["features"].shape[0]
        frames = cache["features"].shape[1]
        dW_out = cache["h3"].T @ dlogits + 2.0 * weight_decay * w["Wo"]
        db_out = dlogits.sum(axis=0)
        dh3 = dlogits @ w["Wo"].T
        dz3 = dh3 * (cache["z3"] > 0.0)
        dW_pool = cache["pooled"].T @ dz3 + 2.0 * weight_decay * w["Wp"]
        db_pool = dz3.sum(axis=0)
        dpool = dz3 @ w["Wp"].T
        dmean = dpool[:, 0:48]
        dmax = dpool[:, 48:96]
        dh2 = np.zeros_like(cache["h2"])
        dh2 += (dmean[:, None, :] * cache["mask"][:, :, None]) / cache["counts"][:, None, :]
        max_indices = np.argmax(np.where(cache["mask"][:, :, None], cache["h2"], np.float32(-np.inf)), axis=1)
        flat_indices = (np.arange(batch_size)[:, None] * frames + max_indices).reshape(-1)
        dh2_flat = dh2.reshape(-1, 48)
        np.add.at(dh2_flat, (flat_indices, np.tile(np.arange(48), batch_size)), dmax.reshape(-1))
        dh2 = dh2_flat.reshape(batch_size, frames, 48)
        dz2 = dh2 * (cache["z2"] > 0.0)
        dz2_flat = dz2.reshape(-1, 48)
        dW2 = cache["h1"].reshape(-1, 96).T @ dz2_flat + 2.0 * weight_decay * w["W2"]
        db2 = dz2_flat.sum(axis=0)
        dh1_flat = dz2_flat @ w["W2"].T
        dh1 = dh1_flat.reshape(batch_size, frames, 96)
        dz1 = dh1 * (cache["z1"] > 0.0)
        dW1 = cache["features"].reshape(-1, 135).T @ dz1.reshape(-1, 96) + 2.0 * weight_decay * w["W1"]
        db1 = dz1.sum(axis=(0, 1))
        return {
            "W1": dW1.astype(np.float32),
            "b1": db1.astype(np.float32),
            "W2": dW2.astype(np.float32),
            "b2": db2.astype(np.float32),
            "Wp": dW_pool.astype(np.float32),
            "bp": db_pool.astype(np.float32),
            "Wo": dW_out.astype(np.float32),
            "bo": db_out.astype(np.float32),
        }

    def predict_logits(self, features, real):
        logits, _ = self.forward(features, real)
        return logits

    def predict(self, features, real):
        return np.argmax(self.predict_logits(features, real), axis=1).astype(np.int64)

    def predict_proba(self, features, real):
        logits = self.predict_logits(features, real)
        shifted = logits - logits.max(axis=1, keepdims=True)
        exponent = np.exp(shifted)
        return (exponent / exponent.sum(axis=1, keepdims=True)).astype(np.float64)

    def fit(self, features, real, y, config):
        self.initialize(int(config["seed"]))
        rng = np.random.default_rng(int(config["seed"]))
        first_moment = {key: np.zeros_like(value) for key, value in self.weights.items()}
        second_moment = {key: np.zeros_like(value) for key, value in self.weights.items()}
        step = 0
        best_score = -np.inf
        best_epoch = 0
        best_weights = None
        history = []
        n_samples = len(y)
        batch_size = int(config["batch_size"])
        lr = float(config["learning_rate"])
        beta1 = float(config["beta1"])
        beta2 = float(config["beta2"])
        epsilon = float(config["epsilon"])
        weight_decay = float(config["weight_decay"])
        patience = int(config["early_stopping_patience"])
        min_delta = float(config["min_delta"])
        stale = 0
        for epoch in range(1, int(config["max_epochs"]) + 1):
            order = rng.permutation(n_samples)
            total_loss = 0.0
            total_batches = 0
            for start in range(0, n_samples, batch_size):
                batch_index = order[start:start + batch_size]
                logits, cache = self.forward(features[batch_index], real[batch_index])
                shifted = logits - logits.max(axis=1, keepdims=True)
                exponent = np.exp(shifted)
                probabilities = exponent / exponent.sum(axis=1, keepdims=True)
                target = np.eye(21, dtype=np.float32)[y[batch_index]]
                data_loss = -np.log(np.clip(probabilities[target.astype(bool)], 1e-12, 1.0)).mean()
                regularization = sum(np.sum(value * value) for key, value in self.weights.items() if key.startswith("W"))
                loss = data_loss + weight_decay * regularization
                gradients = self.backward(cache, ((probabilities - target) / len(batch_index)).astype(np.float32), weight_decay)
                step += 1
                correction1 = 1.0 - beta1 ** step
                correction2 = 1.0 - beta2 ** step
                for key in self.weights:
                    first_moment[key] = beta1 * first_moment[key] + (1.0 - beta1) * gradients[key]
                    second_moment[key] = beta2 * second_moment[key] + (1.0 - beta2) * (gradients[key] * gradients[key])
                    adjusted_first = first_moment[key] / correction1
                    adjusted_second = second_moment[key] / correction2
                    self.weights[key] -= (lr * adjusted_first / (np.sqrt(adjusted_second) + epsilon)).astype(np.float32)
                total_loss += float(loss)
                total_batches += 1
            val_prediction = self.predict(features, real)
            val_metrics = compute_metrics(y, val_prediction)
            score = val_metrics["macro_f1"]
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(total_loss / max(total_batches, 1)),
                    "val_macro_f1": float(score),
                    "val_accuracy": float(val_metrics["accuracy"]),
                    "val_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
                }
            )
            if score > best_score + min_delta:
                best_score = score
                best_epoch = epoch
                best_weights = deepcopy(self.weights)
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                break
        self.weights = best_weights
        return {
            "best_epoch": best_epoch,
            "best_validation_macro_f1": float(best_score),
            "epochs_completed": len(history),
            "stopped_early": len(history) < int(config["max_epochs"]),
            "history": history,
        }


def measure_latency(label, predict_callable, X, repeats=5):
    for _ in range(2):
        predict_callable(X[: min(8, len(X))])
    durations = []
    for _ in range(repeats):
        started = time.perf_counter()
        predict_callable(X)
        durations.append(time.perf_counter() - started)
    mean_seconds = float(np.mean(durations))
    return {
        "model": label,
        "split": "test",
        "batch_size": int(len(X)),
        "repeats": int(repeats),
        "mean_batch_seconds": mean_seconds,
        "std_batch_seconds": float(np.std(durations)),
        "mean_ms_per_video": 1000.0 * mean_seconds / len(X),
    }


def main():
    features_dir = DEFAULT_FEATURES_DIR.resolve()
    model_root = DEFAULT_MODEL_ROOT.resolve()
    report_root = DEFAULT_REPORT_ROOT.resolve()
    if model_root.exists() or report_root.exists():
        raise SystemExit("versioned output directory already exists; refusing to overwrite")

    data = load_and_validate(features_dir)
    input_hashes_before = input_hash_map(features_dir, data["config"])
    train_script_path = Path(__file__).resolve()
    train_script_hash = {
        "path": str(train_script_path),
        "sha256": sha256_file(train_script_path),
        "used": True,
    }
    input_hashes_before["train_script"] = train_script_hash
    y = data["y"]
    split_indices = data["split_indices"]
    label_mapping = data["label_mapping"]
    target_names = [record["class_label"] for record in label_mapping]
    train_idx = split_indices["train"]
    val_idx = split_indices["val"]
    test_idx = split_indices["test"]

    model_root.mkdir(parents=True)
    report_root.mkdir(parents=True)
    model_dirs = {
        "random_forest": model_root / "random_forest",
        "temporal_mlp": model_root / "temporal_mlp",
        "dummy": model_root / "dummy",
    }
    for path in model_dirs.values():
        path.mkdir(parents=True)

    experiment_config = {
        "experiment_id": "synora_baseline_v1_20261121",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "INCLUDE 21-class experiment",
        "feature_artifact": str(features_dir),
        "seed": SEED,
        "split_policy": "official train/val/test indices from video_manifest.csv; unchanged",
        "label_policy": "official label IDs mapped to contiguous 0-20 model indices; original mapping preserved",
        "label_mapping": label_mapping,
        "split_counts": {split: len(split_indices[split]) for split in SPLITS},
        "support_counts": data["support_counts"],
        "selection_policy": "validation macro-F1 only; test evaluated once after configs frozen",
        "models": {
            "random_forest": RF_CONFIG,
            "temporal_mlp": MLP_CONFIG,
            "dummy": DUMMY_CONFIG,
        },
        "known_limitations": [DUPLICATE_HANDEDNESS_LIMITATION, CLAIM_BOUNDARY],
        "no_test_tuning": True,
        "no_augmentation": True,
        "v1_features_immutable": True,
    }
    write_json(report_root / "experiment_config.json", experiment_config)
    write_json(report_root / "validation_checks.json", data["checks"])

    reproducibility = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": package_versions(),
        "thread_environment": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
        },
        "reproduction_command": [
            sys.executable,
            str(Path(__file__).resolve()),
        ],
        "numeric_mode": "float32 NumPy linear algebra; deterministic fixed-seed initialization; no dropout",
    }

    rf_model = RandomForestClassifier(**RF_ESTIMATOR_CONFIG)
    rf_fit_started = time.perf_counter()
    rf_model.fit(data["X_rf"][train_idx], y[train_idx])
    rf_fit_seconds = time.perf_counter() - rf_fit_started
    rf_predictions = {
        split: rf_model.predict(data["X_rf"][indices]).astype(np.int64)
        for split, indices in split_indices.items()
    }
    rf_proba_test = rf_model.predict_proba(data["X_rf"][test_idx])
    joblib.dump(rf_model, model_dirs["random_forest"] / "random_forest.joblib")
    write_json(model_dirs["random_forest"] / "config.json", RF_CONFIG)

    dummy_model = DummyClassifier(**DUMMY_ESTIMATOR_CONFIG)
    dummy_fit_started = time.perf_counter()
    dummy_model.fit(data["X_rf"][train_idx], y[train_idx])
    dummy_fit_seconds = time.perf_counter() - dummy_fit_started
    dummy_predictions = {
        split: dummy_model.predict(data["X_rf"][indices]).astype(np.int64)
        for split, indices in split_indices.items()
    }
    joblib.dump(dummy_model, model_dirs["dummy"] / "dummy.joblib")
    write_json(model_dirs["dummy"] / "config.json", DUMMY_CONFIG)

    temporal_model = TemporalMLP(MLP_CONFIG)
    prepared_features, real_mask = temporal_model.prepare_inputs(data["tensor"])
    mlp_fit_started = time.perf_counter()
    mlp_training_history = temporal_model.fit(prepared_features[train_idx], real_mask[train_idx], y[train_idx], MLP_CONFIG)
    mlp_fit_seconds = time.perf_counter() - mlp_fit_started
    mlp_predictions = {
        split: temporal_model.predict(prepared_features[indices], real_mask[indices]).astype(np.int64)
        for split, indices in split_indices.items()
    }
    mlp_proba_test = temporal_model.predict_proba(prepared_features[test_idx], real_mask[test_idx])
    weight_arrays = {key: np.asarray(value, dtype=np.float32) for key, value in temporal_model.weights.items()}
    np.savez_compressed(model_dirs["temporal_mlp"] / "weights.npz", **weight_arrays)
    write_json(model_dirs["temporal_mlp"] / "config.json", MLP_CONFIG)
    write_json(
        model_dirs["temporal_mlp"] / "training_history.json",
        mlp_training_history,
    )
    write_json(
        model_dirs["temporal_mlp"] / "preprocessing_reference.json",
        {
            "path": str(features_dir / "mlp_preprocessing.npz"),
            "sha256": input_hashes_before["features/mlp_preprocessing.npz"]["sha256"],
            "used_without_modification": True,
        },
    )

    predictions = {
        "random_forest": rf_predictions,
        "temporal_mlp": mlp_predictions,
        "dummy": dummy_predictions,
    }
    validation_metrics = {}
    for name, prediction_map in predictions.items():
        validation_metrics[name] = compute_metrics(y[val_idx], prediction_map["val"])
    selected_model = max(("random_forest", "temporal_mlp"), key=lambda name: validation_metrics[name]["macro_f1"])
    write_json(
        report_root / "validation_metrics.json",
        {
            "metrics": validation_metrics,
            "selection_metric": "macro_f1",
            "selected_by_validation": selected_model,
            "selection_uses_test": False,
        },
    )

    test_metrics = {}
    for name, prediction_map in predictions.items():
        test_metrics[name] = compute_metrics(y[test_idx], prediction_map["test"])
    write_json(
        report_root / "test_metrics.json",
        {
            "metrics": test_metrics,
            "evaluation_policy": "final one-time test evaluation after configs frozen",
            "selected_by_validation": selected_model,
        },
    )

    classification_reports = {}
    confusion_matrices = {}
    for name, prediction_map in predictions.items():
        classification_reports[name] = {}
        confusion_matrices[name] = {}
        for split in SPLITS:
            truth = y[split_indices[split]]
            predicted = prediction_map[split]
            classification_reports[name][split] = json_report(truth, predicted, target_names)
            confusion_matrices[name][split] = confusion_matrix(truth, predicted, labels=list(range(21))).tolist()
    write_json(report_root / "classification_reports.json", classification_reports)
    write_json(report_root / "confusion_matrices.json", confusion_matrices)
    write_json(report_root / "support_counts.json", data["support_counts"])

    per_class_rows = []
    for name in predictions:
        for split in SPLITS:
            report = classification_reports[name][split]
            for record in label_mapping:
                key = record["class_label"]
                per_class_rows.append(
                    {
                        "model": name,
                        "split": split,
                        "model_index": int(record["model_index"]),
                        "original_label_id": int(record["original_label_id"]),
                        "class_label": record["class_label"],
                        "class_folder": record["class_folder"],
                        "precision": report[key]["precision"],
                        "recall": report[key]["recall"],
                        "f1_score": report[key]["f1-score"],
                        "support": int(report[key]["support"]),
                    }
                )
    write_csv(
        report_root / "per_class_metrics.csv",
        per_class_rows,
        ["model", "split", "model_index", "original_label_id", "class_label", "class_folder", "precision", "recall", "f1_score", "support"],
    )

    for name in predictions:
        for split in SPLITS:
            cm = np.asarray(confusion_matrices[name][split], dtype=np.int64)
            plot_confusion_matrix(
                cm,
                target_names,
                f"{name} confusion matrix - {split}",
                report_root / "confusion_matrices" / f"{name}_{split}.png",
                normalize=False,
            )
            if split == "test":
                plot_confusion_matrix(
                    cm,
                    target_names,
                    f"{name} row-normalized confusion matrix - test",
                    report_root / "confusion_matrices" / f"{name}_test_normalized.png",
                    normalize=True,
                )

    prediction_rows = []
    for name, prediction_map in predictions.items():
        for split in SPLITS:
            indices = split_indices[split]
            for position, index in enumerate(indices):
                truth = int(y[index])
                predicted = int(prediction_map[split][position])
                row = data["manifest"][index]
                prediction_rows.append(
                    {
                        "model": name,
                        "split": split,
                        "sample_index": int(row["sample_index"]),
                        "video_id": row["video_id"],
                        "class_label": row["class_label"],
                        "original_label_id": int(row["label_id"]),
                        "true_model_index": truth,
                        "predicted_model_index": predicted,
                        "predicted_class_label": target_names[predicted],
                        "predicted_original_label_id": label_mapping[predicted]["original_label_id"],
                        "correct": int(truth == predicted),
                    }
                )
    write_csv(
        report_root / "predictions.csv",
        prediction_rows,
        ["model", "split", "sample_index", "video_id", "class_label", "original_label_id", "true_model_index", "predicted_model_index", "predicted_class_label", "predicted_original_label_id", "correct"],
    )

    latencies = [
        measure_latency("random_forest", lambda batch: rf_model.predict(batch), data["X_rf"][test_idx]),
        measure_latency("dummy", lambda batch: dummy_model.predict(batch), data["X_rf"][test_idx]),
        measure_latency("temporal_mlp", lambda batch: temporal_model.predict(batch, real_mask[test_idx][: len(batch)]), prepared_features[test_idx]),
    ]
    for entry in latencies:
        entry["prepared_for_temporal_mlp"] = entry["model"] == "temporal_mlp"
    timing = {
        "fit_seconds": {
            "random_forest": float(rf_fit_seconds),
            "dummy": float(dummy_fit_seconds),
            "temporal_mlp": float(mlp_fit_seconds),
        },
        "inference_latency": latencies,
        "mlp_best_epoch": int(mlp_training_history["best_epoch"]),
        "mlp_epochs_completed": int(mlp_training_history["epochs_completed"]),
        "mlp_stopped_early": bool(mlp_training_history["stopped_early"]),
    }
    write_json(report_root / "timing.json", timing)

    reloaded_rf = joblib.load(model_dirs["random_forest"] / "random_forest.joblib")
    reloaded_dummy = joblib.load(model_dirs["dummy"] / "dummy.joblib")
    reloaded_mlp = TemporalMLP(read_json(model_dirs["temporal_mlp"] / "config.json"))
    with np.load(model_dirs["temporal_mlp"] / "weights.npz") as weights:
        reloaded_mlp.weights = {key: weights[key].copy() for key in weights.files}
    reload_predictions = {
        "random_forest": {
            split: reloaded_rf.predict(data["X_rf"][indices]).astype(np.int64)
            for split, indices in split_indices.items()
        },
        "dummy": {
            split: reloaded_dummy.predict(data["X_rf"][indices]).astype(np.int64)
            for split, indices in split_indices.items()
        },
        "temporal_mlp": {
            split: reloaded_mlp.predict(prepared_features[indices], real_mask[indices]).astype(np.int64)
            for split, indices in split_indices.items()
        },
    }
    reload_checks = {}
    for name in predictions:
        for split in SPLITS:
            mismatches = int(np.sum(reload_predictions[name][split] != predictions[name][split]))
            reload_checks[f"{name}:{split}"] = {
                "mismatches": mismatches,
                "reproduced": mismatches == 0,
            }
    if not all(item["reproduced"] for item in reload_checks.values()):
        raise RuntimeError("reloaded model predictions did not reproduce")
    write_json(report_root / "reload_verification.json", reload_checks)

    input_hashes_after = input_hash_map(features_dir, data["config"])
    input_hashes_after["train_script"] = {
        "path": str(train_script_path),
        "sha256": sha256_file(train_script_path),
        "used": True,
    }
    if input_hashes_before != input_hashes_after:
        raise RuntimeError("input artifact hashes changed during experiment")
    reproducibility["input_hashes"] = input_hashes_before
    reproducibility["input_hashes_stable_during_run"] = True
    write_json(report_root / "reproducibility.json", reproducibility)

    np.savez_compressed(
        report_root / "test_probabilities.npz",
        random_forest=rf_proba_test,
        temporal_mlp=mlp_proba_test,
    )

    def metric_line(name):
        validation = validation_metrics[name]
        test = test_metrics[name]
        return (
            f"| {name} | {validation['macro_f1']:.4f} | {validation['accuracy']:.4f} | "
            f"{validation['balanced_accuracy']:.4f} | {test['macro_f1']:.4f} | "
            f"{test['accuracy']:.4f} | {test['balanced_accuracy']:.4f} |"
        )

    summary = f"""# SYNORA Controlled Baseline v1

Generated: {datetime.now(timezone.utc).isoformat()}

This is a controlled isolated-sign baseline on the official 21-class INCLUDE split. It is not evidence of signer-independent generalization.

## Frozen configurations

- Random Forest: 300 trees, Gini, `max_features='sqrt'`, `min_samples_leaf=1`, bootstrap=True, no class weighting, no scaling, seed={SEED}.
- Temporal MLP: shared 135→96→48 ReLU frame network, masked mean+max pooling over real frames, 96→64→21 ReLU head, Adam lr=0.001, weight decay=0.0001, batch size 32, max 120 epochs, early stopping patience 20 on validation macro-F1, seed={SEED}.
- Dummy: `DummyClassifier(strategy='most_frequent')`, reference only.

## Label and split integrity

- 443 videos, 21 classes, official train/val/test = 317/42/84.
- Official label IDs preserved contiguously as 0–20 model indices in `experiment_config.json`.
- No train/val/test video or official-path overlap.
- MLP masks verified; padded frames excluded from pooling.

## Metrics

| Model | Val macro F1 | Val acc | Val balanced acc | Test macro F1 | Test acc | Test balanced acc |
|---|---:|---:|---:|---:|---:|---:|
{metric_line('random_forest')}
{metric_line('temporal_mlp')}
{metric_line('dummy')}

Selected by validation macro-F1: `{selected_model}`. Test was evaluated once after configurations were frozen.

## Timing

- Random Forest fit: {rf_fit_seconds:.3f}s
- Temporal MLP fit: {mlp_fit_seconds:.3f}s
- Dummy fit: {dummy_fit_seconds:.3f}s
- Inference latency details: `timing.json`

## Artifacts

- Models: `{model_root}`
- Reports: `{report_root}`
- Per-class metrics: `per_class_metrics.csv`
- Classification reports: `classification_reports.json`
- Confusion matrices: `confusion_matrices/`
- Predictions: `predictions.csv`
- Reproducibility: `reproducibility.json`

## Limitations

- {DUPLICATE_HANDEDNESS_LIMITATION}
- Validation has 2 videos/class and test has 4 videos/class; all metrics are high-variance estimates.
- No signer IDs are available. {CLAIM_BOUNDARY}
- No synthetic data, augmentation, mirroring, feature changes, test tuning, or signer-independent claims were used.
"""
    (report_root / "experiment_summary.md").write_text(summary, encoding="utf-8", newline="\n")

    print(
        json.dumps(
            {
                "experiment_id": "synora_baseline_v1_20261121",
                "model_root": str(model_root),
                "report_root": str(report_root),
                "selected_by_validation": selected_model,
                "validation_metrics": validation_metrics,
                "test_metrics": test_metrics,
                "fit_seconds": timing["fit_seconds"],
                "reload_checks_passed": all(item["reproduced"] for item in reload_checks.values()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
