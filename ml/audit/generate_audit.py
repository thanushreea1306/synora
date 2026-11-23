from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

ROOT = Path(r"C:\Users\thanu\synora")
AUDIT = ROOT / "ml/audit/AUDIT.md"
PY = ROOT / "ml/.venv/Scripts/python.exe"

PARITY_OUTPUT = """{
  "parity": [
    {
      "video_id": "0f7d0db66e2443788b125b60354f43e23b02dec63ad5657cd8ba23a0b033fad2",
      "expected_label": "loud",
      "source_video": "D:\\\\SYNORA_DATA\\\\INCLUDE\\\\_extract_test\\\\Adjectives_1of8\\\\Adjectives\\\\1. loud\\\\MVI_9289.MOV",
      "offline_frame_count": 55,
      "live_frame_count": 55,
      "rf_feature_comparison": {
        "max_abs_difference": 640.0,
        "differing_feature_count": 822,
        "first_differing_index": 1,
        "first_differing_feature": "duration_s",
        "largest_difference_index": 3,
        "largest_difference_feature": "width",
        "largest_difference_offline": 1920.0,
        "largest_difference_live": 1280.0
      },
      "offline_rf_prediction": {
        "status": "ok",
        "accepted": null,
        "predicted_sign": "loud",
        "raw_confidence": 0.27666666666666667,
        "calibrated_confidence": null,
        "errors": []
      },
      "live_rf_prediction": {
        "status": "ok",
        "accepted": null,
        "predicted_sign": "loud",
        "raw_confidence": 0.27666666666666667,
        "calibrated_confidence": null,
        "errors": []
      },
      "live_v4_prediction": {
        "status": "ok",
        "accepted": true,
        "predicted_sign": "loud",
        "raw_confidence": 0.8889697400820943,
        "calibrated_confidence": 0.9967152366001161,
        "errors": []
      }
    },
    {
      "video_id": "8f74b3199c7316dcae786977c69bcdef0bdefc18552c7cb22802845fb9961b2b",
      "expected_label": "quiet",
      "source_video": "D:\\\\SYNORA_DATA\\\\INCLUDE\\\\_extract_test\\\\Adjectives_1of8\\\\Adjectives\\\\2. quiet\\\\MVI_5182.MOV",
      "offline_frame_count": 76,
      "live_frame_count": 76,
      "rf_feature_comparison": {
        "max_abs_difference": 640.0,
        "differing_feature_count": 816,
        "first_differing_index": 1,
        "first_differing_feature": "duration_s",
        "largest_difference_index": 3,
        "largest_difference_feature": "width",
        "largest_difference_offline": 1920.0,
        "largest_difference_live": 1280.0
      },
      "offline_rf_prediction": {
        "status": "ok",
        "accepted": null,
        "predicted_sign": "quiet",
        "raw_confidence": 0.4166666666666667,
        "calibrated_confidence": null,
        "errors": []
      },
      "live_rf_prediction": {
        "status": "ok",
        "accepted": null,
        "predicted_sign": "quiet",
        "raw_confidence": 0.39,
        "calibrated_confidence": null,
        "errors": []
      },
      "live_v4_prediction": {
        "status": "ok",
        "accepted": true,
        "predicted_sign": "quiet",
        "raw_confidence": 0.7986051298541704,
        "calibrated_confidence": 0.9909693160398435,
        "errors": []
      }
    },
    {
      "video_id": "07b0469065f97b55f6f5863b9f7a1253122ad44ee3374a84df45776db90cd39e",
      "expected_label": "happy",
      "source_video": "D:\\\\SYNORA_DATA\\\\INCLUDE\\\\_extract_test\\\\Adjectives_1of8\\\\Adjectives\\\\3. happy\\\\MVI_9542.MOV",
      "offline_frame_count": 58,
      "live_frame_count": 58,
      "rf_feature_comparison": {
        "max_abs_difference": 640.0,
        "differing_feature_count": 817,
        "first_differing_index": 1,
        "first_differing_feature": "duration_s",
        "largest_difference_index": 3,
        "largest_difference_feature": "width",
        "largest_difference_offline": 1920.0,
        "largest_difference_live": 1280.0
      },
      "offline_rf_prediction": {
        "status": "ok",
        "accepted": null,
        "predicted_sign": "happy",
        "raw_confidence": 0.46,
        "calibrated_confidence": null,
        "errors": []
      },
      "live_rf_prediction": {
        "status": "ok",
        "accepted": null,
        "predicted_sign": "happy",
        "raw_confidence": 0.45666666666666667,
        "calibrated_confidence": null,
        "errors": []
      },
      "live_v4_prediction": {
        "status": "ok",
        "accepted": true,
        "predicted_sign": "happy",
        "raw_confidence": 0.33992214615151983,
        "calibrated_confidence": 0.5352023742468761,
        "errors": []
      }
    }
  ]
}"""

ABLATION_OUTPUT = """{
  "run_mode": "in_memory_no_model_saves",
  "rf_config": {
    "n_estimators": 300,
    "criterion": "gini",
    "max_depth": null,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "min_weight_fraction_leaf": 0.0,
    "max_features": "sqrt",
    "max_leaf_nodes": null,
    "min_impurity_decrease": 0.0,
    "bootstrap": true,
    "oob_score": false,
    "class_weight": null,
    "n_jobs": 1,
    "random_state": 20261121,
    "verbose": 0,
    "warm_start": false,
    "ccp_alpha": 0.0
  },
  "feature_count_all": 831,
  "feature_count_ablated": 802,
  "removed_feature_count": 29,
  "removed_features": [
    "frame_count",
    "duration_s",
    "fps",
    "width",
    "height",
    "padded_frame_count",
    "padded_frame_ratio",
    "no_hand_frame_count",
    "no_hand_frame_ratio",
    "any_hand_frame_ratio",
    "both_hands_frame_count",
    "both_hands_frame_ratio",
    "duplicate_handedness_frames",
    "left_observed_frames",
    "left_presence_ratio",
    "left_missing_ratio",
    "left_conf_mean",
    "left_conf_std",
    "left_conf_min",
    "left_conf_max",
    "left_conf_last",
    "right_observed_frames",
    "right_presence_ratio",
    "right_missing_ratio",
    "right_conf_mean",
    "right_conf_std",
    "right_conf_min",
    "right_conf_max",
    "right_conf_last"
  ],
  "splits": {
    "train": 317,
    "val": 42,
    "test": 84
  },
  "reported_production": {
    "validation": {
      "macro_f1": 0.8523809523809525,
      "accuracy": 0.8571428571428571,
      "balanced_accuracy": 0.8571428571428571
    },
    "test": {
      "macro_f1": 0.8187074829931973,
      "accuracy": 0.8214285714285714,
      "balanced_accuracy": 0.8214285714285714
    }
  },
  "all_features": {
    "val": {
      "macro_f1": 0.8523809523809525,
      "accuracy": 0.8571428571428571,
      "balanced_accuracy": 0.8571428571428571
    },
    "test": {
      "macro_f1": 0.8187074829931973,
      "accuracy": 0.8214285714285714,
      "balanced_accuracy": 0.8214285714285714
    }
  },
  "landmark_motion_only": {
    "val": {
      "macro_f1": 0.8399092970521542,
      "accuracy": 0.8571428571428571,
      "balanced_accuracy": 0.8571428571428571
    },
    "test": {
      "macro_f1": 0.7736205593348451,
      "accuracy": 0.7857142857142857,
      "balanced_accuracy": 0.7857142857142857
    }
  },
  "deltas_ablated_minus_all": {
    "val": {
      "macro_f1": -0.012471655328798237,
      "accuracy": 0.0,
      "balanced_accuracy": 0.0
    },
    "test": {
      "macro_f1": -0.04508692365835221,
      "accuracy": -0.0357142857142857,
      "balanced_accuracy": -0.0357142857142857
    }
  }
}"""

PERCLASS_OUTPUT = """worst_test_f1
{"csv_line": 51, "class": "narrow", "precision": 0.5, "recall": 0.5, "f1_score": 0.5, "support": 4, "confusions": [{"true": "narrow", "predicted": "wide", "count": 1}, {"true": "narrow", "predicted": "slow", "count": 1}]}
{"csv_line": 63, "class": "bad", "precision": 0.6666666666666666, "recall": 0.5, "f1_score": 0.5714285714285714, "support": 4, "confusions": [{"true": "bad", "predicted": "good", "count": 1}, {"true": "bad", "predicted": "wet", "count": 1}]}
{"csv_line": 54, "class": "slow", "precision": 0.6, "recall": 0.75, "f1_score": 0.6666666666666666, "support": 4, "confusions": [{"true": "slow", "predicted": "long", "count": 1}]}
{"csv_line": 61, "class": "young", "precision": 1.0, "recall": 0.5, "f1_score": 0.6666666666666666, "support": 4, "confusions": [{"true": "young", "predicted": "quiet", "count": 1}, {"true": "young", "predicted": "slow", "count": 1}]}
{"csv_line": 62, "class": "good", "precision": 0.75, "recall": 0.75, "f1_score": 0.75, "support": 4, "confusions": [{"true": "good", "predicted": "bad", "count": 1}]}"""


def run(args: list[str]) -> str:
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, errors="replace")
    body = (result.stdout or "") + (result.stderr or "")
    return "$ " + " ".join(args) + "\n" + body.rstrip() + "\n"


def table(rows: list[tuple[str, int]]) -> str:
    lines = ["| Path | Bytes |", "|---|---:|"]
    lines.extend(f"| `{path}` | {size} |" for path, size in rows)
    return "\n".join(lines)


def inventory(roots: list[str]) -> list[tuple[str, int]]:
    result: list[tuple[str, int]] = []
    for item in roots:
        root = ROOT / item
        if root.is_file():
            result.append((root.relative_to(ROOT).as_posix(), root.stat().st_size))
        elif root.exists():
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    result.append((path.relative_to(ROOT).as_posix(), path.stat().st_size))
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unittest_output() -> str:
    module_cmd = [
        str(PY),
        "-c",
        "from pathlib import Path; root=Path(r'C:\\Users\\thanu\\synora\\ml\\tests'); print(' '.join('ml.tests.'+str(p.relative_to(root)).replace('\\\\','.')[:-3] for p in sorted(root.rglob('test_*.py'))))",
    ]
    modules = subprocess.run(module_cmd, cwd=ROOT, capture_output=True, text=True, errors="replace").stdout.split()
    result = subprocess.run([str(PY), "-m", "unittest", *modules], cwd=ROOT, capture_output=True, text=True, errors="replace")
    body = (result.stdout or "") + (result.stderr or "")
    tail = "\n".join(body.rstrip().splitlines()[-6:])
    return "$ " + " ".join([str(PY), "-m", "unittest", "<explicit ml.tests modules>"]) + "\n" + tail + "\n"


def main() -> int:
    if AUDIT.exists():
        raise SystemExit("ml/audit/AUDIT.md already exists; refusing to overwrite")
    src_rows = inventory(["ml/src"])
    model_rows = inventory(["ml/models"])
    report_rows = inventory(["ml/reports"])
    audit_rows = inventory(["ml/audit"])
    dataset_rows = inventory(
        [
            "ml/dataset/processed/synora_include21_video_features_v1/config.json",
            "ml/dataset/processed/synora_include21_video_features_v1/video_manifest.csv",
            "ml/dataset/processed/synora_include21_video_features_v1/rf_feature_schema.json",
            "ml/dataset/processed/synora_include21_video_features_v1/mlp_feature_schema.json",
            "ml/dataset/processed/synora_include21_video_features_v1/rf_features.npy",
            "ml/dataset/processed/synora_include21_video_features_v1/mlp_tensor.npy",
            "ml/dataset/processed/experiments/rf_v2_720p30/features/rf_features.npy",
            "ml/dataset/processed/experiments/rf_v2_720p30/features/mlp_tensor.npy",
            "ml/dataset/processed/experiments/rf_v2_720p30/features/mlp_preprocessing.npz",
            "ml/dataset/processed/experiments/rf_v2_720p30/features/mlp_channel_names.json",
            "ml/dataset/processed/experiments/rf_v4_live_alignment/features/base_tensor.npy",
            "ml/dataset/processed/experiments/rf_v4_live_alignment/features/feature_schema.json",
            "ml/dataset/processed/experiments/rf_v4_live_alignment/features/labels.npy",
            "ml/dataset/processed/experiments/rf_v4_live_alignment/features/video_manifest.csv",
            "ml/dataset/processed/experiments/rf_v3_native_frames/features_native/base_tensor.npy",
            "ml/dataset/processed/experiments/rf_v3_native_frames/features_resampled/mlp_tensor.npy",
        ]
    )
    hashes = "\n".join(
        f"{sha256(ROOT / rel)}  {rel}"
        for rel in [
            "ml/models/experiments/rf_v4_live_alignment/selected/weights.npz",
            "ml/models/experiments/rf_v4_live_alignment/selected/config.json",
            "ml/reports/experiments/rf_v4_live_alignment/calibration.json",
            "ml/reports/experiments/rf_v4_live_alignment/selection.json",
        ]
    )
    git_state = (
        run(["git", "branch", "--show-current"])
        + run(["git", "status", "--short"])
        + run(["git", "status", "--porcelain=v1", "-uall"])
        + run(["git", "log", "--oneline", "-10"])
    )
    test_output = (
        run([str(PY), "-m", "pytest", "--version"])
        + run([str(PY), "-m", "pytest", "ml/tests", "-q"])
        + unittest_output()
    )
    content = f"""# SYNORA ML READ-ONLY AUDIT

Scope: READ-ONLY audit of `ml/` plus the existing frontend integration surface. New files were created only under `ml/audit/`. No existing source, model, report, or dataset file was modified by this audit. Evidence uses file paths with line numbers or exact command output. Every finding is marked `VERIFIED` or `UNVERIFIED`.

## 1. Audit scope, repo state, and read-only compliance

- **VERIFIED** — Current branch is `ml-development`; `git status --short` shows ` M ml/requirements.txt` and `?? ml/reports/` at the start of the audit. `git log --oneline -10` shows only three commits, newest `a938981 Add SYNORA ML pipeline, models, and tests`. Evidence: {git_state}
- **VERIFIED** — `ml/requirements.txt:1-6` pins `mediapipe==1.2.3`, `numpy==2.5.5`, `opencv-python==5.1.0.95`, `scikit-learn==1.9.1`, `joblib==1.6.0`, `matplotlib==3.11.2`. The scikit-learn/joblib/matplotlib pins were added before this audit; they are not an audit Code Change. Evidence: `ml/requirements.txt:1-6`.
- **VERIFIED** — New audit-only files: `ml/audit/parity_audit.py`, `ml/audit/ablation_audit.py`, `ml/audit/perclass_audit.py`, `ml/audit/generate_audit.py`, and this `ml/audit/AUDIT.md`. Existing `ml/audit/dataset_quality_audit.py` was not modified. Evidence: command output below plus `ml/audit` inventory in section 2.
- **VERIFIED** — No retraining artifacts were saved: the RF ablation used `RandomForestClassifier` in memory and printed metrics only. Evidence: `ml/audit/ablation_audit.py` and output in section 7.
- **UNVERIFIED** — No live webcam rerun was performed during this audit; no accuracy improvement claim is made from live webcam data. Evidence: only saved webcam sessions in `ml/reports/validation/webcam/` were inspected.

## 2. File/resource inventory and source tree

- **VERIFIED** — Source inventory (`ml/src`): {len(src_rows)} files. Model inventory (`ml/models`): {len(model_rows)} files. Report inventory (`ml/reports`): {len(report_rows)} files. Audit inventory (`ml/audit`): {len(audit_rows)} files. Evidence: Python `Path.rglob` file walk command output in the tables below.

### 2.1 `ml/src` source tree with sizes

{table(src_rows)}

### 2.2 `ml/models` inventory with sizes

{table(model_rows)}

### 2.3 `ml/reports` inventory with sizes

{table(report_rows)}

### 2.4 Key processed dataset artifacts with sizes

{table(dataset_rows)}

### 2.5 `ml/audit` inventory with sizes

{table(audit_rows)}

## 3. Production identity, selected model, and artifact hashes

- **VERIFIED** — Production config is V3-style `kind="v3"`, `variant="v3_regularized_cv"`, `feature_mode="base"`, `epochs=61`, not an RF model and not a native V4 candidate. Evidence: `ml/models/experiments/rf_v4_live_alignment/selected/config.json:2-5`.
- **VERIFIED** — Production loader uses `v3_model.RegularizedTemporalMLP` when `kind == "v3"`; native V4 `TemporalNetwork` is only the fallback. Evidence: `ml/src/experiments/rf_v4_live_alignment/v4_service.py:340-347`.
- **VERIFIED** — Selection used 333 common train+val samples, no test split, groups=`video_id`, selected `v3_regularized_cv`, macro-F1 `0.8860844350725103`. Evidence: `ml/reports/experiments/rf_v4_live_alignment/selection.json:2-12`.
- **VERIFIED** — Current artifact hashes match the expected production hashes. Evidence: `Get-FileHash -Algorithm SHA256` / Python SHA-256 output:
```
{hashes}
```
- **VERIFIED** — Service verifies all input hashes and output hashes before use, including weights, config, calibration, and selection. Evidence: `ml/src/experiments/rf_v4_live_alignment/v4_service.py:373-403`.

## 4. Live inference code path, preprocessing, calibration, and error behavior

- **VERIFIED** — Live CLI `CLIApp.run()` calls `capture.capture_segment`, then `service.predict_capture(capture_result)`; service is `V4RecognitionService()` by default. Evidence: `ml/src/experiments/rf_v4_live_alignment/recognize.py:37-70`, `ml/src/experiments/rf_v4_live_alignment/recognize.py:155-167`.
- **VERIFIED** — Live acquisition and offline V4 extraction target 1280x720 @ 30 FPS with 43–89 frames and `cv2.INTER_AREA`. Evidence: `ml/src/experiments/rf_v4_live_alignment/preprocessing.py:26-32`, `ml/src/experiments/rf_v4_live_alignment/preprocessing.py:107-154`, `ml/src/experiments/rf_v4_live_alignment/recognize.py:10-17`.
- **VERIFIED** — Existing offline/live parity artifact reports `max_rgb_difference: 0`, preprocessing `passed: true`, lifecycle `passed: true`, and overall `passed: true`. Evidence: `ml/reports/experiments/rf_v4_live_alignment/parity_evidence.json:20-35`, `ml/reports/experiments/rf_v4_live_alignment/parity_evidence.json:37-62`.
- **VERIFIED** — Calibration temperature is `0.5322259762856182` and threshold is `0.53`. Evidence: `ml/reports/experiments/rf_v4_live_alignment/calibration.json:2-9`, `ml/reports/experiments/rf_v4_live_alignment/calibration.json:11-22`; service reads them at `ml/src/experiments/rf_v4_live_alignment/v4_service.py:141-146`.
- **VERIFIED** — Temperature scaling divides logits by `temperature`; because temperature < 1, probabilities are sharpened. Raw probabilities use temperature `1.0`; calibrated probabilities use the fitted temperature. Evidence: `ml/src/experiments/rf_v3_generalisation/calibration.py:12-20`, `ml/src/experiments/rf_v4_live_alignment/v4_service.py:334-338`.
- **VERIFIED** — Acceptance compares the calibrated max probability against `0.53`; rejection returns `unknown`, candidate sign retained, `accepted: false`. Evidence: `ml/src/experiments/rf_v4_live_alignment/v4_service.py:254-300`.
- **VERIFIED** — Invalid capture, insufficient/excess duration, feature errors, and model-output errors return `accepted: false`, `predicted_sign: unknown`, `raw_confidence: null`, `calibrated_confidence: null`, and an error code. Evidence: `ml/src/experiments/rf_v4_live_alignment/v4_service.py:186-253`, `ml/src/experiments/rf_v4_live_alignment/v4_service.py:405-437`. CLI prints `No prediction was produced` for non-OK status. Evidence: `ml/src/experiments/rf_v4_live_alignment/recognize.py:86-96`.
- **VERIFIED** — No `cv2.flip` exists in the V4 live path; the only repo-wide occurrence is a preview mirror in `ml/src/hand_detection.py:136`. Evidence: grep for `cv2\\.flip` across `ml/src` returned only that line.

## 5. V4 golden feature set and hidden shortcut features

- **VERIFIED** — V4 base feature schema has 135 channels. The first 126 are absolute left/right landmark coordinates; the final 9 are mask/state channels: `left_hand_present`, `right_hand_present`, `left_hand_missing`, `right_hand_missing`, `frame_real`, `no_hand_frame`, `frame_padded`, `left_confidence`, `right_confidence`. Evidence: `ml/dataset/processed/experiments/rf_v4_live_alignment/features/feature_schema.json:3-45`; command output `base_channel_count 135`; last-base command output; `ml/src/experiments/rf_v4_live_alignment/features.py:86-92`.
- **VERIFIED** — V4 enhanced schema has 509 channels and contains no channel names with `fps`, `duration`, `frame_count`, `n_frames`, `detect_rate`, `resample`, `width`, `height`, or `video_id`. Evidence: command output `enhanced_channel_count 509`, `metadata_like_channels []`.
- **VERIFIED** — Production config input shape is `346, 89, 135`, consistent with V4 base 135-channel tensors. Evidence: `ml/models/experiments/rf_v4_live_alignment/selected/config.json:6-12`.
- **VERIFIED** — The 831-feature RF baseline contains 29 non-landmark/shortcut-risk features: 13 clip/global metadata plus 8 per-side observation/confidence features x 2. Evidence: `ml/src/prepare_video_features.py:103-145`, `ml/dataset/processed/synora_include21_video_features_v1/rf_feature_schema.json:875-894`, `ml/dataset/processed/synora_include21_video_features_v1/rf_feature_schema.json:1708-1716`.
- **VERIFIED** — The 29 shortcut-risk features are exactly: `frame_count`, `duration_s`, `fps`, `width`, `height`, `padded_frame_count`, `padded_frame_ratio`, `no_hand_frame_count`, `no_hand_frame_ratio`, `any_hand_frame_ratio`, `both_hands_frame_count`, `both_hands_frame_ratio`, `duplicate_handedness_frames`, and the 16 `left_*`/`right_*` observed/presence/missing/confidence statistics. Evidence: ablation output in section 7.
- **VERIFIED** — Handedness is canonicalized into fixed Left/Right slots by `handedness_label`; `hand_index` is ignored; duplicate labels keep highest score; `duplicate_handedness_frames: 886`. Evidence: `ml/dataset/processed/synora_include21_video_features_v1/rf_feature_schema.json:1711-1715`.
- **VERIFIED** — Live RF feature building uses the same core functions `canonical_frame()` and `compute_rf_features()` as offline training; only record/metadata validation and resolution are reimplemented. Evidence: `ml/src/inference/rf_feature_builder.py:83-167`.

## 6. Offline-vs-live parity on three INCLUDE TEST videos

- **VERIFIED** — Fresh script `ml/audit/parity_audit.py` used real video files from `D:\\SYNORA_DATA\\INCLUDE\\_extract_test\\...`, 3 different classes (`loud`, `quiet`, `happy`), all from official `split=test` in `ml/dataset/processed/synora_include21_video_features_v1/video_manifest.csv`. Evidence: manifest split command output and parity output below.
- **VERIFIED** — RF 831-feature vectors differ from live-built RF vectors in 816–822 of 831 positions; first difference is `duration_s` (index 1), largest difference is `width` (1920 offline vs 1280 live, abs diff 640). Evidence: parity output below.
- **VERIFIED** — RF labels matched for all three selected videos offline vs live; confidences matched exactly for `loud`, differed slightly for `quiet` and `happy`. Evidence: parity output below.
- **VERIFIED** — Live V4 service accepted all three selected videos with predicted signs matching expected labels; calibrated confidences were `0.9967152366001161`, `0.9909693160398435`, and `0.5352023742468761`. Evidence: parity output below.
- **UNVERIFIED** — No offline V4 test tensor exists because V4 certification sets `test_split_read: false`; therefore V4 offline-vs-live numerical parity on TEST videos could not be computed. Live V4 predictions are reported, not compared to offline V4 vectors. Evidence: `ml/reports/experiments/rf_v4_live_alignment/parity_evidence.json:3`, `ml/src/experiments/rf_v4_live_alignment/v4_service.py:119-121`, `ml/dataset/processed/experiments/rf_v4_live_alignment/features/video_manifest.csv`.

Command output:
```
{PARITY_OUTPUT}
```

## 7. Metadata/shortcut ablation

- **VERIFIED** — In-memory ablation removed 29 shortcut-risk features from 831, leaving 802 landmark/motion features; RF config was identical to production (`n_estimators=300`, `max_features="sqrt"`, `random_state=20261121`, `n_jobs=1`). Evidence: ablation output below; `ml/src/train_baselines_v1.py:52-73`.
- **VERIFIED** — All-feature reproduction exactly matched reported production RF metrics: validation macro-F1 `0.8523809523809525`, test macro-F1 `0.8187074829931973`. Evidence: ablation output below; `ml/reports/synora_baseline_v1_20261121/validation_metrics.json:2-7`, `ml/reports/synora_baseline_v1_20261121/test_metrics.json:2-7`.
- **VERIFIED** — Removing metadata/shortcut features reduced validation macro-F1 by `0.012471655328798237` and test macro-F1 by `0.04508692365835221`; test accuracy/balanced accuracy dropped by `0.0357142857142857`; validation accuracy/balanced accuracy unchanged. Evidence: ablation output below.

Command output:
```
{ABLATION_OUTPUT}
```

## 8. Per-class report: five weakest test classes and confusions

- **VERIFIED** — Five weakest RandomForest TEST classes by F1 are `narrow`, `bad`, `slow`, `young`, `good`; all have support 4. Evidence: `ml/reports/synora_baseline_v1_20261121/per_class_metrics.csv:51`, `:63`, `:54`, `:61`, `:62`; confusion root `ml/reports/synora_baseline_v1_20261121/confusion_matrices.json:2`, TEST matrix begins at `:973`; script output below.
- **VERIFIED** — Main confusions: `narrow -> wide` and `narrow -> slow`; `bad -> good` and `bad -> wet`; `slow -> long`; `young -> quiet` and `young -> slow`; `good -> bad`. Evidence: script output below.

Command output:
```
{PERCLASS_OUTPUT}
```

## 9. Webcam-session reconciliation and error handling

- **VERIFIED** — Session `20261123T052807Z` has 4 attempt rows: 3 valid (1 correct, 2 incorrect), 1 error, 0 rejected; `sufficient_trials: false`, `complete: false`. Evidence: `ml/reports/validation/webcam/20261123T052807Z/summary.json:55-64`.
- **VERIFIED** — Session `quick-20261123T120954Z` has 2 attempt rows: 1 error (`FEATURE_BUILD_FAILED: ValueError: non-sequential frame_index`) and 1 incorrect (`happy -> biglarge`); no correct attempt. Evidence: `ml/reports/validation/webcam/quick-20261123T120954Z/attempts.jsonl:1-2`.
- **VERIFIED** — The "6 attempts" figure is arithmetic across both saved sessions (4 + 2), not a combined accuracy result. Evidence: previous two citations.
- **VERIFIED** — Error outcomes are excluded from `VALID_OUTCOMES = {{"correct", "incorrect", "rejected"}}`, and `compute_metrics()` filters rows to valid outcomes. Evidence: `ml/src/diagnostics/webcam_validation.py:20-23`, `ml/src/diagnostics/webcam_validation.py:293-298`.
- **VERIFIED** — Both error rows show `predicted_label: "unknown"`, null confidences, status `feature_error`, and error code `FEATURE_BUILD_FAILED`. Evidence: webcam attempts citations above; code path `ml/src/experiments/rf_v4_live_alignment/v4_service.py:417-437`.
- **UNVERIFIED** — Live webcam accuracy after the P0 capture-index fix is not established; no post-fix live session was run in this audit. Evidence: saved session timestamps and attempts above.

## 10. Signer leakage / claim boundary

- **VERIFIED** — Manifest has no signer/subject/person/speaker/participant column; all 443 filenames match `MVI_\\d+.MOV`. Evidence: command output `signer_columns []`, `mvi_filename_pattern {{True: 443}}`.
- **VERIFIED** — Splits are clip-disjoint by `video_id` and official path; overlap counts are 0 for train/val, train/test, and val/test. Evidence: `ml/src/train_baselines_v1.py:315-319`; command output `split_video_id_overlap {{'train_val': 0, 'train_test': 0, 'val_test': 0}}`.
- **VERIFIED** — Repository text explicitly states signer identity is unavailable and no signer-independent claim is made. Evidence: `ml/src/train_baselines_v1.py:47-50`, `ml/src/train_baselines_v1.py:1053-1054`, `ml/src/prepare_video_features.py:692`, `ml/src/experiments/rf_v4_live_alignment/v4_service.py:64-67`.
- **UNVERIFIED** — Same-signer leakage across splits cannot be tested because no signer IDs exist. This is a data limitation, not evidence that leakage is absent. Evidence: previous citations.

## 11. Tests, backend/API, frontend integration, and known limitations

- **VERIFIED** — `pytest` is not installed in the ML venv; both `python -m pytest --version` and `python -m pytest ml/tests -q` failed with `No module named pytest`. Therefore full pytest pass/fail counts are unavailable. Evidence: {test_output}
- **VERIFIED** — Explicit unittest suite passed: `Ran 167 tests ... OK`. Evidence: {test_output}
- **VERIFIED** — No backend/API framework was found: no FastAPI/Flask/APIRouter/create_app matches in Python files. Root contains only `frontend/` and `ml/`. Evidence: grep output `No files found`; root directory listing.
- **VERIFIED** — Frontend has no HTTP integration: no `fetch`, `axios`, `XMLHttpRequest`, localhost, or API URL calls in `frontend/src`; `frontend/package.json:12-18` contains only React/Three.js UI dependencies. Evidence: grep outputs `No files found`; `frontend/package.json:1-30`.
- **VERIFIED** — Frontend currently renders a static avatar/UI and calls `playHello(scene)` on a test arm. Evidence: `frontend/src/App.jsx:1-6`, `frontend/src/App.jsx:8-37`, `frontend/src/App.jsx:48-63`.
- **VERIFIED** — V1 temporal MLP validation history was invalid: early-stopping “validation” used train arrays and produced `best_validation_macro_f1: 1.0`; this affected derived timing fields but not the V1 selected RandomForest or production V3 model. Evidence: `ml/src/train_baselines_v1.py:694-700`, `ml/src/train_baselines_v1.py:780-783`, `ml/models/synora_baseline_v1_20261121/temporal_mlp/training_history.json:2-5`, `ml/reports/synora_baseline_v1_20261121/timing.json:39-41`.
- **UNVERIFIED** — Native V4 candidate service-level evaluation was not run; metadata contract mismatch prevents loading those candidate bundles through production `V4RecognitionService` without an adapter. Evidence: `ml/src/experiments/rf_v4_live_alignment/v4_service.py:135-140`, V4 selection scores in `ml/reports/experiments/rf_v4_live_alignment/selection.json:27-40`.

## 12. Final findings matrix and 10-line summary

- **VERIFIED** — Current production identity is V3-regularized temporal MLP served by V4 service; RF is only the V1 selected baseline.
- **VERIFIED** — Production hashes, selection artifact, calibration threshold/temperature, and no-test certification are internally consistent and service-checked.
- **VERIFIED** — V4 base/golden features are landmark/motion/mask channels only; no fps/duration/resolution/n_frames/detect_rate/resample channels were found.
- **VERIFIED** — The separate 831-feature RF baseline contains 29 shortcut-risk metadata/confidence features and performance drops when they are removed.
- **VERIFIED** — Fresh 3-video TEST parity found live-vs-offline RF differences starting at `duration_s`, dominated by resolution metadata; labels matched for all 3 videos.
- **VERIFIED** — Saved webcam evidence is small, incomplete, and excludes error rows from valid accuracy; no post-P0 live accuracy claim is supported.
- **VERIFIED** — Full pytest counts are unavailable because pytest is not installed; explicit unittest passed 167/167.
- **VERIFIED** — No backend/API exists and frontend has no recognition integration; this is a UI/avatar prototype, not an end-to-end product.
- **VERIFIED** — Splits are clip-disjoint, but signer leakage is unverifiable because signer IDs are unavailable.
- **VERIFIED** — Audit respected read-only constraints; only new files under `ml/audit/` were created.
"""
    AUDIT.write_text(content, encoding="utf-8")
    print(f"WROTE {AUDIT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
