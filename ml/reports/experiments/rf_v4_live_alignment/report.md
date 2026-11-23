# SYNORA V4 Live-Alignment Report

- Experiment: `synora_rf_v4_live_alignment_20261123`
- Date: 2026-11-22
- Test split read: no
- Signer identity: unavailable; this is not a signer-independent evaluation
- Outcome: live-alignment preprocessing and capture fixes are implemented and verified, but V3 was selected because all V4 model variants regressed on the shared 333-clip cross-validation protocol.

## What V4 changed

- Offline extraction now matches live acquisition conditions:
  - source frames are not duplicated to fake 30 FPS;
  - 1920x1080 frames are resized once with `cv2.INTER_AREA` to 1280x720;
  - frames use `cv2.COLOR_BGR2RGB`;
  - landmark tracking is reset at recording start;
  - timestamps are generated at 30 FPS with zero-based offsets `0, 33, 67, ...`.
- V4 capture is separate from V2/V3 paths and resets tracking at recording start.
- No webcam frames, recordings, landmarks, or diagnostics are saved.
- Mirroring remains disabled.
- Augmentation stays in training folds only and does not duplicate source clips as new samples.
- V4 features include:
  - frozen 135-channel base tensor matching the V2/V3 builder exactly;
  - 509-channel enhanced tensor with absolute and wrist-relative coordinates, hand scale, orientation, handedness, confidence, validity, inter-hand geometry, motion, speed, and missing-data masks.

## Data and protocol

- V4 extraction completed `346` clips and excluded `13` clips for `BELOW_LIVE_MIN_FRAMES`; there were no extraction failures.
- Identical-sample comparison used `333` clips shared by V2/V3 and V4:
  - V2: `346`
  - V4: `346`
  - shared: `333`
  - V2-only: `13`
  - V4-only: `13`
  - train/val split counts: `293 / 40`
- Validation: 5-fold `StratifiedGroupKFold` grouped by `video_id`, with an inner split for early stopping.
- Calibration and threshold fitting used out-of-fold train+val predictions only.
- Selection data explicitly did not include the test split.

## Model comparison on identical samples

| Variant | Kind | Macro-F1 | Accuracy | Balanced accuracy | ECE before | ECE after | Threshold |
|---|---|---:|---:|---:|---:|---:|---:|
| v3_regularized_cv | v3 | 0.886084 | 0.885886 | 0.884803 | 0.217725 | 0.022213 | 0.53 |
| v4_base_mlp_cv | v4 | 0.828254 | 0.828829 | 0.829333 | 0.187290 | 0.034199 | 0.48 |
| v4_enhanced_tcn_cv | v4 | 0.686699 | 0.690691 | 0.689466 | 0.137301 | 0.035829 | 0.38 |
| v4_enhanced_attention_cv | v4 | 0.614318 | 0.621622 | 0.622347 | 0.076977 | 0.051158 | 0.34 |

Selection score was `oof_macro_f1 - worst_class_regression - ECE_after` against the V3 common baseline:

| Variant | Selection score | Worst-class F1 | Worst-class regression |
|---|---:|---:|---:|
| v3_regularized_cv | 0.885862 | 0.750000 | 0.000000 |
| v4_base_mlp_cv | 0.591801 | 0.592593 | 0.157407 |
| v4_enhanced_tcn_cv | 0.138264 | 0.384615 | 0.365385 |
| v4_enhanced_attention_cv | -0.066749 | 0.296296 | 0.453704 |

## Per-class regressions versus V3 common baseline

Selected model is the V3 common model, so `per_class_before_after.csv` reports zero deltas. The rejected V4 variants regressed as follows:

| Variant | Largest per-class F1 regressions |
|---|---|
| v4_base_mlp_cv | hot -0.162162, narrow -0.157407, wide -0.121212, old -0.121212, warm -0.085389 |
| v4_enhanced_tcn_cv | bad -0.469697, good -0.420467, fast -0.390492, narrow -0.365385, happy -0.293333 |
| v4_enhanced_attention_cv | fast -0.545809, cool -0.489564, hot -0.399046, quiet -0.391304, wet -0.390476 |

## Selected deployment model

- Variant: `v3_regularized_cv`
- Kind: `v3`
- Feature mode: `base`
- Final training samples: `333`
- Final epochs: `61`
- Parameters: `25,285`
- Weight file size: `95,915` bytes
- Final fit time: `745.481` seconds
- OOF macro-F1: `0.886084`
- OOF accuracy: `0.885886`
- OOF balanced accuracy: `0.884803`

Reason: V4 corrected the train/live mismatch, but the V4 representations/models did not improve generalization on identical samples. V3 was therefore deployed through the V4 service/capture wrapper.

## Calibration and rejection

- Temperature: `0.532226`
- NLL before/after temperature scaling: `0.606779 / 0.410551`
- ECE before/after temperature scaling: `0.217725 / 0.022213`
- Rejection threshold: `0.53`
- Minimum coverage constraint: `0.90`
- Coverage at threshold: `0.900901`
- Accepted count: `300`
- Rejected count: `33`
- Accepted accuracy: `0.926667`
- Accepted macro-F1: `0.925777`
- Raw softmax is not presented as a probability of correctness; calibrated confidence is used only after temperature scaling.

## Offline/live parity evidence

`ml/reports/experiments/rf_v4_live_alignment/parity_evidence.json` reports:

- `passed: true`
- preprocessing shapes: offline `(720, 1280, 3)`, live `(720, 1280, 3)`
- max RGB difference: `0`
- resize interpolation: `cv2.INTER_AREA`
- color conversion: `cv2.COLOR_BGR2RGB`
- offline/live landmarker factory count: `1 / 1`
- offline/live landmarker closed: `true / true`
- timestamps: `0, 33, 67`
- tracking lifecycle: `reset_at_recording_start`
- record comparison differences: `[]`

Inference latency was measured separately on 20 CPU-only service calls over one real validation record, excluding capture and feature-building:

- median: `0.005348` s
- mean: `0.005012` s
- min/max: `0.002465 / 0.007403` s

## Artifacts

- Extraction config: `ml/dataset/processed/experiments/rf_v4_live_alignment/config.json`
- Extraction summary: `ml/dataset/processed/experiments/rf_v4_live_alignment/extraction_summary.json`
- V4 features: `ml/dataset/processed/experiments/rf_v4_live_alignment/features/`
- Selected model: `ml/models/experiments/rf_v4_live_alignment/selected/`
- CV results: `ml/reports/experiments/rf_v4_live_alignment/cv_results.json`
- Calibration: `ml/reports/experiments/rf_v4_live_alignment/calibration.json`
- Selection: `ml/reports/experiments/rf_v4_live_alignment/selection.json`
- Per-class deltas: `ml/reports/experiments/rf_v4_live_alignment/per_class_before_after.csv`
- Classification reports: `ml/reports/experiments/rf_v4_live_alignment/classification_reports.json`
- Confusion matrices: `ml/reports/experiments/rf_v4_live_alignment/confusion_matrices.json`
- Training manifest: `ml/reports/experiments/rf_v4_live_alignment/training_manifest.json`
- Parity evidence: `ml/reports/experiments/rf_v4_live_alignment/parity_evidence.json`
- Frozen baseline hashes: `ml/reports/experiments/rf_v4_live_alignment/frozen_baseline_hashes.json`

## Verification

- V4 explicit suite: `Ran 37 tests ... OK`.
- V3 suite: `Ran 29 tests ... OK`.
- Full explicit ML suite including V4 modules: `Ran 145 tests ... OK`.
- Frozen baseline hashes: `34` files checked, `0` missing, `0` mismatched.
- `git status --short`: `?? ml/`; no commit or push was made.

## Limitations

- No test-split metrics and no live webcam accuracy claims are made.
- Offline/live parity verifies preprocessing and landmark-session behavior; it does not validate real-world signing conditions.
- Signer IDs are unavailable, so shared-signer and shared-setup bias cannot be removed.
- The selected model is V3 because V4 regressed on the shared CV protocol.
- Acceptance accuracy applies only to accepted samples above the 0.53 threshold; rejected samples are reported as unknown.
