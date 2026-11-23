# SYNORA RF v2 720p30 Training Report

- Experiment: `synora_rf_v2_720p30_20261122`
- Test split read: no
- Train samples: 305; validation samples: 41
- Excluded over-cap clips: 13

## Validation metrics

| Model | Macro-F1 | Accuracy | Balanced accuracy |
|---|---:|---:|---:|
| frozen_v1_rf_on_v2_val | 0.838095 | 0.853659 | 0.857143 |
| rf_v2 | 0.885714 | 0.902439 | 0.904762 |
| temporal_mlp_v2 | 0.923810 | 0.926829 | 0.928571 |

## Frozen v1 reference

- Original v1 validation macro-F1: 0.852381
- Frozen v1 RF evaluated on v2 validation features: 0.838095

## Selection

- Selected model: `temporal_mlp_v2`
- Rejection threshold: 0.59
- Threshold coverage: 0.902439
- Threshold accepted accuracy: 0.972973

## Weaknesses

- Validation-set size is 41 after excluding one over-cap clip; class-wise estimates are unstable.
- A single signer recorded at a close-range desk setup is not a signer-independent evaluation.
- The frozen v1 model was exposed to v2 metadata values (`30 FPS`, `1280x720`) outside its original acquisition schema; its v2-feature score is diagnostic only.

## Implemented files

- `ml/src/experiments/rf_v2_720p30/dataset.py`
- `ml/src/experiments/rf_v2_720p30/extract_v2.py`
- `ml/src/experiments/rf_v2_720p30/build_features_v2.py`
- `ml/src/experiments/rf_v2_720p30/train_v2.py`
- `ml/src/experiments/rf_v2_720p30/v2_service.py`
- `ml/src/experiments/rf_v2_720p30/recognize.py`
- `ml/tests/experiments/rf_v2_720p30/test_capture.py`
- `ml/tests/experiments/rf_v2_720p30/test_pipeline_artifacts.py`
- `ml/tests/experiments/rf_v2_720p30/test_v2_service.py`
- `ml/tests/experiments/rf_v2_720p30/test_recognize.py`

## Dataset and preprocessing

- Official train/val only; `include_test.txt` was not read and `config.json` records `test_split_read: false`.
- Selected 317 train and 42 val clips; extracted 305 train and 41 val clips.
- Excluded 13 clips where `round(source_duration_s * 30) > 89` and recorded them in `excluded.jsonl`; no silent truncation.
- No failed videos.
- Every decoded frame was spatially resized to `1280x720` with `cv2.INTER_AREA`.
- Temporal resampling used the nearest real source frame at each 30 FPS timestamp; no synthesis, smoothing, mirroring, augmentation, crop, skip, or truncation.
- `NORM_RECT without IMAGE_DIMENSIONS` was observed in MediaPipe output. No v1 code was changed; v2 outputs passed schema, finite-value, shape, and mask checks.

## Matrices and artifacts

- `rf_features.npy`: `(346, 831)`, `float32`, frozen 831-feature order.
- `labels.npy`: `(346,)`, original INCLUDE label IDs.
- `mlp_tensor.npy`: `(346, 89, 135)`, `float32`.
- `mlp_flat.npy`: `(346, 12015)`, exact reshape of `mlp_tensor.npy`.
- `mlp_preprocessing.npz`: mean/std fit on v2 train split only.
- Selected model: `ml/models/experiments/rf_v2_720p30/temporal_mlp/weights.npz`.
- Rejection threshold: `0.59`, derived from validation only.

## Recognition command

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; & "ml/.venv/Scripts/python.exe" -m ml.src.experiments.rf_v2_720p30.recognize
```

- Add `--no-preview` for headless capture.
- Capture requires the camera to read back exactly `1280x720 @ 30 FPS`; no fallback is attempted.
- Predictions below `0.59` raw max probability return `unknown`.
- The CLI does not persist video, frames, landmarks, or prediction traces.

## Verification

- All tests: `Ran 70 tests ... OK`.
- Parity check: `PARITY_SUMMARY records=443 max_abs_diff=0 tolerance=9.9999999999999995e-07`.
- Frozen v1 RF SHA-256 remains `cdfb6673a6499f999c263c8503e2ae1d48ff888165d1bfa478db73c44bab65c5`.
- No commit or push was made.
- No webcam recognition test was run by the implementation agent; the CLI was exercised with mocked capture/service tests only.

## Final assessment

- The best validated v2 model is the framewise temporal MLP, with validation macro-F1 `0.923810`, accuracy `0.926829`, and balanced accuracy `0.928571`.
- v2 RF improves over frozen v1 on the v2 validation features, but the validation set has only 41 clips after exclusion; class-wise estimates remain unstable.
- No signer-independent or unconstrained-environment claim is supported.
