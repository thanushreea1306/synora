# SYNORA v3 Generalisation Report

- Experiment: `synora_rf_v3_generalisation_20261122`
- Test split read: no
- Validation protocol: 5-fold stratified group CV over all 346 train+val clips, grouped by `video_id`
- Signer identity: unavailable; this is not a signer-independent evaluation

## Model comparison on the same CV protocol

| Variant | Macro-F1 | Accuracy | Balanced accuracy | ECE before | ECE after | Threshold |
|---|---:|---:|---:|---:|---:|---:|
| v2_baseline_cv | 0.849886 | 0.852601 | 0.849828 | 0.022225 | 0.027216 | 0.52 |
| regularized_cv | 0.862485 | 0.864162 | 0.862474 | 0.141675 | 0.021916 | 0.54 |
| strong_regularized_cv | 0.888388 | 0.890173 | 0.888697 | 0.221482 | 0.018397 | 0.57 |

## Selected model

- Variant: `strong_regularized_cv`
- OOF macro-F1: 0.888388
- OOF accuracy: 0.890173
- Final training epochs: 55
- Temperature: 0.518920
- Rejection threshold: 0.57
- Accepted accuracy at threshold: 0.923567
- Coverage at threshold: 0.907514

## Limitations

- The original v2 validation set has only 41 clips, 20 classes with 2 clips and narrow with 1.
- CV improves sample support per class but cannot remove shared-signer or shared-setup bias because no signer IDs exist.
- Offline extraction resamples 25 FPS sources to 30 FPS with duplicated source frames; live capture does not duplicate frames.
- Raw confidence is not a calibrated probability; temperature scaling is fitted on out-of-fold train/val predictions only.
- No webcam predictions, test-split metrics, or real-world accuracy claims are made.
