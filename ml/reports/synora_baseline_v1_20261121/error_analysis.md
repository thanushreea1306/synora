# SYNORA Post-Baseline RF Error and Shortcut Analysis

Generated: 2026-11-21T13:18:01.112918+00:00

Scope: read-only analysis of saved predictions, probabilities, RF model, feature schema/manifest, and documented v1 audit records. No retraining, model selection, test-driven tuning, feature regeneration, or modification of existing experiment artifacts was performed.

## Verification

- Verified 46 existing input files before and after analysis by SHA-256, size, and modification time; all hashes and timestamps matched.
- Verified saved RF probabilities are finite, row-normalized, `(84, 21)`, and their argmax exactly matches saved RF test predictions.
- Verified the saved RF model has 831 importances matching the saved 831 feature names.
- The report and CSVs listed at the end are newly created files only.

## 1. Per-class RF test performance

All test classes have support `4`. Values come from the saved classification report and are also written to `rf_test_per_class.csv`.

| F1 rank | Class | Support | Precision | Recall | F1 |
|---|---|---|---|---|---|
| 1 | short | 4 | 1.0000 | 1.0000 | 1.0000 |
| 2 | tall | 4 | 1.0000 | 1.0000 | 1.0000 |
| 3 | fast | 4 | 1.0000 | 1.0000 | 1.0000 |
| 4 | cool | 4 | 1.0000 | 1.0000 | 1.0000 |
| 5 | new | 4 | 1.0000 | 1.0000 | 1.0000 |
| 6 | quiet | 4 | 0.8000 | 1.0000 | 0.8889 |
| 7 | long | 4 | 0.8000 | 1.0000 | 0.8889 |
| 8 | old | 4 | 0.8000 | 1.0000 | 0.8889 |
| 9 | biglarge | 4 | 1.0000 | 0.7500 | 0.8571 |
| 10 | hot | 4 | 1.0000 | 0.7500 | 0.8571 |
| 11 | warm | 4 | 1.0000 | 0.7500 | 0.8571 |
| 12 | happy | 4 | 0.6667 | 1.0000 | 0.8000 |
| 13 | loud | 4 | 0.7500 | 0.7500 | 0.7500 |
| 14 | wide | 4 | 0.7500 | 0.7500 | 0.7500 |
| 15 | smalllittle | 4 | 0.7500 | 0.7500 | 0.7500 |
| 16 | good | 4 | 0.7500 | 0.7500 | 0.7500 |
| 17 | wet | 4 | 0.7500 | 0.7500 | 0.7500 |
| 18 | young | 4 | 1.0000 | 0.5000 | 0.6667 |
| 19 | slow | 4 | 0.6000 | 0.7500 | 0.6667 |
| 20 | bad | 4 | 0.6667 | 0.5000 | 0.5714 |
| 21 | narrow | 4 | 0.5000 | 0.5000 | 0.5000 |

- Strongest by test F1: `short, tall, fast, cool, new` with F1 `1.0000` each.
- Weakest by test F1: `narrow` with F1 `0.5000`.
- With only 4 test videos per class, these class ranks are high-variance and should not be treated as stable class-level capability estimates.

## 2. RF confusion matrix and error pairs

- RF test errors: `15/84` (`0.1786`).
- Distinct directed error pairs: `15`.

| True class | Predicted class | Count | Reverse count | Direction | % of errors |
|---|---|---|---|---|---|
| bad | good | 1 | 1 | reciprocal | 0.0667 |
| bad | wet | 1 | 0 | one-directional | 0.0667 |
| biglarge | loud | 1 | 0 | one-directional | 0.0667 |
| good | bad | 1 | 1 | reciprocal | 0.0667 |
| hot | narrow | 1 | 0 | one-directional | 0.0667 |
| loud | smalllittle | 1 | 0 | one-directional | 0.0667 |
| narrow | slow | 1 | 0 | one-directional | 0.0667 |
| narrow | wide | 1 | 0 | one-directional | 0.0667 |
| slow | long | 1 | 0 | one-directional | 0.0667 |
| smalllittle | narrow | 1 | 0 | one-directional | 0.0667 |
| warm | old | 1 | 0 | one-directional | 0.0667 |
| wet | happy | 1 | 0 | one-directional | 0.0667 |
| wide | happy | 1 | 0 | one-directional | 0.0667 |
| young | quiet | 1 | 0 | one-directional | 0.0667 |
| young | slow | 1 | 0 | one-directional | 0.0667 |

- Reciprocal directed pairs: `2`; one-directional directed pairs: `13`.
- Reciprocal pair set: `bad <-> good`.
- No semantic similarity claim is made: confusion frequency alone does not establish linguistic, visual, or conceptual similarity.
- Counts are based on the saved confusion matrix/predictions only; no raw videos or landmark trajectories were inspected for error causes.

## 3. Prediction confidence

Confidence is the maximum saved RF test probability. This is a post-hoc reliability diagnostic on the same test set; no calibration model was fitted and no independent calibration split existed, so calibrated probabilities are not claimed.

| Statistic | Correct predictions | Incorrect predictions |
|---|---|---|
| n | 69 | 15 |
| mean | 0.4585990338164251 | 0.3191111111111111 |
| std | 0.15185928922704384 | 0.12194499584067406 |
| min | 0.17 | 0.16 |
| p10 | 0.276 | 0.18866666666666665 |
| p25 | 0.33666666666666667 | 0.22833333333333333 |
| median | 0.4533333333333333 | 0.2966666666666667 |
| p75 | 0.5533333333333333 | 0.37833333333333335 |
| p90 | 0.6313333333333334 | 0.49533333333333335 |
| max | 0.8466666666666667 | 0.5666666666666667 |

- Confidence AUC for separating correct from incorrect predictions: `0.7657`.
- 5-bin equal-width expected calibration-error diagnostic: `0.3877` (descriptive only; not a calibration procedure).

Threshold summary:

| Threshold | Predictions >= threshold | Errors >= threshold | Error rate >= threshold |
|---|---|---|---|
| 0.50 | 29 | 2 | 0.0690 |
| 0.70 | 4 | 0 | 0.0000 |
| 0.80 | 3 | 0 | 0.0000 |
| 0.90 | 0 | 0 | 0.0000 |

Reliability bins:

| Confidence bin | n | Mean confidence | Accuracy | Errors |
|---|---|---|---|---|
| [0.0,0.5) | 55 | 0.3439 | 0.7636 | 13 |
| [0.5,0.6) | 19 | 0.5488 | 0.8947 | 2 |
| [0.6,0.7) | 6 | 0.6400 | 1.0000 | 0 |
| [0.7,0.8) | 1 | 0.7667 | 1.0000 | 0 |
| [0.8,0.9) | 3 | 0.8278 | 1.0000 | 0 |
| [0.9,1.0] | 0 | n/a | n/a | 0 |

High-confidence incorrect predictions, defined here as confidence `>= 0.80`: `0`.

None.

Assessment: confidence is directionally associated with correctness (AUC `0.7657`; higher confidence bins have higher empirical accuracy), and errors at confidence `>= 0.70` were `0` in this test set. However, errors occur at confidence `>= 0.50` and the 5-bin ECE diagnostic is `0.3877`, so confidence should not be treated as calibrated or trusted per-sample without a separate calibration study.

## 4. RF feature importance

Importances are saved-tree Gini importances from the frozen RF. They measure split contribution within this fitted forest; they are not causal effects and are biased toward features with many possible split points.

Top 20 features:

| Rank | Feature | Group | Importance | Share % | Cumulative |
|---|---|---|---|---|---|
| 1 | right_coord_min_lm00_x | hand_landmark_geometry | 0.005623 | 0.562 | 0.005623 |
| 2 | right_coord_mean_lm02_x | hand_landmark_geometry | 0.004852 | 0.485 | 0.010475 |
| 3 | right_coord_max_lm19_x | hand_landmark_geometry | 0.004570 | 0.457 | 0.015045 |
| 4 | right_coord_min_lm02_x | hand_landmark_geometry | 0.004181 | 0.418 | 0.019226 |
| 5 | right_coord_std_lm08_x | hand_landmark_geometry | 0.004156 | 0.416 | 0.023382 |
| 6 | interhand_lm00_distance_mean | hand_landmark_geometry | 0.004022 | 0.402 | 0.027404 |
| 7 | right_coord_max_lm07_x | hand_landmark_geometry | 0.004014 | 0.401 | 0.031418 |
| 8 | right_coord_min_lm03_x | hand_landmark_geometry | 0.004013 | 0.401 | 0.035430 |
| 9 | right_conf_mean | hand_detection_confidence | 0.003906 | 0.391 | 0.039336 |
| 10 | right_coord_min_lm06_x | hand_landmark_geometry | 0.003905 | 0.391 | 0.043242 |
| 11 | right_coord_min_lm07_x | hand_landmark_geometry | 0.003882 | 0.388 | 0.047124 |
| 12 | right_coord_std_lm19_x | hand_landmark_geometry | 0.003866 | 0.387 | 0.050990 |
| 13 | right_coord_min_lm16_z | hand_landmark_geometry | 0.003785 | 0.379 | 0.054776 |
| 14 | right_coord_min_lm20_z | hand_landmark_geometry | 0.003758 | 0.376 | 0.058534 |
| 15 | right_coord_mean_lm08_x | hand_landmark_geometry | 0.003681 | 0.368 | 0.062215 |
| 16 | right_coord_max_lm18_x | hand_landmark_geometry | 0.003679 | 0.368 | 0.065894 |
| 17 | right_coord_mean_abs_delta_lm14_y | hand_landmark_geometry | 0.003555 | 0.355 | 0.069449 |
| 18 | right_coord_min_lm19_x | hand_landmark_geometry | 0.003529 | 0.353 | 0.072978 |
| 19 | interhand_lm00_distance_std | hand_landmark_geometry | 0.003496 | 0.350 | 0.076474 |
| 20 | right_coord_max_lm14_x | hand_landmark_geometry | 0.003386 | 0.339 | 0.079860 |

Group totals:

| Group | Features | Total importance | Share % | Mean importance |
|---|---|---|---|---|
| hand_landmark_geometry | 802 | 0.975410 | 97.541 | 0.00121622 |
| hand_detection_confidence | 10 | 0.013406 | 1.341 | 0.00134060 |
| quality_presence_padding | 13 | 0.009639 | 0.964 | 0.00074145 |
| video_metadata | 5 | 0.001170 | 0.117 | 0.00023401 |
| duplicate_handedness_indicator | 1 | 0.000375 | 0.038 | 0.00037525 |

- Hand-landmark geometry contributes `97.541%` of total importance; non-landmark groups contribute `2.459%`.

## 5. Shortcut audit

Shortcut-audit importances are from the same saved RF and are descriptive, not causal.

| Feature | Importance rank | Importance | Share % | Group |
|---|---|---|---|---|
| frame_count | 616 | 0.00057422 | 0.0574 | video_metadata |
| duration_s | 600 | 0.00059582 | 0.0596 | video_metadata |
| fps | 822 | 0.00000000 | 0.0000 | video_metadata |
| width | 819 | 0.00000000 | 0.0000 | video_metadata |
| height | 829 | 0.00000000 | 0.0000 | video_metadata |
| padded_frame_count | 660 | 0.00051994 | 0.0520 | quality_presence_padding |
| padded_frame_ratio | 550 | 0.00068132 | 0.0681 | quality_presence_padding |
| no_hand_frame_count | 409 | 0.00093554 | 0.0936 | quality_presence_padding |
| no_hand_frame_ratio | 259 | 0.00141241 | 0.1412 | quality_presence_padding |
| any_hand_frame_ratio | 649 | 0.00053625 | 0.0536 | quality_presence_padding |
| both_hands_frame_count | 790 | 0.00030380 | 0.0304 | quality_presence_padding |
| both_hands_frame_ratio | 335 | 0.00114411 | 0.1144 | quality_presence_padding |
| left_observed_frames | 759 | 0.00035823 | 0.0358 | quality_presence_padding |
| right_observed_frames | 505 | 0.00076134 | 0.0761 | quality_presence_padding |
| left_presence_ratio | 601 | 0.00059276 | 0.0593 | quality_presence_padding |
| left_missing_ratio | 556 | 0.00067217 | 0.0672 | quality_presence_padding |
| right_presence_ratio | 547 | 0.00068579 | 0.0686 | quality_presence_padding |
| right_missing_ratio | 373 | 0.00103519 | 0.1035 | quality_presence_padding |
| duplicate_handedness_frames | 747 | 0.00037525 | 0.0375 | duplicate_handedness_indicator |

- Combined listed metadata/quality/duplicate-indicator importance: `0.011184` (`1.118%` of total).
- Hand-detection confidence group (10 features) contributes `1.341%` of total importance.
- Possible shortcut risks: frame count/duration/padding may proxy recording length; fps/width/height may proxy capture setup or source batch; presence/no-hand ratios and detection confidences may proxy lighting, signer distance, or detector reliability; duplicate-handedness count may proxy frames where hand selection discarded information.
- These are hypotheses only. The saved artifacts do not contain signer IDs, recording-site labels, camera metadata beyond fps/dimensions, or controlled interventions, so source/capture/signer shortcuts cannot be tested validly here.

## 6. Duplicate-handedness limitation

- Saved v1 records document `886` real frames with duplicate handedness labels; policy kept the highest-handedness-score hand and counted extras, without changing the policy here.
- Documented frame counts by split: `train=652`, `val=82`, `test=152`.
- From the saved manifest, `291/443` videos have at least one duplicate-handedness frame; by split: `train=213`, `val=28`, `test=50`; affecting `21` classes.

| Duplicate-affected test videos | Videos | Errors | Error rate | Mean duplicate frames |
|---|---|---|---|---|
| False | 34 | 7 | 0.2059 | 0.000 |
| True | 50 | 8 | 0.1600 | 3.040 |

- `duplicate_handedness_frames` feature importance: `0.00037525` (rank `747`, `0.0375%` of total).
- The affected/unaffected error-rate comparison is observational and class-imbalanced; it is not evidence that duplicate-handedness caused errors. No features were regenerated and no hand-selection rule was changed.

## 7. Limits and analyses not validly performable

- No causal interpretation of feature importance or error pairs is valid from these artifacts.
- No signer-independent, demographic, or recording-site analysis is possible because those labels are unavailable.
- No per-frame visual/trajectory error diagnosis was performed; only saved tabular features, predictions, probabilities, and documented audit summaries were used.
- Calibration was not fitted on a separate calibration split; the confidence diagnostics are descriptive only.
- Per-class test support is 4, so per-class rankings and affected/unaffected subsets are high-variance.

## Files created

- `ml/reports/synora_baseline_v1_20261121/error_analysis.md`
- `ml/reports/synora_baseline_v1_20261121/input_verification_hashes.csv`
- `ml/reports/synora_baseline_v1_20261121/rf_feature_group_importance.csv`
- `ml/reports/synora_baseline_v1_20261121/rf_feature_importance.csv`
- `ml/reports/synora_baseline_v1_20261121/rf_high_confidence_errors.csv`
- `ml/reports/synora_baseline_v1_20261121/rf_test_confusion_pairs.csv`
- `ml/reports/synora_baseline_v1_20261121/rf_test_per_class.csv`
- `ml/reports/synora_baseline_v1_20261121/rf_test_prediction_confidence.csv`

No existing model, report, prediction, probability, dataset, or feature file was modified.
