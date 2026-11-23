# SYNORA RF Feature-Group Ablation v1

Generated: 2026-11-21T13:25:47.747698+00:00

Validation-only experiment. No test set was used for fitting, selection, metrics, or prediction inspection. Existing baseline and v1 feature artifacts were not modified.

## Protocol

- Frozen RF parameters copied from the saved baseline model; random_state `20261121`.
- Official train/validation rows from the saved v1 manifest only; validation size `42`, with `2` videos per class.
- No augmentation, synthetic data, mirroring, feature regeneration, test evaluation, or test-driven selection.
- A descriptive `±0.01` Macro-F1 band is used only for wording; it is not a statistical significance test.

## Baseline reproduction check

- Variant A prediction mismatches vs saved baseline validation predictions: `0` of `42`.
- Saved baseline validation Macro-F1 `0.85238`; reproduced A `0.85238`; delta `+0.000000000000`.
- Saved baseline validation accuracy `0.85714`; reproduced A `0.85714`; delta `+0.000000000000`.
- Saved baseline validation balanced accuracy `0.85714`; reproduced A `0.85714`; delta `+0.000000000000`.
- Exact reproduction: `True`. Comparisons below are interpreted only because this check passed.

## Variants and feature counts

| Variant | Letter | Features | Included groups | Excluded groups | Definition |
|---|---|---:|---|---|---|
| A_full_831 | A | 831 | video_metadata, hand_detection_confidence, hand_landmark_geometry, quality_presence_padding, duplicate_handedness_indicator | none | Full 831-feature RF reproducing the baseline configuration. |
| B_geometry_only | B | 802 | hand_landmark_geometry | video_metadata, hand_detection_confidence, quality_presence_padding, duplicate_handedness_indicator | Normalized landmark-derived geometry only: coordinate summaries and interhand distances computed from normalized landmarks. |
| C_geometry_plus_confidence | C | 812 | hand_landmark_geometry, hand_detection_confidence | video_metadata, quality_presence_padding, duplicate_handedness_indicator | Hand geometry plus hand-detection confidence features. |
| D_mean_coordinates_only | D | 126 | hand_landmark_geometry | video_metadata, hand_detection_confidence, quality_presence_padding, duplicate_handedness_indicator, landmark_std_delta_and_interhand_summaries | Reduced geometry subset: per-landmark normalized mean x/y/z coordinates only; excludes std, delta, and interhand summaries. |

Exact feature-name lists are in `feature_group_manifest.json` and each saved model `config.json`.

## Validation comparison

| Variant | Features | Macro-F1 | Accuracy | Balanced accuracy | Δ Macro-F1 vs A | Δ Accuracy vs A |
|---|---:|---:|---:|---:|---:|---:|
| A_full_831 | 831 | 0.85238 | 0.85714 | 0.85714 | +0.00000 | +0.00000 |
| B_geometry_only | 802 | 0.83991 | 0.85714 | 0.85714 | -0.01247 | +0.00000 |
| C_geometry_plus_confidence | 812 | 0.83492 | 0.85714 | 0.85714 | -0.01746 | +0.00000 |
| D_mean_coordinates_only | 126 | 0.68254 | 0.69048 | 0.69048 | -0.16984 | -0.16667 |

- B_geometry_only: Macro-F1 `0.83991` (`-0.01247` vs A), accuracy `0.85714`, balanced accuracy `0.85714`; below.
- C_geometry_plus_confidence: Macro-F1 `0.83492` (`-0.01746` vs A), accuracy `0.85714`, balanced accuracy `0.85714`; below.
- D_mean_coordinates_only: Macro-F1 `0.68254` (`-0.16984` vs A), accuracy `0.69048`, balanced accuracy `0.69048`; below.

Materiality assessment: removing auxiliary groups did not change validation accuracy (`0.85714` for A, B, and C), while Macro-F1 declined modestly but outside the descriptive 0.01 band for B (`-0.01247`) and C (`-0.01746`). This suggests the auxiliary groups contributed little to overall validation correctness in this split but may have helped per-class balance; the 2-videos-per-class validation set makes that inference unstable. The much larger D decline (`-0.16984` Macro-F1 and `-0.16667` accuracy) indicates that richer landmark summaries—std, delta, and interhand features—were more important to validation performance than metadata/quality groups, without implying causation.

Per-class validation F1, precision, recall, and support are in `per_class_metrics.csv`. Confusion matrices and normalized plots are in `confusion_matrices/`.

## Interpretation

- This experiment tests whether removing auxiliary groups changes validation performance under the frozen RF; it does not prove that the model, dataset, or task causally depends on any feature group.
- Feature groups can be correlated; removing a group can also remove redundant signal or alter tree split behavior.
- Hand geometry was defined from feature names/schema only: coordinate summaries and interhand-distance summaries computed from normalized landmarks. No new grouping was inferred from labels or outcomes.
- Variant D was included because feature names permit a defensible reduced subset: per-landmark normalized mean x/y/z coordinates only.

## Limitations

- Validation has only 42 videos, exactly 2 per class; all metrics and class-level differences are unstable.
- No confidence intervals, repeated splits, signer-stratified analysis, or statistical significance tests were performed.
- Variant comparisons are descriptive and validation-only; no validation-based model selection was used to choose a final model.
- No test metrics or test predictions were used, and no signer-independent claim is made.

## Integrity

- Existing artifact files checked before and after: `54`; unchanged: `True`.
- New outputs were written only under the versioned model/report folders for this experiment.
