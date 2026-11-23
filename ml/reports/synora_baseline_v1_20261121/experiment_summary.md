# SYNORA Controlled Baseline v1

Generated: 2026-11-21T13:00:41.900346+00:00

This is a controlled isolated-sign baseline on the official 21-class INCLUDE split. It is not evidence of signer-independent generalization.

## Frozen configurations

- Random Forest: 300 trees, Gini, `max_features='sqrt'`, `min_samples_leaf=1`, bootstrap=True, no class weighting, no scaling, seed=20261121.
- Temporal MLP: shared 135→96→48 ReLU frame network, masked mean+max pooling over real frames, 96→64→21 ReLU head, Adam lr=0.001, weight decay=0.0001, batch size 32, max 120 epochs, early stopping patience 20 on validation macro-F1, seed=20261121.
- Dummy: `DummyClassifier(strategy='most_frequent')`, reference only.

## Label and split integrity

- 443 videos, 21 classes, official train/val/test = 317/42/84.
- Official label IDs preserved contiguously as 0–20 model indices in `experiment_config.json`.
- No train/val/test video or official-path overlap.
- MLP masks verified; padded frames excluded from pooling.

## Metrics

| Model | Val macro F1 | Val acc | Val balanced acc | Test macro F1 | Test acc | Test balanced acc |
|---|---:|---:|---:|---:|---:|---:|
| random_forest | 0.8524 | 0.8571 | 0.8571 | 0.8187 | 0.8214 | 0.8214 |
| temporal_mlp | 0.8286 | 0.8333 | 0.8333 | 0.7984 | 0.8095 | 0.8095 |
| dummy | 0.0043 | 0.0476 | 0.0476 | 0.0043 | 0.0476 | 0.0476 |

Selected by validation macro-F1: `random_forest`. Test was evaluated once after configurations were frozen.

## Timing

- Random Forest fit: 4.073s
- Temporal MLP fit: 15.972s
- Dummy fit: 0.002s
- Inference latency details: `timing.json`

## Artifacts

- Models: `C:\Users\thanu\synora\ml\models\synora_baseline_v1_20261121`
- Reports: `C:\Users\thanu\synora\ml\reports\synora_baseline_v1_20261121`
- Per-class metrics: `per_class_metrics.csv`
- Classification reports: `classification_reports.json`
- Confusion matrices: `confusion_matrices/`
- Predictions: `predictions.csv`
- Reproducibility: `reproducibility.json`

## Limitations

- 886 real frames contained duplicate handedness labels; the highest handedness-score hand was retained by the v1 feature preparation. This is a known representation limitation.
- Validation has 2 videos/class and test has 4 videos/class; all metrics are high-variance estimates.
- No signer IDs are available. Controlled isolated-sign recognition on the official INCLUDE split. Signer identity is unverified; no signer-independent generalization claim is made.
- No synthetic data, augmentation, mirroring, feature changes, test tuning, or signer-independent claims were used.
