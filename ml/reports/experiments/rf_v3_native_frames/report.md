# V3 native-frame live-alignment experiment

## Objective

Test whether training the existing V3 architecture on the exact V4-native tensor representation improves final hold-out recognition compared with the existing selected V3 model trained on V2 resampled tensors.

## Data representation

The native representation uses one MediaPipe detection per real source frame after resizing each frame to 1280x720 with `cv2.INTER_AREA`. It has no frame duplication, resampling, smoothing, or synthesis. The comparison representation is the V2 nearest-real-frame resampling plan at 30 FPS with duplicate source frames where needed.

All tensors use the frozen 135-channel base feature schema and frozen coordinate mean/std fitted only on the V2 train split. The V3 model applies the same presence masking, padded-frame zeroing, and masked mean/max pooling in both conditions.

Exact tensor equivalence was required before training:

- Existing V4 train/val rows exactly reproducible by the live tensor path: yes (346/346)
- Native test rows exactly reproducible by the live tensor path: yes (84/84)
- Resampled test rows exactly reproducible with the frozen V2 tensor path: yes (79/79)

Test extraction completed:

- Official test entries: 84
- Native complete: 84
- V2 resampled complete: 79

## Protocol

- Training and validation used only the 333 common official train+val clips shared with V4: 293 train and 40 validation.
- Candidates and calibration used only grouped 5-fold cross-validation over those 333 clips with `video_id` groups and an inner split for early stopping.
- The official test split was read only to build final-evaluation features and was not used for training, model selection, early stopping, temperature fitting, or threshold choice.
- Final native model epochs: 47
- Final parameter count: 25285
- Final native training samples: 333

## Final test results

Pairwise final hold-out sample count: 79.
Native full final hold-out sample count: 84.
Samples excluded from pairwise comparison by the V2 89-frame cap: 5.

| Metric | Native | V3 resampled baseline | Native - baseline |
|---|---:|---:|---:|
| Accuracy | 0.911392 | 0.898734 | 0.012658 |
| Macro F1 | 0.899711 | 0.891156 | 0.008555 |
| Balanced accuracy | 0.904762 | 0.896825 | 0.007937 |

Native-only final test results, including the clips excluded from V2 resampling:

| Scope | n | Accuracy | Macro F1 | Balanced accuracy |
|---|---:|---:|---:|---:|
| Native full test | 84 | 0.892857 | 0.883296 | 0.892857 |
| Native on V2-excluded clips | 5 | 0.600000 | 0.428571 | 0.600000 |

Rejection results use calibration fitted only on out-of-fold train+val predictions:

| Metric | Native | V3 resampled baseline | Native - baseline |
|---|---:|---:|---:|
| Temperature | 0.534129 | 0.532226 | n/a |
| Threshold | 0.560000 | 0.530000 | n/a |
| Coverage | 0.924051 | 0.936709 | -0.012658 |
| Accepted accuracy | 0.972603 | 0.932432 | 0.040170 |
| Unknown/rejected count | 6 | 5 | n/a |

## Robustness summary

Values below are raw top-1 accuracy and macro-F1 on the final test split. Macro-F1 uses all 21 labels for every bucket.

| Dimension | Bucket | n | Native acc. | Baseline acc. | Acc. delta | Native F1 | Baseline F1 | F1 delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| missing_hands | any_no_hand | 46 | 0.847826 | 0.869565 | -0.021739 | 0.587542 | 0.604233 | -0.016691 |
| missing_hands | no_missing_hand | 33 | 1.000000 | 0.939394 | 0.060606 | 0.523810 | 0.493878 | 0.029932 |
| two_hand_dominant | both_hands_majority | 11 | 1.000000 | 0.909091 | 0.090909 | 0.333333 | 0.317460 | 0.015873 |
| two_hand_dominant | single_or_absent | 68 | 0.897059 | 0.897059 | 0.000000 | 0.886243 | 0.869992 | 0.016251 |
| low_confidence | mean_confidence_below_0_8 | 33 | 0.787879 | 0.818182 | -0.030303 | 0.567952 | 0.592063 | -0.024112 |
| low_confidence | mean_confidence_at_least_0_8 | 46 | 1.000000 | 0.956522 | 0.043478 | 0.857143 | 0.827211 | 0.029932 |
| duration | short_43_49 | 23 | 0.869565 | 0.869565 | 0.000000 | 0.698413 | 0.730159 | -0.031746 |
| duration | mid_50_69 | 53 | 0.924528 | 0.924528 | 0.000000 | 0.842404 | 0.838095 | 0.004308 |
| duration | long_70_89 | 3 | 1.000000 | 0.666667 | 0.333333 | 0.142857 | 0.095238 | 0.047619 |
| boundary | exact_43 | 2 | 0.500000 | 0.500000 | 0.000000 | 0.047619 | 0.047619 | 0.000000 |
| boundary | near_cap_80_plus | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| boundary | interior | 77 | 0.922078 | 0.909091 | 0.012987 | 0.909524 | 0.898753 | 0.010771 |
| padded_ratio | below_70_percent_real | 69 | 0.898551 | 0.898551 | 0.000000 | 0.884354 | 0.891081 | -0.006727 |
| padded_ratio | at_least_70_percent_real | 10 | 1.000000 | 0.900000 | 0.100000 | 0.380952 | 0.333333 | 0.047619 |

## Interpretation

The native-frame model changed final hold-out macro-F1 by 0.008555 and accuracy by 0.012658 relative to the selected V3 resampled model.
This is a final hold-out comparison, not a live-webcam accuracy measurement. Signer identity is unavailable, so `video_id` grouping does not guarantee signer-disjoint validation. No real webcam E2E evaluation was performed.
The result is only meaningful because the offline native tensors were proven byte-for-byte reproducible through the same tensor-building path used by the live V4 service.

## Reproduction

```text
PYTHONDONTWRITEBYTECODE=1 ml/.venv/Scripts/python.exe -m ml.src.experiments.rf_v3_native_frames.prepare_test --stage all
PYTHONDONTWRITEBYTECODE=1 ml/.venv/Scripts/python.exe -m ml.src.experiments.rf_v3_native_frames.train
PYTHONDONTWRITEBYTECODE=1 ml/.venv/Scripts/python.exe -m ml.src.experiments.rf_v3_native_frames.evaluate
PYTHONDONTWRITEBYTECODE=1 ml/.venv/Scripts/python.exe -m ml.src.experiments.rf_v3_native_frames.report
```

Artifact hashes:

- Native selected weights: `bc8cac0e7b39feb64fec8511233eea21dbf6249be40875e14f22f0be930ff159`
- Baseline selected weights: `8225eacdca2ed1b4208bad3fce0463b8e8478e0845f68c1ebd21509413dabf61`
- Native test tensor: `202bf969714dc2eb9095b8072d45093b9ea1ce2ddaa2b5530bf11fbd96361c34`
- Resampled test tensor: `d5f11c838f112cf8e2b07cac799d770076a61ecb81205ddaa9924d7cb763bb42`

## Artifacts

- Native test features: `C:\Users\thanu\synora\ml\dataset\processed\experiments\rf_v3_native_frames\features_native`
- V2-resampled test features: `C:\Users\thanu\synora\ml\dataset\processed\experiments\rf_v3_native_frames\features_resampled`
- Native selected model: `C:\Users\thanu\synora\ml\models\experiments\rf_v3_native_frames\selected`
- Reports: `C:\Users\thanu\synora\ml\reports\experiments\rf_v3_native_frames`
