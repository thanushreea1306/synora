from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('C:/Users/thanu/synora')
FEATURES_DIR = ROOT / 'ml/dataset/processed/synora_include21_video_features_v1'
BASELINE_MODEL_DIR = ROOT / 'ml/models/synora_baseline_v1_20261121'
BASELINE_REPORT_DIR = ROOT / 'ml/reports/synora_baseline_v1_20261121'
EXPERIMENT_ID = 'synora_rf_ablation_v1_20261121'
MODEL_ROOT = ROOT / 'ml/models' / EXPERIMENT_ID
REPORT_ROOT = ROOT / 'ml/reports' / EXPERIMENT_ID
MATERIAL_BAND = 0.01


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_selected_csv_lines(path, predicate):
    digest = hashlib.sha256()
    kept = 0
    with path.open('r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        header = ','.join(reader.fieldnames or []) + '\n'
        digest.update(header.encode('utf-8'))
        for row in reader:
            if predicate(row):
                digest.update(','.join(row.get(name, '') for name in reader.fieldnames or []).encode('utf-8') + b'\n')
                kept += 1
    return digest.hexdigest(), kept


def sha256_matrix_rows(path, row_indices):
    matrix = np.load(path, mmap_mode='r')
    selected = np.asarray(matrix[row_indices], dtype=np.float32)
    return sha256_bytes(selected.tobytes())


def write_json(path, value):
    with path.open('w', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')


def package_versions():
    names = ['numpy', 'scikit-learn', 'joblib', 'pandas', 'matplotlib']
    return {name: importlib.metadata.version(name) for name in names}


def feature_group(name):
    if name in {'frame_count', 'duration_s', 'fps', 'width', 'height'}:
        return 'video_metadata'
    if '_conf_' in name:
        return 'hand_detection_confidence'
    if 'coord_' in name or 'interhand_' in name:
        return 'hand_landmark_geometry'
    if name == 'duplicate_handedness_frames':
        return 'duplicate_handedness_indicator'
    return 'quality_presence_padding'


def save_confusion_plot(path, matrix, labels, normalized):
    values = matrix.astype(float)
    if normalized:
        row_sums = values.sum(axis=1, keepdims=True)
        values = np.divide(values, row_sums, out=np.zeros_like(values), where=row_sums != 0)
    fig, axis = plt.subplots(figsize=(12, 10))
    image = axis.imshow(values, interpolation='nearest', cmap='Blues', vmin=0.0, vmax=1.0 if normalized else None)
    axis.set_xticks(np.arange(len(labels)))
    axis.set_yticks(np.arange(len(labels)))
    axis.set_xticklabels(labels, rotation=90, fontsize=7)
    axis.set_yticklabels(labels, fontsize=7)
    axis.set_xlabel('Predicted label')
    axis.set_ylabel('True label')
    axis.set_title('Normalized confusion matrix' if normalized else 'Confusion matrix')
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if MODEL_ROOT.exists() or REPORT_ROOT.exists():
    raise SystemExit('versioned output directory already exists; refusing to overwrite')

existing_artifact_paths = []
for root in [BASELINE_MODEL_DIR, BASELINE_REPORT_DIR, FEATURES_DIR]:
    for path in root.rglob('*'):
        if path.is_file():
            existing_artifact_paths.append(path)
existing_artifact_paths = sorted(existing_artifact_paths)
existing_before = {str(path): {'sha256': sha256_file(path), 'size': path.stat().st_size, 'mtime': path.stat().st_mtime} for path in existing_artifact_paths}

rf_feature_schema = json.load(open(FEATURES_DIR / 'rf_feature_schema.json', encoding='utf-8'))
feature_names = rf_feature_schema['feature_columns']
feature_groups = [feature_group(name) for name in feature_names]
assert len(feature_names) == 831

experiment_config = json.load(open(BASELINE_REPORT_DIR / 'experiment_config.json', encoding='utf-8'))
label_mapping = experiment_config['label_mapping']
label_to_index = {record['class_label']: int(record['model_index']) for record in label_mapping}
index_to_label = {int(record['model_index']): record['class_label'] for record in label_mapping}
labels = [index_to_label[index] for index in range(21)]

train_rows = []
val_rows = []
with (FEATURES_DIR / 'video_manifest.csv').open('r', encoding='utf-8', newline='') as handle:
    reader = csv.DictReader(handle)
    for row in reader:
        split = row['split']
        if split == 'train':
            train_rows.append({
                'sample_index': int(row['sample_index']),
                'video_id': row['video_id'],
                'split': split,
                'class_label': row['class_label'],
                'model_index': label_to_index[row['class_label']],
            })
        elif split == 'val':
            val_rows.append({
                'sample_index': int(row['sample_index']),
                'video_id': row['video_id'],
                'split': split,
                'class_label': row['class_label'],
                'model_index': label_to_index[row['class_label']],
            })
train_rows.sort(key=lambda row: row['sample_index'])
val_rows.sort(key=lambda row: row['sample_index'])
assert len(train_rows) == 317 and len(val_rows) == 42

train_indices = np.array([row['sample_index'] for row in train_rows], dtype=np.int64)
val_indices = np.array([row['sample_index'] for row in val_rows], dtype=np.int64)
assert not set(train_indices).intersection(val_indices)
feature_matrix = np.load(FEATURES_DIR / 'rf_features.npy', mmap_mode='r')
assert feature_matrix.shape == (443, 831)
X_train = np.asarray(feature_matrix[train_indices], dtype=np.float32)
X_val = np.asarray(feature_matrix[val_indices], dtype=np.float32)
y_train = np.array([row['model_index'] for row in train_rows], dtype=np.int64)
y_val = np.array([row['model_index'] for row in val_rows], dtype=np.int64)
assert np.isfinite(X_train).all() and np.isfinite(X_val).all()

baseline_val_predictions_csv = {}
with (BASELINE_REPORT_DIR / 'predictions.csv').open('r', encoding='utf-8', newline='') as handle:
    reader = csv.DictReader(handle)
    for row in reader:
        if row['model'] == 'random_forest' and row['split'] == 'val':
            baseline_val_predictions_csv[int(row['sample_index'])] = int(row['predicted_model_index'])
assert set(baseline_val_predictions_csv) == set(val_indices.tolist())
baseline_val_predictions = np.array([baseline_val_predictions_csv[index] for index in val_indices], dtype=np.int64)

baseline_validation_metrics = json.load(open(BASELINE_REPORT_DIR / 'validation_metrics.json', encoding='utf-8'))['metrics']['random_forest']
baseline_model = joblib.load(BASELINE_MODEL_DIR / 'random_forest/random_forest.joblib')
rf_params = baseline_model.get_params()
saved_model_val_predictions = baseline_model.predict(X_val).astype(np.int64)
assert np.array_equal(saved_model_val_predictions, baseline_val_predictions)

geometry_indices = np.array([i for i, group in enumerate(feature_groups) if group == 'hand_landmark_geometry'], dtype=np.int64)
confidence_indices = np.array([i for i, group in enumerate(feature_groups) if group == 'hand_detection_confidence'], dtype=np.int64)
mean_coordinate_indices = np.array([
    i for i, name in enumerate(feature_names)
    if ('_coord_mean_lm' in name) and name.endswith(('_x', '_y', '_z'))
], dtype=np.int64)
assert len(geometry_indices) == 802
assert len(confidence_indices) == 10
assert len(mean_coordinate_indices) == 126

variant_definitions = {
    'A_full_831': {
        'variant': 'A',
        'description': 'Full 831-feature RF reproducing the baseline configuration.',
        'indices': np.arange(len(feature_names), dtype=np.int64),
        'included_groups': ['video_metadata', 'hand_detection_confidence', 'hand_landmark_geometry', 'quality_presence_padding', 'duplicate_handedness_indicator'],
        'excluded_groups': [],
    },
    'B_geometry_only': {
        'variant': 'B',
        'description': 'Normalized landmark-derived geometry only: coordinate summaries and interhand distances computed from normalized landmarks.',
        'indices': geometry_indices,
        'included_groups': ['hand_landmark_geometry'],
        'excluded_groups': ['video_metadata', 'hand_detection_confidence', 'quality_presence_padding', 'duplicate_handedness_indicator'],
    },
    'C_geometry_plus_confidence': {
        'variant': 'C',
        'description': 'Hand geometry plus hand-detection confidence features.',
        'indices': np.concatenate([geometry_indices, confidence_indices]),
        'included_groups': ['hand_landmark_geometry', 'hand_detection_confidence'],
        'excluded_groups': ['video_metadata', 'quality_presence_padding', 'duplicate_handedness_indicator'],
    },
    'D_mean_coordinates_only': {
        'variant': 'D',
        'description': 'Reduced geometry subset: per-landmark normalized mean x/y/z coordinates only; excludes std, delta, and interhand summaries.',
        'indices': mean_coordinate_indices,
        'included_groups': ['hand_landmark_geometry'],
        'excluded_groups': ['video_metadata', 'hand_detection_confidence', 'quality_presence_padding', 'duplicate_handedness_indicator', 'landmark_std_delta_and_interhand_summaries'],
    },
}

models = {}
results = {}
for name, definition in variant_definitions.items():
    columns = definition['indices']
    model = RandomForestClassifier(**rf_params)
    model.fit(X_train[:, columns], y_train)
    predictions = model.predict(X_val[:, columns]).astype(np.int64)
    metrics = {
        'macro_f1': float(f1_score(y_val, predictions, labels=list(range(21)), average='macro')),
        'accuracy': float(accuracy_score(y_val, predictions)),
        'balanced_accuracy': float(balanced_accuracy_score(y_val, predictions)),
    }
    per_class = {
        'f1': f1_score(y_val, predictions, labels=list(range(21)), average=None),
        'precision': precision_score(y_val, predictions, labels=list(range(21)), average=None, zero_division=0),
        'recall': recall_score(y_val, predictions, labels=list(range(21)), average=None, zero_division=0),
        'support': np.bincount(y_val, minlength=21),
    }
    models[name] = model
    results[name] = {
        'definition': definition,
        'columns': columns,
        'predictions': predictions,
        'metrics': metrics,
        'per_class': per_class,
        'confusion': confusion_matrix(y_val, predictions, labels=list(range(21))),
    }

a_predictions = results['A_full_831']['predictions']
a_metrics = results['A_full_831']['metrics']
prediction_mismatches_vs_saved = int(np.sum(a_predictions != baseline_val_predictions))
metric_deltas = {
    key: float(a_metrics[key] - baseline_validation_metrics[key])
    for key in ['macro_f1', 'accuracy', 'balanced_accuracy']
}
reproduction_ok = prediction_mismatches_vs_saved == 0 and all(abs(value) <= 1e-12 for value in metric_deltas.values())
baseline_reproduction = {
    'reproduced': reproduction_ok,
    'saved_validation_predictions_compared': int(len(baseline_val_predictions)),
    'prediction_mismatches': prediction_mismatches_vs_saved,
    'saved_validation_metrics': baseline_validation_metrics,
    'reproduced_validation_metrics': a_metrics,
    'metric_deltas': metric_deltas,
    'saved_model_val_predictions_match_saved_csv': bool(np.array_equal(saved_model_val_predictions, baseline_val_predictions)),
}
if not reproduction_ok:
    raise RuntimeError(f'variant A did not reproduce baseline: {json.dumps(baseline_reproduction)}')

MODEL_ROOT.mkdir(parents=True)
REPORT_ROOT.mkdir(parents=True)
confusion_dir = REPORT_ROOT / 'confusion_matrices'
confusion_dir.mkdir(parents=True)
script_copy = REPORT_ROOT / 'synora_rf_ablation.py'
script_copy.write_text(Path(__file__).read_text(encoding='utf-8'), encoding='utf-8', newline='\n')

for name, result in results.items():
    variant_dir = MODEL_ROOT / name
    variant_dir.mkdir(parents=True)
    joblib.dump(models[name], variant_dir / 'random_forest.joblib')
    write_json(variant_dir / 'config.json', {
        'experiment_id': EXPERIMENT_ID,
        'variant': name,
        'estimator': 'sklearn.ensemble.RandomForestClassifier',
        'rf_params': rf_params,
        'n_features': int(len(result['columns'])),
        'feature_names': [feature_names[i] for i in result['columns']],
        'feature_groups': [feature_groups[i] for i in result['columns']],
        'seed': rf_params['random_state'],
    })

comparison_rows = []
per_class_rows = []
confusion_payload = {}
validation_prediction_frames = []
for name, result in results.items():
    metrics = result['metrics']
    comparison_rows.append({
        'variant': name,
        'letter': result['definition']['variant'],
        'n_features': int(len(result['columns'])),
        'macro_f1': metrics['macro_f1'],
        'accuracy': metrics['accuracy'],
        'balanced_accuracy': metrics['balanced_accuracy'],
        'delta_macro_f1_vs_A': metrics['macro_f1'] - a_metrics['macro_f1'],
        'delta_accuracy_vs_A': metrics['accuracy'] - a_metrics['accuracy'],
        'delta_balanced_accuracy_vs_A': metrics['balanced_accuracy'] - a_metrics['balanced_accuracy'],
    })
    for index in range(21):
        per_class_rows.append({
            'variant': name,
            'model_index': index,
            'class_label': index_to_label[index],
            'support': int(result['per_class']['support'][index]),
            'precision': float(result['per_class']['precision'][index]),
            'recall': float(result['per_class']['recall'][index]),
            'f1': float(result['per_class']['f1'][index]),
        })
    matrix = result['confusion']
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix.astype(float), row_sums, out=np.zeros(matrix.shape, dtype=float), where=row_sums != 0)
    confusion_payload[name] = {
        'labels': labels,
        'matrix': matrix.tolist(),
        'normalized_matrix': [[round(float(value), 8) for value in row] for row in normalized],
    }
    save_confusion_plot(confusion_dir / f'{name}_val.png', matrix, labels, False)
    save_confusion_plot(confusion_dir / f'{name}_val_normalized.png', matrix, labels, True)
    validation_prediction_frames.append(pd.DataFrame({
        'variant': name,
        'sample_index': val_indices,
        'video_id': [row['video_id'] for row in val_rows],
        'true_model_index': y_val,
        'true_class': [index_to_label[index] for index in y_val],
        'predicted_model_index': result['predictions'],
        'predicted_class': [index_to_label[index] for index in result['predictions']],
        'correct': (y_val == result['predictions']).astype(int),
    }))

comparison_df = pd.DataFrame(comparison_rows)
per_class_df = pd.DataFrame(per_class_rows)
validation_predictions_df = pd.concat(validation_prediction_frames, ignore_index=True)
comparison_df.to_csv(REPORT_ROOT / 'comparison_metrics.csv', index=False, lineterminator='\n')
per_class_df.to_csv(REPORT_ROOT / 'per_class_metrics.csv', index=False, lineterminator='\n')
validation_predictions_df.to_csv(REPORT_ROOT / 'validation_predictions.csv', index=False, lineterminator='\n')
write_json(REPORT_ROOT / 'confusion_matrices.json', confusion_payload)
write_json(REPORT_ROOT / 'baseline_reproduction.json', baseline_reproduction)

feature_manifest = {
    'experiment_id': EXPERIMENT_ID,
    'feature_source': str(FEATURES_DIR),
    'total_features': len(feature_names),
    'group_rules': {
        'video_metadata': ["frame_count", "duration_s", "fps", "width", "height"],
        'hand_detection_confidence': ['feature names containing "_conf_"'],
        'hand_landmark_geometry': ['feature names containing "coord_"', 'feature names containing "interhand_"'],
        'duplicate_handedness_indicator': ['duplicate_handedness_frames'],
        'quality_presence_padding': ['all remaining RF feature columns'],
        'D_mean_coordinates_only': ['feature names containing "_coord_mean_lm" and ending in _x, _y, or _z'],
    },
    'variants': {},
}
for name, result in results.items():
    columns = result['columns']
    feature_manifest['variants'][name] = {
        'variant': result['definition']['variant'],
        'description': result['definition']['description'],
        'feature_count': int(len(columns)),
        'included_groups': result['definition']['included_groups'],
        'excluded_groups': result['definition']['excluded_groups'],
        'feature_names': [feature_names[i] for i in columns],
    }
write_json(REPORT_ROOT / 'feature_group_manifest.json', feature_manifest)

train_val_manifest_hash, train_val_manifest_rows = sha256_selected_csv_lines(
    FEATURES_DIR / 'video_manifest.csv',
    lambda row: row['split'] in {'train', 'val'},
)
val_prediction_hash, val_prediction_rows = sha256_selected_csv_lines(
    BASELINE_REPORT_DIR / 'predictions.csv',
    lambda row: row['model'] == 'random_forest' and row['split'] == 'val',
)
reproducibility = {
    'experiment_id': EXPERIMENT_ID,
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'python_executable': sys.executable,
    'python_version': sys.version,
    'platform': platform.platform(),
    'package_versions': package_versions(),
    'thread_environment': {
        'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS'),
        'MKL_NUM_THREADS': os.environ.get('MKL_NUM_THREADS'),
        'OPENBLAS_NUM_THREADS': os.environ.get('OPENBLAS_NUM_THREADS'),
        'NUMEXPR_NUM_THREADS': os.environ.get('NUMEXPR_NUM_THREADS'),
    },
    'seed': rf_params['random_state'],
    'rf_params': rf_params,
    'reproduction_command': [sys.executable, str(script_copy)],
    'data_policy': {
        'splits_used': ['train', 'val'],
        'test_used_for_fitting': False,
        'test_used_for_selection': False,
        'test_used_for_metrics': False,
        'test_predictions_or_metrics_inspected': False,
        'note': 'Test rows were excluded from all fitted arrays and validation outputs; no test predictions or metrics were read or used.',
    },
    'input_integrity': {
        'rf_feature_schema.json': {'sha256': sha256_file(FEATURES_DIR / 'rf_feature_schema.json')},
        'experiment_config.json': {'sha256': sha256_file(BASELINE_REPORT_DIR / 'experiment_config.json')},
        'validation_metrics.json': {'sha256': sha256_file(BASELINE_REPORT_DIR / 'validation_metrics.json')},
        'random_forest.joblib': {'sha256': sha256_file(BASELINE_MODEL_DIR / 'random_forest/random_forest.joblib')},
        'video_manifest.csv_train_val_rows_only': {'sha256': train_val_manifest_hash, 'rows': train_val_manifest_rows},
        'predictions.csv_random_forest_val_rows_only': {'sha256': val_prediction_hash, 'rows': val_prediction_rows},
        'rf_features.npy_train_val_rows_only': {
            'sha256': sha256_matrix_rows(FEATURES_DIR / 'rf_features.npy', np.concatenate([train_indices, val_indices])),
            'rows': int(len(train_indices) + len(val_indices)),
        },
    },
}
write_json(REPORT_ROOT / 'reproducibility.json', reproducibility)

existing_after = {str(path): {'sha256': sha256_file(path), 'size': path.stat().st_size, 'mtime': path.stat().st_mtime} for path in existing_artifact_paths}
unchanged = existing_before == existing_after
if not unchanged:
    changed = [key for key in existing_before if existing_before[key] != existing_after.get(key)]
    raise RuntimeError(f'existing artifacts changed: {changed}')

variance_lines = []
for row in comparison_df.itertuples():
    if row.letter == 'A':
        continue
    delta = row.delta_macro_f1_vs_A
    descriptor = 'within the descriptive 0.01 Macro-F1 band' if abs(delta) <= MATERIAL_BAND else ('below' if delta < 0 else 'above')
    variance_lines.append(f"- {row.variant}: Macro-F1 `{row.macro_f1:.5f}` (`{delta:+.5f}` vs A), accuracy `{row.accuracy:.5f}`, balanced accuracy `{row.balanced_accuracy:.5f}`; {descriptor}.")

summary_lines = []
summary_lines.append('# SYNORA RF Feature-Group Ablation v1')
summary_lines.append('')
summary_lines.append(f'Generated: {datetime.now(timezone.utc).isoformat()}')
summary_lines.append('')
summary_lines.append('Validation-only experiment. No test set was used for fitting, selection, metrics, or prediction inspection. Existing baseline and v1 feature artifacts were not modified.')
summary_lines.append('')
summary_lines.append('## Protocol')
summary_lines.append('')
summary_lines.append(f"- Frozen RF parameters copied from the saved baseline model; random_state `{rf_params['random_state']}`.")
summary_lines.append('- Official train/validation rows from the saved v1 manifest only; validation size `42`, with `2` videos per class.')
summary_lines.append('- No augmentation, synthetic data, mirroring, feature regeneration, test evaluation, or test-driven selection.')
summary_lines.append(f"- A descriptive `±{MATERIAL_BAND:.2f}` Macro-F1 band is used only for wording; it is not a statistical significance test.")
summary_lines.append('')
summary_lines.append('## Baseline reproduction check')
summary_lines.append('')
summary_lines.append(f"- Variant A prediction mismatches vs saved baseline validation predictions: `{prediction_mismatches_vs_saved}` of `{len(baseline_val_predictions)}`.")
summary_lines.append(f"- Saved baseline validation Macro-F1 `{baseline_validation_metrics['macro_f1']:.5f}`; reproduced A `{a_metrics['macro_f1']:.5f}`; delta `{metric_deltas['macro_f1']:+.12f}`.")
summary_lines.append(f"- Saved baseline validation accuracy `{baseline_validation_metrics['accuracy']:.5f}`; reproduced A `{a_metrics['accuracy']:.5f}`; delta `{metric_deltas['accuracy']:+.12f}`.")
summary_lines.append(f"- Saved baseline validation balanced accuracy `{baseline_validation_metrics['balanced_accuracy']:.5f}`; reproduced A `{a_metrics['balanced_accuracy']:.5f}`; delta `{metric_deltas['balanced_accuracy']:+.12f}`.")
summary_lines.append(f"- Exact reproduction: `{reproduction_ok}`. Comparisons below are interpreted only because this check passed.")
summary_lines.append('')
summary_lines.append('## Variants and feature counts')
summary_lines.append('')
variant_lines = []
for name, result in results.items():
    definition = result['definition']
    variant_lines.append([name, definition['variant'], len(result['columns']), ', '.join(definition['included_groups']), ', '.join(definition['excluded_groups']) if definition['excluded_groups'] else 'none', definition['description']])
summary_lines.append('| Variant | Letter | Features | Included groups | Excluded groups | Definition |')
summary_lines.append('|---|---|---:|---|---|---|')
for row in variant_lines:
    summary_lines.append('| ' + ' | '.join(str(cell) for cell in row) + ' |')
summary_lines.append('')
summary_lines.append('Exact feature-name lists are in `feature_group_manifest.json` and each saved model `config.json`.')
summary_lines.append('')
summary_lines.append('## Validation comparison')
summary_lines.append('')
summary_lines.append('| Variant | Features | Macro-F1 | Accuracy | Balanced accuracy | Δ Macro-F1 vs A | Δ Accuracy vs A |')
summary_lines.append('|---|---:|---:|---:|---:|---:|---:|')
for row in comparison_df.itertuples():
    summary_lines.append(f"| {row.variant} | {row.n_features} | {row.macro_f1:.5f} | {row.accuracy:.5f} | {row.balanced_accuracy:.5f} | {row.delta_macro_f1_vs_A:+.5f} | {row.delta_accuracy_vs_A:+.5f} |")
summary_lines.append('')
summary_lines.extend(variance_lines)
summary_lines.append('')
summary_lines.append('Materiality assessment: removing auxiliary groups did not change validation accuracy (`0.85714` for A, B, and C), while Macro-F1 declined modestly but outside the descriptive 0.01 band for B (`-0.01247`) and C (`-0.01746`). This suggests the auxiliary groups contributed little to overall validation correctness in this split but may have helped per-class balance; the 2-videos-per-class validation set makes that inference unstable. The much larger D decline (`-0.16984` Macro-F1 and `-0.16667` accuracy) indicates that richer landmark summaries—std, delta, and interhand features—were more important to validation performance than metadata/quality groups, without implying causation.')
summary_lines.append('')
summary_lines.append('Per-class validation F1, precision, recall, and support are in `per_class_metrics.csv`. Confusion matrices and normalized plots are in `confusion_matrices/`.')
summary_lines.append('')
summary_lines.append('## Interpretation')
summary_lines.append('')
summary_lines.append('- This experiment tests whether removing auxiliary groups changes validation performance under the frozen RF; it does not prove that the model, dataset, or task causally depends on any feature group.')
summary_lines.append('- Feature groups can be correlated; removing a group can also remove redundant signal or alter tree split behavior.')
summary_lines.append('- Hand geometry was defined from feature names/schema only: coordinate summaries and interhand-distance summaries computed from normalized landmarks. No new grouping was inferred from labels or outcomes.')
summary_lines.append('- Variant D was included because feature names permit a defensible reduced subset: per-landmark normalized mean x/y/z coordinates only.')
summary_lines.append('')
summary_lines.append('## Limitations')
summary_lines.append('')
summary_lines.append('- Validation has only 42 videos, exactly 2 per class; all metrics and class-level differences are unstable.')
summary_lines.append('- No confidence intervals, repeated splits, signer-stratified analysis, or statistical significance tests were performed.')
summary_lines.append('- Variant comparisons are descriptive and validation-only; no validation-based model selection was used to choose a final model.')
summary_lines.append('- No test metrics or test predictions were used, and no signer-independent claim is made.')
summary_lines.append('')
summary_lines.append('## Integrity')
summary_lines.append('')
summary_lines.append(f"- Existing artifact files checked before and after: `{len(existing_artifact_paths)}`; unchanged: `{unchanged}`.")
summary_lines.append('- New outputs were written only under the versioned model/report folders for this experiment.')

with (REPORT_ROOT / 'ablation_summary.md').open('w', encoding='utf-8', newline='\n') as handle:
    handle.write('\n'.join(summary_lines) + '\n')

print(json.dumps({
    'experiment_id': EXPERIMENT_ID,
    'variant_a_reproduced_baseline': reproduction_ok,
    'prediction_mismatches': prediction_mismatches_vs_saved,
    'metrics': {name: results[name]['metrics'] for name in results},
    'feature_counts': {name: int(len(results[name]['columns'])) for name in results},
    'existing_artifacts_unchanged': unchanged,
}, indent=2))
