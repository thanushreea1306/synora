from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

try:
    from ml.src.experiments.rf_v2_720p30 import dataset as v2_dataset
    from ml.src.experiments.rf_v3_native_frames import config
except ModuleNotFoundError as exc:
    if exc.name not in ("ml", "ml.src"):
        raise
    SOURCE_ROOT = str(Path(__file__).resolve().parents[2])
    if SOURCE_ROOT not in sys.path:
        sys.path.insert(0, SOURCE_ROOT)
    from experiments.rf_v2_720p30 import dataset as v2_dataset
    from experiments.rf_v3_native_frames import config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=config.DEFAULT_DATA_ROOT)
    parser.add_argument("--model-root", type=Path, default=config.DEFAULT_MODEL_ROOT)
    parser.add_argument("--report-root", type=Path, default=config.DEFAULT_REPORT_ROOT)
    return parser.parse_args(argv)


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}" if isinstance(value, float) else str(value)
    return str(value)


def metric_row(label: str, native: Mapping[str, Any], baseline: Mapping[str, Any], delta: Mapping[str, Any]) -> str:
    return (
        f"| {label} | {fmt(native.get('accuracy'))} | {fmt(baseline.get('accuracy'))} | "
        f"{fmt(delta.get('accuracy'))} | {fmt(native.get('macro_f1'))} | {fmt(baseline.get('macro_f1'))} | "
        f"{fmt(delta.get('macro_f1'))} |"
    )


def build_report(report_root: Path, data_root: Path, model_root: Path) -> str:
    equivalence = v2_dataset.read_json(data_root / "equivalence.json")
    extraction_summary = v2_dataset.read_json(data_root / "extraction_summary.json")
    training = v2_dataset.read_json(report_root / "training.json")
    training_manifest = v2_dataset.read_json(report_root / "training_manifest.json")
    test_metrics = v2_dataset.read_json(report_root / "test_metrics.json")
    robustness = v2_dataset.read_json(report_root / "robustness.json")
    test_manifest = v2_dataset.read_json(report_root / "test_manifest.json")
    native = test_metrics["native"]
    baseline = test_metrics["baseline_v3_resampled"]
    native_full = test_metrics["native_full_test"]
    native_excluded = test_metrics["native_excluded_from_pairwise"]
    delta = test_metrics["native_minus_baseline"]
    native_rejection = native["rejection"]
    baseline_rejection = baseline["rejection"]

    lines = [
        "# V3 native-frame live-alignment experiment",
        "",
        "## Objective",
        "",
        "Test whether training the existing V3 architecture on the exact V4-native tensor representation improves final hold-out recognition compared with the existing selected V3 model trained on V2 resampled tensors.",
        "",
        "## Data representation",
        "",
        "The native representation uses one MediaPipe detection per real source frame after resizing each frame to 1280x720 with `cv2.INTER_AREA`. It has no frame duplication, resampling, smoothing, or synthesis. The comparison representation is the V2 nearest-real-frame resampling plan at 30 FPS with duplicate source frames where needed.",
        "",
        "All tensors use the frozen 135-channel base feature schema and frozen coordinate mean/std fitted only on the V2 train split. The V3 model applies the same presence masking, padded-frame zeroing, and masked mean/max pooling in both conditions.",
        "",
        "Exact tensor equivalence was required before training:",
        "",
        f"- Existing V4 train/val rows exactly reproducible by the live tensor path: {fmt(equivalence['existing_v4_trainval']['all_exact'])} ({equivalence['existing_v4_trainval']['exact_rows']}/{equivalence['existing_v4_trainval']['rows']})",
        f"- Native test rows exactly reproducible by the live tensor path: {fmt(equivalence['native_test']['all_exact'])} ({equivalence['native_test']['exact_rows']}/{equivalence['native_test']['rows']})",
        f"- Resampled test rows exactly reproducible with the frozen V2 tensor path: {fmt(equivalence['resampled_test']['all_exact'])} ({equivalence['resampled_test']['exact_rows']}/{equivalence['resampled_test']['rows']})",
        "",
        "Test extraction completed:",
        "",
        f"- Official test entries: {extraction_summary['selected_count']}",
        f"- Native complete: {extraction_summary['native_complete']}",
        f"- V2 resampled complete: {extraction_summary['resampled_complete']}",
        "",
        "## Protocol",
        "",
        "- Training and validation used only the 333 common official train+val clips shared with V4: 293 train and 40 validation.",
        "- Candidates and calibration used only grouped 5-fold cross-validation over those 333 clips with `video_id` groups and an inner split for early stopping.",
        "- The official test split was read only to build final-evaluation features and was not used for training, model selection, early stopping, temperature fitting, or threshold choice.",
        f"- Final native model epochs: {training['final_info']['epochs']}",
        f"- Final parameter count: {training['final_info']['parameter_count']}",
        f"- Final native training samples: {training_manifest['final_model_training_samples']}",
        "",
        "## Final test results",
        "",
        f"Pairwise final hold-out sample count: {test_metrics['sample_count']}.",
        f"Native full final hold-out sample count: {test_metrics['native_full_test_sample_count']}.",
        f"Samples excluded from pairwise comparison by the V2 89-frame cap: {test_metrics['excluded_from_pairwise_count']}.",
        "",
        "| Metric | Native | V3 resampled baseline | Native - baseline |",
        "|---|---:|---:|---:|",
        f"| Accuracy | {fmt(native['raw_metrics']['accuracy'])} | {fmt(baseline['raw_metrics']['accuracy'])} | {fmt(delta['accuracy'])} |",
        f"| Macro F1 | {fmt(native['raw_metrics']['macro_f1'])} | {fmt(baseline['raw_metrics']['macro_f1'])} | {fmt(delta['macro_f1'])} |",
        f"| Balanced accuracy | {fmt(native['raw_metrics']['balanced_accuracy'])} | {fmt(baseline['raw_metrics']['balanced_accuracy'])} | {fmt(delta['balanced_accuracy'])} |",
        "",
        "Native-only final test results, including the clips excluded from V2 resampling:",
        "",
        "| Scope | n | Accuracy | Macro F1 | Balanced accuracy |",
        "|---|---:|---:|---:|---:|",
        f"| Native full test | {test_metrics['native_full_test_sample_count']} | {fmt(native_full['raw_metrics']['accuracy'])} | {fmt(native_full['raw_metrics']['macro_f1'])} | {fmt(native_full['raw_metrics']['balanced_accuracy'])} |",
        f"| Native on V2-excluded clips | {test_metrics['excluded_from_pairwise_count']} | {fmt(native_excluded['raw_metrics']['accuracy'])} | {fmt(native_excluded['raw_metrics']['macro_f1'])} | {fmt(native_excluded['raw_metrics']['balanced_accuracy'])} |",
        "",
        "Rejection results use calibration fitted only on out-of-fold train+val predictions:",
        "",
        "| Metric | Native | V3 resampled baseline | Native - baseline |",
        "|---|---:|---:|---:|",
        f"| Temperature | {fmt(native_rejection['temperature'])} | {fmt(baseline_rejection['temperature'])} | n/a |",
        f"| Threshold | {fmt(native_rejection['threshold'])} | {fmt(baseline_rejection['threshold'])} | n/a |",
        f"| Coverage | {fmt(native_rejection['coverage'])} | {fmt(baseline_rejection['coverage'])} | {fmt(delta['rejection_coverage'])} |",
        f"| Accepted accuracy | {fmt(native_rejection['accepted_accuracy'])} | {fmt(baseline_rejection['accepted_accuracy'])} | {fmt(delta['accepted_accuracy'])} |",
        f"| Unknown/rejected count | {native_rejection['rejected_count']} | {baseline_rejection['rejected_count']} | n/a |",
        "",
        "## Robustness summary",
        "",
        "Values below are raw top-1 accuracy and macro-F1 on the final test split. Macro-F1 uses all 21 labels for every bucket.",
        "",
        "| Dimension | Bucket | n | Native acc. | Baseline acc. | Acc. delta | Native F1 | Baseline F1 | F1 delta |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group_name, buckets in robustness["groups"].items():
        for bucket_name, values in buckets.items():
            lines.append(
                f"| {group_name} | {bucket_name} | {values['count']} | "
                f"{fmt(values['native']['accuracy'])} | {fmt(values['baseline']['accuracy'])} | "
                f"{fmt(values['native_minus_baseline']['accuracy'])} | "
                f"{fmt(values['native']['macro_f1'])} | {fmt(values['baseline']['macro_f1'])} | "
                f"{fmt(values['native_minus_baseline']['macro_f1'])} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"The native-frame model changed final hold-out macro-F1 by {fmt(delta['macro_f1'])} and accuracy by {fmt(delta['accuracy'])} relative to the selected V3 resampled model.",
            "This is a final hold-out comparison, not a live-webcam accuracy measurement. Signer identity is unavailable, so `video_id` grouping does not guarantee signer-disjoint validation. No real webcam E2E evaluation was performed.",
            "The result is only meaningful because the offline native tensors were proven byte-for-byte reproducible through the same tensor-building path used by the live V4 service.",
            "",
            "## Reproduction",
            "",
            "```text",
            "PYTHONDONTWRITEBYTECODE=1 ml/.venv/Scripts/python.exe -m ml.src.experiments.rf_v3_native_frames.prepare_test --stage all",
            "PYTHONDONTWRITEBYTECODE=1 ml/.venv/Scripts/python.exe -m ml.src.experiments.rf_v3_native_frames.train",
            "PYTHONDONTWRITEBYTECODE=1 ml/.venv/Scripts/python.exe -m ml.src.experiments.rf_v3_native_frames.evaluate",
            "PYTHONDONTWRITEBYTECODE=1 ml/.venv/Scripts/python.exe -m ml.src.experiments.rf_v3_native_frames.report",
            "```",
            "",
            "Artifact hashes:",
            "",
            f"- Native selected weights: `{test_manifest['input_hashes']['native_selected/weights.npz']}`",
            f"- Baseline selected weights: `{test_manifest['input_hashes']['baseline_selected/weights.npz']}`",
            f"- Native test tensor: `{test_manifest['input_hashes']['native_features/base_tensor.npy']}`",
            f"- Resampled test tensor: `{test_manifest['input_hashes']['resampled_features/mlp_tensor.npy']}`",
            "",
            "## Artifacts",
            "",
            f"- Native test features: `{data_root / 'features_native'}`",
            f"- V2-resampled test features: `{data_root / 'features_resampled'}`",
            f"- Native selected model: `{model_root / 'selected'}`",
            f"- Reports: `{report_root}`",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report_root = args.report_root.resolve()
    report_root.mkdir(parents=True, exist_ok=True)
    report = build_report(report_root, args.data_root.resolve(), args.model_root.resolve())
    path = report_root / "report.md"
    path.write_text(report, encoding="utf-8", newline="\n")
    print(str(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
