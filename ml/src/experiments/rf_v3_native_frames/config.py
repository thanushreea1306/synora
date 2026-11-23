from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT_ID = "synora_rf_v3_native_frames_20261123"
VARIANT = "v3_native_frames_cv"
SPLIT = "test"
EXPECTED_TEST_COUNT = 84
EXPECTED_RESAMPLED_TEST_COUNT = 79
SEED = 20261123

DEFAULT_V1_FEATURES = REPO_ROOT / "ml/dataset/processed/synora_include21_video_features_v1"
DEFAULT_V1_LANDMARK_CONFIG = REPO_ROOT / "ml/dataset/processed/synora_include30_landmarks_v1/config.json"
DEFAULT_V1_EXPERIMENT_CONFIG = REPO_ROOT / "ml/reports/synora_baseline_v1_20261121/experiment_config.json"
DEFAULT_V2_FEATURES_ROOT = REPO_ROOT / "ml/dataset/processed/experiments/rf_v2_720p30/features"
DEFAULT_V2_EXTRACTION_ROOT = REPO_ROOT / "ml/dataset/processed/experiments/rf_v2_720p30"
DEFAULT_V4_FEATURES_ROOT = REPO_ROOT / "ml/dataset/processed/experiments/rf_v4_live_alignment/features"
DEFAULT_V4_EXTRACTION_ROOT = REPO_ROOT / "ml/dataset/processed/experiments/rf_v4_live_alignment"
DEFAULT_SELECTED_V3_ROOT = REPO_ROOT / "ml/models/experiments/rf_v4_live_alignment/selected"
DEFAULT_V4_REPORT_ROOT = REPO_ROOT / "ml/reports/experiments/rf_v4_live_alignment"

DEFAULT_DATA_ROOT = REPO_ROOT / "ml/dataset/processed/experiments/rf_v3_native_frames"
DEFAULT_NATIVE_EXTRACTION_ROOT = DEFAULT_DATA_ROOT / "extraction_native"
DEFAULT_NATIVE_FEATURES_ROOT = DEFAULT_DATA_ROOT / "features_native"
DEFAULT_RESAMPLED_EXTRACTION_ROOT = DEFAULT_DATA_ROOT / "extraction_resampled"
DEFAULT_RESAMPLED_FEATURES_ROOT = DEFAULT_DATA_ROOT / "features_resampled"

DEFAULT_MODEL_ROOT = REPO_ROOT / "ml/models/experiments/rf_v3_native_frames"
DEFAULT_REPORT_ROOT = REPO_ROOT / "ml/reports/experiments/rf_v3_native_frames"
