# Code Structure

## Reproducible Entry Point

- `src/experiments/run_reproducible_temperature_study.py`
  - Full reproducible temperature study from original Panasonic `.mat` files.
  - Step 1: rebuilds CSV files and `manifest.csv`.
  - Step 2: runs strict matched 25degC, 10degC, 0degC, and 25degC-to-low-temperature transfer experiments.

## Raw Data Processing

- `src/data_processing/convert_panasonic_mat_to_csv.py`
  - Converts `dataset/dataset_trad/Panasonic 18650PF Data/**/*.mat` into clean CSV files.
  - Writes outputs to `dataset/processed/panasonic_raw_csv/`.
- `src/data_processing/legacy_original_scripts/`
  - Older raw-data checker/converter retained only for traceability.

## Controlled Temperature Experiments

- `src/experiments/shared/temperature_pipeline_contract.py`
  - Documents the controlled constants shared by all temperature lines.
  - Same train/test profile split, downsampling, SOC reference, feature set, sequence length, and model list.
- `src/experiments/temperature_25degC/run_25degC_within_temperature.py`
  - Single-temperature 25degC line.
- `src/experiments/temperature_10degC/run_10degC_within_temperature.py`
  - Single-temperature 10degC line.
- `src/experiments/temperature_0degC/run_0degC_within_temperature.py`
  - Single-temperature 0degC line.
- `src/experiments/temperature_transfer/run_25degC_to_10degC_0degC_transfer.py`
  - Trains at 25degC and evaluates on 25degC, 10degC, and 0degC.
- `src/data/temperature_experiments/run_strict_matched_temperature_pipeline_25C_10C_0C.py`
  - Main implementation used by the reproducible entry point and wrappers.

## Original 25degC Model Scripts

- `src/experiments/temperature_25degC/original_model_scripts/`
  - Original 25degC MLP, LSTM, CNN-LSTM teacher/student/distillation scripts.
  - Kept for auditability; the strict matched temperature runner is the preferred current pipeline.

## Traditional Baselines and Analysis

- `src/experiments/analysis/traditional_baselines/`
  - Coulomb Counting, CC sensitivity, terminal-voltage lookup, and OCV-corrected CC scripts.
- `src/experiments/analysis/traditional_baselines/compare_voltage_lookup_vs_filtered_dynamic_25degC.py`
  - 25degC error-vs-current and error-vs-dynamic-intensity comparison between terminal-voltage lookup, OCV-corrected CC, and filtered-feature models.
- `src/data/traditional_baselines/run_cc_sensitivity_current_noise_25C.py`
  - Current sensor nonideality sensitivity: scale bias, offset, Gaussian noise, drift, and combined cases.
- `src/experiments/analysis/ocv/`
  - OCV-SOC lookup construction and inspection scripts.
- `src/experiments/analysis/paper_outputs/`
  - Paper tables and 25degC figure builders.
- `src/data/temperature_experiments/generate_profile_error_analysis_10C_0C.py`
  - Profile-wise RMSE and error analysis by time, SOC range, current magnitude, and dynamic intensity.

## Deployment-Oriented Validation

- `src/data/deployment_validation/run_mcu_oriented_lightweight_validation_25C.py`
  - Parameter count, model size, estimated FP32/INT8 memory, and CPU latency proxy.

## Key Outputs

- `dataset/processed/panasonic_raw_csv/manifest.csv`
- `dataset/processed/temperature_experiments/strict_matched_temperature_pipeline_25C_10C_0C/strict_matched_test_average.md`
- `dataset/processed/temperature_experiments/strict_matched_temperature_pipeline_25C_10C_0C/strict_matched_transfer_degradation_factors.csv`
- `dataset/processed/temperature_experiments/strict_matched_temperature_pipeline_25C_10C_0C/figures/strict_matched_transfer_rmse_heatmap.png`
- `dataset/processed/traditional_baselines/voltage_lookup_vs_filtered_dynamic_25C/`
- `docs/literature_support_for_traditional_baselines.md`
- `docs/code_study_order.md`
