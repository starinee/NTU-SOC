# Code Structure

This clean release tree contains only the strict matched pipeline code and the
latest strict results used by the thesis.

## Reproducible Entry Point

- `scripts/run_reproducible_pipeline.py`
  - `--temperature all`: refreshes final paper tables and deployment-oriented
    proxy outputs from the committed strict results. If a processed Panasonic
    manifest is available, it also reruns the strict matched training pipeline.
  - `--temperature strict`: reruns the strict matched training pipeline and
    requires `dataset/processed/panasonic_raw_csv/manifest.csv`.
  - `--temperature tables`: regenerates the final 25degC paper tables.
  - `--temperature deployment`: regenerates the deployment-oriented proxy
    outputs.

## Raw Data Processing

- `src/data_processing/convert_panasonic_mat_to_csv.py`
  - Converts raw Panasonic `.mat` files into processed CSV files and writes
    `dataset/processed/panasonic_raw_csv/manifest.csv`.

## Strict Matched Temperature Pipeline

- `src/data/temperature_experiments/run_strict_matched_temperature_pipeline_25C_10C_0C.py`
  - Official strict matched 25degC, 10degC, 0degC, and 25degC-to-low-temperature
    transfer pipeline.
- `src/data/temperature_experiments/run_within_temperature_full_pipeline_10C_0C.py`
  - Shared model, feature, split, metric, and helper implementation used by the
    strict runner.

## Paper Tables

- `src/experiments/analysis/paper_outputs/make_final_paper_tables_25degC.py`
  - Builds final 25degC thesis tables from the strict matched outputs.

## Deployment-Oriented Proxy Validation

- `src/data/deployment_validation/run_mcu_oriented_lightweight_validation_25C.py`
  - Parameter count, model size, estimated FP32/INT8 weight storage, and CPU
    latency proxy.
- `src/data/deployment_validation/run_minimum_mcu_proxy_validation_25C.py`
  - TorchScript export and INT8 weight-storage archive estimate. This is not
    physical MCU deployment and not INT8 runtime inference.

## Official Outputs

- `dataset/processed/temperature_experiments/strict_matched_temperature_pipeline_25C_10C_0C/strict_matched_test_average.csv`
  - Official profile-wise macro-average results used in the thesis tables.
- `dataset/processed/temperature_experiments/strict_matched_temperature_pipeline_25C_10C_0C/strict_matched_pooled_metrics.csv`
  - Pooled metrics computed over concatenated test samples, kept separately for
    reproducibility checks.
- `dataset/processed/final_paper_tables_25degC/`
  - Final paper table outputs.
- `dataset/processed/deployment_validation/`
  - Deployment-oriented proxy outputs.
