# Code Study Order

This guide explains what each main code group does, what model or method it
implements, and why it exists in the thesis workflow.

## 1. Raw Data Processing

**File**

- `src/data_processing/convert_panasonic_mat_to_csv.py`

**What it does**

Converts the original Panasonic `.mat` files from
`dataset/dataset_trad/Panasonic 18650PF Data/` into clean CSV files and writes a
manifest.

**Purpose**

This is the start of the reproducible pipeline. The 25degC, 10degC, and 0degC
experiments all start from the same raw-data conversion logic.

**Main output**

- `dataset/processed/panasonic_raw_csv/manifest.csv`

## 2. Shared Temperature Experiment Contract

**File**

- `src/experiments/shared/temperature_pipeline_contract.py`

**What it does**

Documents the controlled experimental settings shared by all temperature lines:
train/test profiles, downsampling, SOC reference, feature construction, sequence
length, and model list.

**Purpose**

This file makes the temperature comparison defensible. It records that 25degC,
10degC, and 0degC follow the same pipeline so that temperature is the main
changed variable.

## 3. Main Strict Matched Temperature Pipeline

**File**

- `src/data/temperature_experiments/run_strict_matched_temperature_pipeline_25C_10C_0C.py`

**What it does**

Runs the controlled experiments for 25degC, 10degC, and 0degC. It trains and
evaluates the same set of models within each temperature and also evaluates
25degC-trained models directly on 10degC and 0degC.

**Models**

- `Instantaneous MLP`: tabular neural network using instantaneous voltage,
  current, temperature, and power features.
- `Filtered-feature MLP`: lightweight tabular neural network using smoothed
  voltage/current/power and current-change features.
- `LSTM`: sequence model that learns temporal dependency from recent samples.
- `CNN-LSTM Teacher`: larger sequence model with convolutional feature extraction
  followed by LSTM temporal modeling.
- `Tiny CNN-LSTM Student`: smaller sequence model intended for lightweight
  deployment.
- `Distilled Tiny CNN-LSTM`: student model trained with teacher guidance.
- `Filtered CNN-LSTM Teacher/Student/Distilled`: same teacher-student idea, but
  with filtered features.

**Purpose**

This is the core code for model comparison, profile-wise RMSE, and temperature
transfer.

**Main outputs**

- `dataset/processed/temperature_experiments/strict_matched_temperature_pipeline_25C_10C_0C/strict_matched_test_average.md`
- `dataset/processed/temperature_experiments/strict_matched_temperature_pipeline_25C_10C_0C/strict_matched_profilewise_metrics.md`
- `dataset/processed/temperature_experiments/strict_matched_temperature_pipeline_25C_10C_0C/strict_matched_transfer_degradation_factors.csv`

## 4. Convenience Wrappers

**Files**

- `src/experiments/temperature_25degC/run_25degC_within_temperature.py`
- `src/experiments/temperature_10degC/run_10degC_within_temperature.py`
- `src/experiments/temperature_0degC/run_0degC_within_temperature.py`
- `src/experiments/temperature_transfer/run_25degC_to_10degC_0degC_transfer.py`

**What they do**

Run selected subsets of the strict matched pipeline.

**Purpose**

These are easier entry points when you only want one temperature line or only the
25degC-to-low-temperature transfer results.

## 5. Traditional Baselines

**Files**

- `src/experiments/analysis/traditional_baselines/run_coulomb_counting_batch.py`
- `src/experiments/analysis/traditional_baselines/run_coulomb_counting_sensitivity.py`
- `src/experiments/analysis/traditional_baselines/run_ocv_corrected_cc_25degC.py`
- `src/data/traditional_baselines/run_cc_sensitivity_current_noise_25C.py`
- `src/experiments/analysis/traditional_baselines/compare_voltage_lookup_vs_filtered_dynamic_25degC.py`

**Methods**

- `Coulomb Counting`: integrates current over time to estimate SOC.
- `CC sensitivity`: tests how initial SOC error, capacity mismatch, and
  integration assumptions change CC error.
- `Terminal-voltage lookup`: maps loaded terminal voltage directly to SOC using
  the OCV-SOC lookup curve.
- `OCV-corrected CC`: combines CC propagation with OCV-based correction.
- `Current-noise CC sensitivity`: adds sensor bias, offset, Gaussian noise, and
  drift to simulate non-ideal engineering conditions.

**Purpose**

These baselines answer whether traditional assumptions are reliable and why the
learned filtered-feature models are needed under dynamic current.

**Main outputs**

- `dataset/processed/baseline_results_ocv_corrected/`
- `dataset/processed/traditional_baselines/cc_current_noise_sensitivity_25C/`
- `dataset/processed/traditional_baselines/voltage_lookup_vs_filtered_dynamic_25C/`

## 6. OCV-SOC Curve

**File**

- `src/experiments/analysis/ocv/build_ocv_soc_lookup_25degC.py`

**What it does**

Builds the OCV-SOC lookup curve used by terminal-voltage lookup and
OCV-corrected CC.

**Purpose**

This directly supports the required OCV-SOC figure.

**Main outputs**

- `dataset/processed/ocv_lookup/ocv_soc_curve_25degC.png`
- `dataset/processed/ocv_lookup/ocv_soc_curve_25degC_reversed_x.png`

## 7. Error Analysis Figures

**File**

- `src/data/temperature_experiments/generate_profile_error_analysis_10C_0C.py`

**What it does**

Builds profile-wise metrics and error-analysis figures for 10degC and 0degC:
error vs time, error vs SOC range, error vs current magnitude, and error vs
dynamic intensity.

**Purpose**

This answers the requirement to show where the model fails instead of reporting
only average RMSE.

**Main outputs**

- `dataset/processed/temperature_experiments/profile_error_analysis_10C_0C/`

## 8. Deployment-Oriented Lightweight Validation

**File**

- `src/data/deployment_validation/run_mcu_oriented_lightweight_validation_25C.py`

**What it does**

Computes model parameter count, checkpoint size, estimated FP32/INT8 parameter
memory, and MacBook CPU inference latency.

**Purpose**

This is an MCU-oriented proxy validation. It supports the lightweight-deployment
argument, but it is not a physical MCU-board test.

**Main output**

- `dataset/processed/deployment_validation/mcu_oriented_lightweight_validation_25C/mcu_oriented_lightweight_validation_25C.md`

## 9. Paper Output Builders

**Folder**

- `src/experiments/analysis/paper_outputs/`

**What it does**

Builds final tables and 25degC figures for thesis writing.

**Purpose**

These scripts are not the core training pipeline. They are result-packaging
helpers for paper-ready tables and plots.

## 10. One-Command Reproducible Entry Point

**Files**

- `src/experiments/run_reproducible_temperature_study.py`
- `scripts/run_reproducible_pipeline.sh`

**What they do**

Run the whole reproducible study from raw `.mat` conversion through the strict
matched temperature experiments.

**Purpose**

Use these when you want to rebuild the main GitHub-style result package.
