# Temperature Experiments

This folder contains the experiments added for the senior-review requirements on temperature robustness and error analysis.

## Scripts

- `run_within_temperature_full_pipeline_10C_0C.py`
  - Runs the full SOC-estimation pipeline at constant 10degC and 0degC.
  - Training files: Cycle_1-Cycle_4 and US06.
  - Test files: UDDS, LA92, and NN.
  - Models: CC, CC sensitivity, instantaneous MLP, filtered-feature MLP, LSTM, filtered CNN-LSTM teacher, tiny student, and distilled tiny student.
  - Output: `dataset/processed/temperature_experiments/within_temperature_full_pipeline_10C_0C/`.

- `generate_profile_error_analysis_10C_0C.py`
  - Generates profile-wise RMSE, error-vs-time, error-vs-SOC-range, error-vs-current-magnitude, and error-vs-dynamic-intensity figures.
  - Input: outputs from `run_within_temperature_full_pipeline_10C_0C.py`.
  - Output: `dataset/processed/temperature_experiments/profile_error_analysis_10C_0C/`.

- `run_temperature_transfer_25C_to_10C_0C.py`
  - Trains models at 25degC and evaluates direct transfer to 10degC and 0degC.
  - Also imports 10degC->10degC and 0degC->0degC within-temperature references.
  - Output: `dataset/processed/temperature_experiments/temperature_transfer_25C_to_10C_0C/`.

## Notes

Only constant-temperature data are selected for these experiments. Temperature-rise profiles are intentionally excluded so the transfer setting remains clean.
