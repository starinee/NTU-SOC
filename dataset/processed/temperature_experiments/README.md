# Temperature Experiments Outputs

This folder groups all temperature-related SOC-estimation outputs added for the senior-review requirements.

## Folders

- `within_temperature_full_pipeline_10C_0C/`
  - Full pipeline reruns at constant 10degC and 0degC.
  - Includes CC, CC sensitivity, MLP, filtered-feature MLP, LSTM, CNN-LSTM teacher, tiny student, and distilled tiny student.

- `profile_error_analysis_10C_0C/`
  - Profile-wise RMSE.
  - Error vs time.
  - Error vs SOC range.
  - Error vs current magnitude.
  - Error vs dynamic intensity.

- `temperature_transfer_25C_to_10C_0C/`
  - 25degC-trained models tested on 25degC, 10degC, and 0degC.
  - Also includes within-temperature 10degC->10degC and 0degC->0degC references.

## Key Tables

- `within_temperature_full_pipeline_10C_0C/all_temperatures_test_average.md`
- `profile_error_analysis_10C_0C/profilewise_metrics.md`
- `profile_error_analysis_10C_0C/error_by_soc_range.md`
- `profile_error_analysis_10C_0C/error_by_current_bin.md`
- `profile_error_analysis_10C_0C/error_by_dynamic_bin.md`
- `temperature_transfer_25C_to_10C_0C/temperature_transfer_test_average.md`
- `temperature_transfer_25C_to_10C_0C/temperature_transfer_profilewise_metrics.md`
