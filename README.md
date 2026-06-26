# Lightweight SOC Estimation for Panasonic 18650PF Cells

This repository contains the code and result artifacts for a master's thesis project on lithium-ion battery state-of-charge (SOC) estimation. The project compares traditional SOC baselines, filtered feature engineering, sequence models, lightweight CNN-LSTM students, and temperature-transfer behavior.

## Research Question

The core problem is not simply whether a more complex model gives lower SOC error. The project asks:

- whether traditional assumptions such as ideal Coulomb Counting, accurate initial SOC, and known capacity are reliable;
- whether terminal voltage remains informative under dynamic current profiles;
- whether filtered temporal features improve robustness under drive-cycle loads;
- whether model accuracy can be preserved after compression into lightweight deployable models;
- whether models trained at 25degC transfer to lower temperatures.

## Repository Layout

The public repository uses `dataset/processed/` for compact, versioned result artifacts. The original Panasonic MAT files and regenerated per-sample CSV files are deliberately excluded.

```text
dataset/processed/                                  Selected tables, figures, and compact result artifacts.
src/data_processing/                                MAT-to-CSV conversion.
src/experiments/                                    Matched 25degC, 10degC, 0degC, and transfer pipelines.
src/data/temperature_experiments/                   Profile-wise and error-analysis scripts.
src/data/deployment_validation/                     Deployment-oriented proxy validation scripts.
src/data/traditional_baselines/                     CC sensitivity and traditional baseline analysis.
scripts/                                            One-command reproducibility runners.
```

## Main Results

Key result tables:

- `dataset/processed/final_paper_tables_25degC/final_data_driven_performance_table_paper.md`
- `dataset/processed/final_paper_tables_25degC/model_complexity_and_compression_table_paper.md`
- `dataset/processed/temperature_experiments/profile_error_analysis_10C_0C/profilewise_metrics.md`
- `dataset/processed/temperature_experiments/strict_matched_temperature_pipeline_25C_10C_0C/strict_matched_test_average.md`
- `dataset/processed/temperature_experiments/strict_matched_temperature_pipeline_25C_10C_0C/strict_matched_profilewise_metrics.md`
- `dataset/processed/temperature_experiments/strict_matched_temperature_pipeline_25C_10C_0C/strict_matched_transfer_degradation_factors.csv`
- `dataset/processed/deployment_validation/mcu_oriented_lightweight_validation_25C/mcu_oriented_lightweight_validation_25C.md`
- `dataset/processed/deployment_validation/minimum_acceptable_mcu_proxy_25C/minimum_acceptable_mcu_proxy_25C.md`
- `dataset/processed/traditional_baselines/cc_current_noise_sensitivity_25C/cc_current_noise_sensitivity_25C_average_by_case.md`

Main figures:

- `dataset/processed/ocv_lookup/ocv_soc_curve_25degC.png`
- `dataset/processed/temperature_experiments/profile_error_analysis_10C_0C/profilewise_rmse_heatmap.png`
- `dataset/processed/temperature_experiments/strict_matched_temperature_pipeline_25C_10C_0C/figures/strict_matched_transfer_rmse_heatmap.png`
- `dataset/processed/traditional_baselines/cc_current_noise_sensitivity_25C/cc_current_noise_average_rmse_by_case.png`

## Results Gallery

All committed result figures are displayed in [the results gallery](docs/RESULTS_GALLERY.md), grouped by experiment and source folder.

## Reproducibility

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the reproducible core pipeline:

```bash
python scripts/run_reproducible_pipeline.py --temperature all
```

This starts from the original Panasonic `.mat` files, rebuilds the processed CSV manifest, runs the strict matched 25degC, 10degC, 0degC, and 25degC-to-low-temperature transfer pipeline, regenerates profile/error-analysis figures, and refreshes the deployment-oriented proxy validation. Subsets can be run with `--temperature 25`, `--temperature 10`, `--temperature 0`, or `--temperature transfer`.

The raw Panasonic data are not included in this repository. See `DATASET.md` for the dataset source, usage conditions, and expected local layout.

## Citation and License

Please cite this repository using `CITATION.cff`. The source code and included documentation are released under the MIT License; the Panasonic dataset remains subject to its original license and must be obtained separately.

## Publishing

This directory is the clean public release tree. Create a new GitHub repository,
copy these contents into it, and commit the files directly. Do not add raw MAT
files, regenerated per-sample CSV files, model checkpoints, thesis drafts, or
reference PDFs.

## Current Limitation

The deployment experiment is MCU-oriented but not yet a physical MCU-board deployment. The minimum acceptable proxy exports the filtered distilled tiny CNN-LSTM as TorchScript plus a per-tensor INT8 weight archive and reports model size, estimated RAM, MACs/FLOPs, and CPU batch-1 latency. It must be described as deployment-oriented proxy validation, not real MCU validation. A strict embedded validation should additionally run the compact model on a real MCU board such as STM32, ESP32, or Arduino Nano 33 BLE.
