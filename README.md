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

The final/public repository should be read using relative paths rather than local machine paths. In the thesis text these folders are described as `data/raw/`, `data/processed/`, `src/`, `scripts/`, `results/`, and `figures/`. In this working copy, generated result folders are stored under `dataset/processed/`.

```text
data/raw/                                           Original Panasonic MAT data, not uploaded publicly.
data/processed/                                     Rebuilt CSV files, tables, figures, and compact result artifacts.
src/data_processing/                                MAT-to-CSV conversion.
src/experiments/analysis/paper_outputs/             Final paper table generation.
src/data/temperature_experiments/                   Strict matched temperature pipeline.
src/data/deployment_validation/                     Deployment-oriented proxy validation scripts.
scripts/                                            One-command strict release reproducibility helper.
results/                                            Recommended public alias for generated tables.
figures/                                            Recommended public alias for generated figures.
```

## Main Results

The official result source for the thesis is the strict matched temperature
pipeline:

- `dataset/processed/temperature_experiments/strict_matched_temperature_pipeline_25C_10C_0C/strict_matched_test_average.csv`
- `dataset/processed/temperature_experiments/strict_matched_temperature_pipeline_25C_10C_0C/strict_matched_profilewise_metrics.csv`
- `dataset/processed/temperature_experiments/strict_matched_temperature_pipeline_25C_10C_0C/strict_matched_pooled_metrics.csv`

`strict_matched_test_average.csv` reports the unweighted mean of profile-wise
test metrics over UDDS, LA92, and NN. Metrics computed over concatenated test
samples are kept in the separate pooled metrics file to avoid mixing aggregation
definitions.

Key result tables:

- `dataset/processed/final_paper_tables_25degC/final_data_driven_performance_table_paper.md`
- `dataset/processed/final_paper_tables_25degC/model_complexity_and_compression_table_paper.md`
- `dataset/processed/temperature_experiments/strict_matched_temperature_pipeline_25C_10C_0C/strict_matched_test_average.md`
- `dataset/processed/temperature_experiments/strict_matched_temperature_pipeline_25C_10C_0C/strict_matched_profilewise_metrics.md`
- `dataset/processed/temperature_experiments/strict_matched_temperature_pipeline_25C_10C_0C/strict_matched_transfer_degradation_factors.csv`
- `dataset/processed/deployment_validation/mcu_oriented_lightweight_validation_25C/mcu_oriented_lightweight_validation_25C.md`
- `dataset/processed/deployment_validation/minimum_acceptable_mcu_proxy_25C/minimum_acceptable_mcu_proxy_25C.md`

Main figures:

- `dataset/processed/temperature_experiments/strict_matched_temperature_pipeline_25C_10C_0C/figures/strict_matched_transfer_rmse_heatmap.png`

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

If the processed Panasonic manifest is available, this reruns the strict
matched 25degC, 10degC, 0degC, and 25degC-to-low-temperature transfer pipeline.
If the manifest is not available, it uses the committed strict outputs to
refresh the final paper tables. It does not overwrite the committed
deployment-oriented CPU latency proxy outputs, because CPU latency is
machine-dependent. To force a full retraining run after preparing the dataset,
use:

```bash
python scripts/run_reproducible_pipeline.py --temperature strict
```

Other supported modes are `--temperature tables` and `--temperature deployment`.
Use `--temperature deployment` only when you intentionally want to refresh the
machine-dependent deployment proxy CSV files.

The raw Panasonic data are not included in this repository. See `DATASET.md` for the expected local dataset layout.

## Preparing a Clean GitHub Upload

Do not upload this working folder directly. It contains local thesis drafts,
reference PDFs, raw data, regenerated CSV dumps, and a virtual environment. To
create a clean upload folder, run:

```bash
bash scripts/create_github_release_tree.sh
```

This creates `github_release/`, excluding local-only files and large generated
artifacts. See `GITHUB_UPLOAD_GUIDE.md` for details.

## Current Limitation

The deployment experiment is a deployment-oriented proxy validation, not a
physical MCU-board deployment. The minimum acceptable proxy exports the filtered
distilled tiny CNN-LSTM as TorchScript plus a per-tensor INT8 weight-storage
archive and reports model size, estimated RAM, MACs/FLOPs, and CPU batch-1
latency. It is not INT8 runtime inference and must not be described as real MCU
validation. A strict embedded validation should additionally run the compact
model on a real MCU board such as STM32, ESP32, or Arduino Nano 33 BLE.
