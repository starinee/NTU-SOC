# Dataset Notes

## Source

The experiments use Panasonic NCR18650PF lithium-ion cell data with constant-temperature and drive-cycle profiles. The dataset is P. Kollmeyer's *Panasonic 18650PF Li-ion Battery Data and Example FNN and LSTM Neural Network SOC Estimator Training Script*, Mendeley Data, Version 1 (2018), https://doi.org/10.17632/wykht8y7tg.1. The local project expects both raw MAT files and processed CSV files.

## Expected Local Layout

```text
dataset/
  dataset_trad/
    Panasonic 18650PF Data/
      25degC/
      10degC/
      0degC/
      -10degC/
      -20degC/
  dataset_datadriven/
  processed/
    panasonic_raw_csv/
      manifest.csv
      *.csv
```

The raw-to-CSV conversion script is:

```bash
python src/data_processing/convert_panasonic_mat_to_csv.py
```

The full reproducible entry point is:

```bash
python src/experiments/run_reproducible_temperature_study.py
```

## Splits Used in This Project

For 25degC, 10degC, and 0degC experiments:

- Training profiles: `Cycle_1`, `Cycle_2`, `Cycle_3`, `Cycle_4`, `US06`
- Test profiles: `UDDS`, `LA92`, `NN`
- Temperature-rise profiles are excluded in the formal temperature-transfer experiments.

## Reference SOC

The reference SOC is built from Ah integration using a nominal capacity of 2.9 Ah. Because ideal Coulomb Counting and the reference SOC share nearly the same integration logic, ideal CC can show unrealistically low error. The project therefore includes sensitivity tests for initial SOC error, capacity error, current scale bias, current offset, current noise, and current drift.

## GitHub Data Policy

The original Panasonic `.mat` files are not redistributed here. Obtain them from the source above and comply with its license and terms of use. Generated high-volume CSV files are also excluded. For GitHub, keep:

- documentation files such as this `DATASET.md`;
- small summary tables, e.g. RMSE summaries and profile-wise metrics;
- selected figures used in the dissertation;
- scripts that reproduce the processed files from the local raw dataset.

Do not commit:

- `dataset/dataset_trad/`;
- `dataset/dataset_datadriven/`;
- `dataset/processed/panasonic_raw_csv/`;
- per-sample prediction dumps;
- model checkpoints.
