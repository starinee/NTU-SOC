# Dataset Notes

## Source

The experiments use Panasonic NCR18650PF lithium-ion cell data with constant-temperature and drive-cycle profiles. The local project expects both raw MAT files and processed CSV files.

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

The clean release entry point is:

```bash
python scripts/run_reproducible_pipeline.py --temperature all
```

`--temperature all` refreshes the final paper tables and deployment-oriented
proxy outputs from the committed strict results. If
`dataset/processed/panasonic_raw_csv/manifest.csv` is available, it also reruns
the strict matched training pipeline. To force a full strict retraining run,
prepare the Panasonic dataset first and run:

```bash
python scripts/run_reproducible_pipeline.py --temperature strict
```

## Splits Used in This Project

For 25degC, 10degC, and 0degC experiments:

- Training profiles: `Cycle_1`, `Cycle_2`, `Cycle_3`, `Cycle_4`, `US06`
- Test profiles: `UDDS`, `LA92`, `NN`
- Temperature-rise profiles are excluded in the formal temperature-transfer experiments.

## Reference SOC

The reference SOC is built from Ah integration using a nominal capacity of 2.9 Ah. Because ideal Coulomb Counting and the reference SOC share nearly the same integration logic, ideal CC can show unrealistically low error. The project therefore includes sensitivity tests for initial SOC error, capacity error, current scale bias, current offset, current noise, and current drift.

## GitHub Data Policy

The original Panasonic `.mat` files and regenerated high-volume CSV files should
not be committed to a public repository unless redistribution is explicitly
permitted. For GitHub, keep:

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
