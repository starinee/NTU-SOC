# Reproducible experiment entry points

Run the whole temperature study from original Panasonic `.mat` files:

```bash
python src/experiments/run_reproducible_temperature_study.py
```

The master script has two stages:

1. `src/data_processing/convert_panasonic_mat_to_csv.py`
   converts the original raw dataset into `dataset/processed/panasonic_raw_csv`.
2. `src/data/temperature_experiments/run_strict_matched_temperature_pipeline_25C_10C_0C.py`
   runs the strict matched SOC experiments for 25degC, 10degC, 0degC, and
   direct transfer from 25degC to 10degC/0degC.

Folder roles:

- `shared/`: controlled constants and the experimental contract.
- `temperature_25degC/`: 25degC within-temperature line.
- `temperature_10degC/`: 10degC within-temperature line.
- `temperature_0degC/`: 0degC within-temperature line.
- `temperature_transfer/`: 25degC-trained direct transfer to 10degC and 0degC.
- `analysis/`: paper figures and error-analysis scripts.
