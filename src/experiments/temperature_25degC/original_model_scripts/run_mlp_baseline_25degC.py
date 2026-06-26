from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =========================
# Paths
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[4]
MANIFEST_PATH = PROJECT_ROOT / "dataset/processed/panasonic_raw_csv/manifest.csv"
OUTPUT_DIR = PROJECT_ROOT / "dataset/processed/mlp_baseline_25degC"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Experiment config
# =========================
CAPACITY_AH = 2.9
SOC0 = 100.0

# Use 1 Hz data for the first CPU-friendly baseline.
# The raw drive-cycle files are usually logged at 0.1 s.
DOWNSAMPLE_STEP = 10

FEATURE_COLUMNS = [
    "voltage_V",
    "current_A",
    "battery_temp_C",
]

# Cycle-level split avoids random row leakage.
TRAIN_CYCLE_KEYWORDS = [
    "Cycle_1",
    "Cycle_2",
    "Cycle_3",
    "Cycle_4",
    "US06",
]

TEST_CYCLE_KEYWORDS = [
    "UDDS",
    "LA92",
    "NN",
]


def make_safe_name(file_name: str) -> str:
    return Path(file_name).stem.replace(" ", "_").replace(".", "p")


def soc_from_ah(
    df: pd.DataFrame,
    ah_col: str = "ah",
    capacity_ah: float = CAPACITY_AH,
    soc0: float = SOC0,
) -> np.ndarray:
    """
    Build Ah-based reference SOC.

    Panasonic drive-cycle Ah is usually negative during discharge, so:
        SOC = SOC0 + Ah / capacity * 100
    """
    ah = df[ah_col].to_numpy(dtype=float)
    soc_ref = soc0 + ah / capacity_ah * 100.0
    return np.clip(soc_ref, 0.0, 100.0)


def evaluate_soc(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_pred - y_true

    return {
        "MAE_percent": float(mean_absolute_error(y_true, y_pred)),
        "RMSE_percent": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAX_ERROR_percent": float(np.max(np.abs(error))),
        "FINAL_ERROR_percent": float(error[-1]),
    }


def contains_any(text: str, keywords: list[str]) -> bool:
    text_lower = str(text).lower()
    return any(k.lower() in text_lower for k in keywords)


def select_25degC_drive_files(manifest: pd.DataFrame) -> pd.DataFrame:
    """
    Select split 25degC drive-cycle files.

    The current manifest has blank ambient_temp_C for some 25degC files because
    the original raw folder has an extra parent directory. Filename matching is
    used here to stay consistent with the previous baseline scripts.
    """
    drive_25 = manifest[
        manifest["file_name"].astype(str).str.contains("25degC", case=False, na=False)
        & manifest["file_name"].astype(str).str.contains(
            "Cycle_1|Cycle_2|Cycle_3|Cycle_4|US06|UDDS|LA92|NN",
            case=False,
            na=False,
            regex=True,
        )
    ].copy()

    # Exclude combined HWFET / mixed long files for this first clean baseline.
    drive_25 = drive_25[
        ~drive_25["file_name"].astype(str).str.contains("HWFT|HWFET", case=False, na=False)
    ].copy()

    if len(drive_25) == 0:
        raise ValueError("No 25degC drive-cycle files found.")

    return drive_25.sort_values("file_name").reset_index(drop=True)


def assign_split(row: pd.Series) -> str:
    file_name = row["file_name"]

    if contains_any(file_name, TRAIN_CYCLE_KEYWORDS):
        return "train"
    if contains_any(file_name, TEST_CYCLE_KEYWORDS):
        return "test"
    return "unused"


def load_one_cycle(row: pd.Series, split: str) -> pd.DataFrame:
    file_path = row["output_csv"]
    file_name = row["file_name"]
    cycle_name = row.get("cycle_name", "")

    df = pd.read_csv(file_path)
    df = df.sort_values("time_s").reset_index(drop=True)

    required_cols = ["time_s", "voltage_V", "current_A", "ah", *FEATURE_COLUMNS]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{file_name}: missing columns {missing}")

    df = df.dropna(subset=required_cols).reset_index(drop=True)

    if DOWNSAMPLE_STEP > 1:
        df = df.iloc[::DOWNSAMPLE_STEP].reset_index(drop=True)

    df["soc_ref_ah"] = soc_from_ah(df)
    df["split"] = split
    df["cycle_name"] = cycle_name
    df["file_name"] = file_name

    return df


def plot_prediction(df: pd.DataFrame, metrics: dict, output_dir: Path):
    file_name = df["file_name"].iloc[0]
    safe_name = make_safe_name(file_name)

    plt.figure(figsize=(10, 6))
    plt.plot(df["time_s"], df["soc_ref_ah"], label="Ah-based Reference SOC", linewidth=2)
    plt.plot(df["time_s"], df["soc_mlp"], label="MLP Predicted SOC", linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel("SOC (%)")
    plt.title(f"MLP SOC Estimation: {file_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / f"mlp_soc_prediction_{safe_name}.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(df["time_s"], df["soc_mlp"] - df["soc_ref_ah"])
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xlabel("Time (s)")
    plt.ylabel("SOC Error (%)")
    plt.title(
        f"MLP Error: {file_name} "
        f"(RMSE={metrics['RMSE_percent']:.3f}%, MAE={metrics['MAE_percent']:.3f}%)"
    )
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / f"mlp_soc_error_{safe_name}.png", dpi=200)
    plt.close()


def main():
    print("Loading manifest:")
    print(MANIFEST_PATH)
    manifest = pd.read_csv(MANIFEST_PATH)

    drive_25 = select_25degC_drive_files(manifest)
    drive_25["split"] = drive_25.apply(assign_split, axis=1)

    print("\nSelected 25degC drive-cycle files:")
    print(drive_25[["file_name", "cycle_name", "split", "output_csv"]].to_string(index=False))

    train_rows = drive_25[drive_25["split"] == "train"]
    test_rows = drive_25[drive_25["split"] == "test"]

    if len(train_rows) == 0:
        raise ValueError("No training files selected.")
    if len(test_rows) == 0:
        raise ValueError("No test files selected.")

    print("\nLoading training data...")
    train_frames = [load_one_cycle(row, "train") for _, row in train_rows.iterrows()]
    train_df = pd.concat(train_frames, ignore_index=True)

    print("Loading test data...")
    test_frames = [load_one_cycle(row, "test") for _, row in test_rows.iterrows()]
    test_df = pd.concat(test_frames, ignore_index=True)

    print("\nTraining samples:", len(train_df))
    print("Test samples:", len(test_df))
    print("Features:", FEATURE_COLUMNS)
    print("Downsample step:", DOWNSAMPLE_STEP)

    x_train = train_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_train = train_df["soc_ref_ah"].to_numpy(dtype=float)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    batch_size=512,
                    learning_rate_init=1e-3,
                    max_iter=300,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=20,
                    random_state=42,
                    verbose=True,
                ),
            ),
        ]
    )

    print("\nTraining MLP baseline...")
    model.fit(x_train, y_train)

    records = []

    # Save train metrics too, but use test metrics for thesis comparison.
    train_pred = model.predict(x_train)
    train_pred = np.clip(train_pred, 0.0, 100.0)
    train_metrics = evaluate_soc(y_train, train_pred)
    records.append(
        {
            "split": "train",
            "file_name": "ALL_TRAIN_FILES",
            "cycle_name": "train_combined",
            "sample_count": len(train_df),
            **train_metrics,
        }
    )

    print("\nTraining-set metrics:")
    for k, v in train_metrics.items():
        print(f"{k}: {v:.6f}")

    print("\nEvaluating test cycles...")
    prediction_frames = []

    for file_name, cycle_df in test_df.groupby("file_name", sort=False):
        cycle_df = cycle_df.copy().reset_index(drop=True)

        x_test = cycle_df[FEATURE_COLUMNS].to_numpy(dtype=float)
        y_test = cycle_df["soc_ref_ah"].to_numpy(dtype=float)

        y_pred = model.predict(x_test)
        y_pred = np.clip(y_pred, 0.0, 100.0)

        cycle_df["soc_mlp"] = y_pred
        cycle_df["soc_error_percent"] = cycle_df["soc_mlp"] - cycle_df["soc_ref_ah"]

        metrics = evaluate_soc(y_test, y_pred)

        record = {
            "split": "test",
            "file_name": file_name,
            "cycle_name": cycle_df["cycle_name"].iloc[0],
            "sample_count": len(cycle_df),
            **metrics,
        }
        records.append(record)

        safe_name = make_safe_name(file_name)
        result_path = OUTPUT_DIR / f"mlp_prediction_{safe_name}.csv"
        cycle_df.to_csv(result_path, index=False)
        plot_prediction(cycle_df, metrics, OUTPUT_DIR)
        prediction_frames.append(cycle_df)

        print(
            f"{file_name}: "
            f"MAE={metrics['MAE_percent']:.4f}%, "
            f"RMSE={metrics['RMSE_percent']:.4f}%, "
            f"MAX={metrics['MAX_ERROR_percent']:.4f}%"
        )

    results_df = pd.DataFrame(records)

    test_only = results_df[results_df["split"] == "test"]
    average_record = {
        "split": "test_average",
        "file_name": "AVERAGE_TEST_CYCLES",
        "cycle_name": "test_average",
        "sample_count": int(test_only["sample_count"].sum()),
        "MAE_percent": float(test_only["MAE_percent"].mean()),
        "RMSE_percent": float(test_only["RMSE_percent"].mean()),
        "MAX_ERROR_percent": float(test_only["MAX_ERROR_percent"].mean()),
        "FINAL_ERROR_percent": float(test_only["FINAL_ERROR_percent"].mean()),
    }

    results_df = pd.concat([results_df, pd.DataFrame([average_record])], ignore_index=True)

    summary_path = OUTPUT_DIR / "mlp_baseline_25degC_summary.csv"
    results_df.to_csv(summary_path, index=False)

    all_predictions_path = OUTPUT_DIR / "mlp_baseline_25degC_test_predictions.csv"
    pd.concat(prediction_frames, ignore_index=True).to_csv(all_predictions_path, index=False)

    print("\nMLP baseline finished.")
    print("Summary saved to:")
    print(summary_path)
    print("\nCombined test predictions saved to:")
    print(all_predictions_path)
    print("\nOutput plots saved to:")
    print(OUTPUT_DIR)
    print("\nSummary:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
