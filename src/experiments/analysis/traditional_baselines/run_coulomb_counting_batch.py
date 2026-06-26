import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
MANIFEST_PATH = PROJECT_ROOT / "dataset/processed/panasonic_raw_csv/manifest.csv"
OUTPUT_DIR = PROJECT_ROOT / "dataset/processed/baseline_results_batch"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def coulomb_counting(
    df,
    time_col="time_s",
    current_col="current_A",
    capacity_ah=2.9,
    soc0=100.0,
    discharge_current_positive=False,
):
    df = df.copy().sort_values(time_col)

    time_s = df[time_col].to_numpy(dtype=float)
    current_a = df[current_col].to_numpy(dtype=float)

    if not discharge_current_positive:
        current_a = -current_a

    dt = np.diff(time_s, prepend=time_s[0])
    dt[0] = 0.0

    capacity_coulomb = capacity_ah * 3600.0
    delta_soc_percent = current_a * dt / capacity_coulomb * 100.0

    soc = soc0 - np.cumsum(delta_soc_percent)
    soc = np.clip(soc, 0.0, 100.0)

    return soc


def soc_from_ah(df, ah_col="ah", capacity_ah=2.9, soc0=100.0):
    ah = df[ah_col].to_numpy(dtype=float)
    soc_ref = soc0 + ah / capacity_ah * 100.0
    soc_ref = np.clip(soc_ref, 0.0, 100.0)
    return soc_ref


def evaluate_soc(y_true, y_pred):
    error = y_pred - y_true

    return {
        "MAE_percent": np.mean(np.abs(error)),
        "RMSE_percent": np.sqrt(np.mean(error ** 2)),
        "MAX_ERROR_percent": np.max(np.abs(error)),
    }


def plot_and_save(df, file_name, output_dir):
    safe_name = Path(file_name).stem.replace(" ", "_")

    plt.figure()
    plt.plot(df["time_s"], df["soc_ref_ah"], label="Ah-based Reference SOC")
    plt.plot(df["time_s"], df["soc_cc"], label="Coulomb Counting SOC", linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel("SOC (%)")
    plt.title(f"Coulomb Counting Baseline: {file_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / f"soc_curve_{safe_name}.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(df["time_s"], df["soc_cc"] - df["soc_ref_ah"])
    plt.xlabel("Time (s)")
    plt.ylabel("SOC Error (%)")
    plt.title(f"Coulomb Counting Error: {file_name}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / f"soc_error_{safe_name}.png", dpi=200)
    plt.close()


manifest = pd.read_csv(MANIFEST_PATH)

# 先选 25degC 的 drive-cycle-like 文件
drive_25 = manifest[
    manifest["file_name"].str.contains("25degC", case=False, na=False)
    &
    manifest["file_name"].str.contains(
        "UDDS|LA92|NN|HWFET|US06",
        case=False,
        na=False,
        regex=True,
    )
].copy()

print("25degC drive cycle files selected:")
print(drive_25[["file_name", "cycle_name", "output_csv"]])

if len(drive_25) == 0:
    raise ValueError("No 25degC drive cycle files found.")

capacity_ah = 2.9
soc0 = 100.0

records = []

for idx, row in drive_25.iterrows():
    file_name = row["file_name"]
    file_path = row["output_csv"]

    print("\nProcessing:", file_name)

    df = pd.read_csv(file_path)

    required_cols = ["time_s", "current_A", "voltage_V", "ah"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"{file_name}: missing column {col}")

    df["soc_cc"] = coulomb_counting(
        df,
        time_col="time_s",
        current_col="current_A",
        capacity_ah=capacity_ah,
        soc0=soc0,
        discharge_current_positive=False,
    )

    df["soc_ref_ah"] = soc_from_ah(
        df,
        ah_col="ah",
        capacity_ah=capacity_ah,
        soc0=soc0,
    )

    metrics = evaluate_soc(df["soc_ref_ah"], df["soc_cc"])

    duration_s = df["time_s"].iloc[-1] - df["time_s"].iloc[0]
    final_soc_ref = df["soc_ref_ah"].iloc[-1]
    final_soc_cc = df["soc_cc"].iloc[-1]
    final_ah = df["ah"].iloc[-1]

    record = {
        "file_name": file_name,
        "cycle_name": row.get("cycle_name", ""),
        "ambient_temp_C": row.get("ambient_temp_C", ""),
        "duration_s": duration_s,
        "capacity_ah": capacity_ah,
        "initial_soc_percent": soc0,
        "final_ah": final_ah,
        "final_soc_ref_percent": final_soc_ref,
        "final_soc_cc_percent": final_soc_cc,
        **metrics,
    }

    records.append(record)

    safe_name = Path(file_name).stem.replace(" ", "_")
    result_csv_path = OUTPUT_DIR / f"cc_result_{safe_name}.csv"
    df.to_csv(result_csv_path, index=False)

    plot_and_save(df, file_name, OUTPUT_DIR)

    print(
        f"MAE={metrics['MAE_percent']:.4f}%, "
        f"RMSE={metrics['RMSE_percent']:.4f}%, "
        f"MAX={metrics['MAX_ERROR_percent']:.4f}%"
    )


results_df = pd.DataFrame(records)

summary_path = OUTPUT_DIR / "coulomb_counting_25degC_drive_cycle_summary.csv"
results_df.to_csv(summary_path, index=False)

print("\nBatch Coulomb Counting finished.")
print("Summary saved to:")
print(summary_path)

print("\nSummary:")
print(results_df)
