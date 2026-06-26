import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# =========================
# 1. Basic paths
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[4]
MANIFEST_PATH = PROJECT_ROOT / "dataset/processed/panasonic_raw_csv/manifest.csv"

OUTPUT_DIR = PROJECT_ROOT / "dataset/processed/baseline_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 2. Coulomb Counting functions
# =========================

def coulomb_counting(
    df,
    time_col="time_s",
    current_col="current_A",
    capacity_ah=2.9,
    soc0=100.0,
    discharge_current_positive=False,
):
    """
    Coulomb Counting SOC baseline.

    For Panasonic 18650PF raw data:
    current_A is usually negative during discharge.
    Therefore, discharge_current_positive should be False.
    """

    df = df.copy().sort_values(time_col)

    time_s = df[time_col].to_numpy(dtype=float)
    current_a = df[current_col].to_numpy(dtype=float)

    # Convert current direction.
    # After this step, positive current means discharge.
    if not discharge_current_positive:
        current_a = -current_a

    dt = np.diff(time_s, prepend=time_s[0])
    dt[0] = 0.0

    capacity_coulomb = capacity_ah * 3600.0

    delta_soc_percent = current_a * dt / capacity_coulomb * 100.0

    soc = soc0 - np.cumsum(delta_soc_percent)
    soc = np.clip(soc, 0.0, 100.0)

    return soc


def soc_from_ah(
    df,
    ah_col="ah",
    capacity_ah=2.9,
    soc0=100.0,
):
    """
    Build an Ah-based reference SOC from the dataset Ah column.

    In this dataset, Ah usually decreases during discharge.
    Example:
        Ah = 0       -> SOC = 100%
        Ah = -2.9    -> SOC = 0%
    """

    ah = df[ah_col].to_numpy(dtype=float)

    soc_ref = soc0 + ah / capacity_ah * 100.0
    soc_ref = np.clip(soc_ref, 0.0, 100.0)

    return soc_ref


def evaluate_soc(y_true, y_pred):
    error = y_pred - y_true

    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))
    max_error = np.max(np.abs(error))

    return {
        "MAE_percent": mae,
        "RMSE_percent": rmse,
        "MAX_ERROR_percent": max_error,
    }


# =========================
# 3. Select one suitable drive cycle file
# =========================

manifest = pd.read_csv(MANIFEST_PATH)

print("Manifest loaded.")
print("Manifest shape:", manifest.shape)
print("\nManifest columns:")
print(manifest.columns.tolist())

print("\nUnique ambient_temp_C:")
print(manifest["ambient_temp_C"].unique())

print("\nUnique test_type:")
print(manifest["test_type"].unique())


# First priority:
# choose 25degC drive-cycle-like files by filename.
drive_candidates = manifest[
    manifest["file_name"].str.contains("25degC", case=False, na=False)
    &
    manifest["file_name"].str.contains(
        "UDDS|LA92|NN|HWFET|US06",
        case=False,
        na=False,
        regex=True,
    )
].copy()

print("\n25degC drive cycle candidates:")
if len(drive_candidates) > 0:
    print(drive_candidates[["ambient_temp_C", "test_group", "test_type", "file_name", "output_csv"]])
else:
    print("No 25degC drive cycle candidate found by filename.")


# Second priority:
# use manifest metadata if filename method fails.
if len(drive_candidates) == 0:
    drive_candidates = manifest[
        (
            manifest["ambient_temp_C"].astype(str).str.contains("25", na=False)
        )
        &
        (
            (manifest["test_type"] == "drive_cycle")
            |
            (manifest["test_group"].astype(str).str.contains("Drive", case=False, na=False))
        )
    ].copy()

    print("\nFallback 1: 25degC drive candidates by metadata:")
    print(drive_candidates[["ambient_temp_C", "test_group", "test_type", "file_name", "output_csv"]])


# Third priority:
# use any drive cycle file.
if len(drive_candidates) == 0:
    drive_candidates = manifest[
        (manifest["test_type"] == "drive_cycle")
        |
        (manifest["test_group"].astype(str).str.contains("Drive", case=False, na=False))
        |
        (manifest["file_name"].str.contains("UDDS|LA92|NN|HWFET|US06", case=False, na=False, regex=True))
    ].copy()

    print("\nFallback 2: all drive cycle candidates:")
    print(drive_candidates[["ambient_temp_C", "test_group", "test_type", "file_name", "output_csv"]])


if len(drive_candidates) == 0:
    raise ValueError("No drive cycle file found. Please check manifest.csv.")


# Prefer UDDS first if available, otherwise use the first candidate.
udds_candidates = drive_candidates[
    drive_candidates["file_name"].str.contains("UDDS", case=False, na=False)
]

if len(udds_candidates) > 0:
    selected_row = udds_candidates.iloc[0]
else:
    selected_row = drive_candidates.iloc[0]


file_path = selected_row["output_csv"]
file_name = selected_row["file_name"]

print("\nSelected file:")
print(file_name)
print(file_path)


# =========================
# 4. Load selected CSV
# =========================

df = pd.read_csv(file_path)

print("\nSelected CSV shape:", df.shape)
print("\nSelected CSV columns:")
print(df.columns.tolist())

required_cols = ["time_s", "current_A", "voltage_V", "ah"]

for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Required column missing: {col}")

print("\nBasic data description:")
print(df[["time_s", "voltage_V", "current_A", "ah"]].describe())


# =========================
# 5. Run Coulomb Counting
# =========================

capacity_ah = 2.9
soc0 = 100.0

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

metrics = evaluate_soc(
    y_true=df["soc_ref_ah"],
    y_pred=df["soc_cc"],
)

print("\nCoulomb Counting baseline result:")
for k, v in metrics.items():
    print(f"{k}: {v:.6f}")


# =========================
# 6. Plot results
# =========================

plt.figure()
plt.plot(df["time_s"], df["voltage_V"])
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.title(f"Voltage vs Time: {file_name}")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(df["time_s"], df["current_A"])
plt.xlabel("Time (s)")
plt.ylabel("Current (A)")
plt.title(f"Current vs Time: {file_name}")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(df["time_s"], df["ah"])
plt.xlabel("Time (s)")
plt.ylabel("Ah")
plt.title(f"Ah vs Time: {file_name}")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(df["time_s"], df["soc_ref_ah"], label="Ah-based Reference SOC")
plt.plot(df["time_s"], df["soc_cc"], label="Coulomb Counting SOC", linestyle="--")
plt.xlabel("Time (s)")
plt.ylabel("SOC (%)")
plt.title(f"Coulomb Counting Baseline: {file_name}")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(df["time_s"], df["soc_cc"] - df["soc_ref_ah"])
plt.xlabel("Time (s)")
plt.ylabel("SOC Error (%)")
plt.title(f"Coulomb Counting Error: {file_name}")
plt.grid(True)
plt.show()


# =========================
# 7. Save output
# =========================

safe_name = Path(file_name).stem.replace(" ", "_")
result_csv_path = f"{OUTPUT_DIR}/coulomb_counting_{safe_name}.csv"
metrics_csv_path = f"{OUTPUT_DIR}/coulomb_counting_metrics_{safe_name}.csv"

df.to_csv(result_csv_path, index=False)

metrics_df = pd.DataFrame([{
    "file_name": file_name,
    "capacity_ah": capacity_ah,
    "soc0": soc0,
    **metrics,
}])

metrics_df.to_csv(metrics_csv_path, index=False)

print("\nSaved result CSV to:")
print(result_csv_path)

print("\nSaved metrics CSV to:")
print(metrics_csv_path)
