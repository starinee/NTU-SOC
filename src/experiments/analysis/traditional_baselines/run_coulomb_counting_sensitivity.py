import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# =========================
# 1. Paths
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[4]
MANIFEST_PATH = PROJECT_ROOT / "dataset/processed/panasonic_raw_csv/manifest.csv"

OUTPUT_DIR = PROJECT_ROOT / "dataset/processed/baseline_results_sensitivity"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 2. Reference setting
# =========================
# Reference SOC is fixed.
# Do NOT change this for sensitivity cases.
# Otherwise the error comparison will be meaningless.

REF_CAPACITY_AH = 2.9
REF_SOC0 = 100.0


# =========================
# 3. Sensitivity test cases
# =========================

TEST_CASES = [
    {
        "case_name": "ideal_capacity_2p9_soc0_100",
        "capacity_ah": 2.9,
        "soc0": 100.0,
    },
    {
        "case_name": "low_capacity_2p8_soc0_100",
        "capacity_ah": 2.8,
        "soc0": 100.0,
    },
    {
        "case_name": "high_capacity_3p0_soc0_100",
        "capacity_ah": 3.0,
        "soc0": 100.0,
    },
    {
        "case_name": "capacity_2p9_soc0_95",
        "capacity_ah": 2.9,
        "soc0": 95.0,
    },
    {
        "case_name": "capacity_2p9_soc0_90",
        "capacity_ah": 2.9,
        "soc0": 90.0,
    },
    {
        "case_name": "low_capacity_2p8_soc0_95",
        "capacity_ah": 2.8,
        "soc0": 95.0,
    },
    {
        "case_name": "high_capacity_3p0_soc0_95",
        "capacity_ah": 3.0,
        "soc0": 95.0,
    },
    {
        "case_name": "low_capacity_2p8_soc0_90",
        "capacity_ah": 2.8,
        "soc0": 90.0,
    },
    {
        "case_name": "high_capacity_3p0_soc0_90",
        "capacity_ah": 3.0,
        "soc0": 90.0,
    },
]


# =========================
# 4. Functions
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
    Coulomb Counting SOC estimation.

    Panasonic 18650PF raw data:
    current_A is mainly negative during discharge.
    Therefore, discharge_current_positive should be False.
    """

    df = df.copy().sort_values(time_col)

    time_s = df[time_col].to_numpy(dtype=float)
    current_a = df[current_col].to_numpy(dtype=float)

    # Convert current direction:
    # after this step, positive current means discharge.
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
    Ah-based reference SOC.

    This is fixed using REF_CAPACITY_AH and REF_SOC0.
    """

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
        "FINAL_ERROR_percent": error[-1],
    }


def make_safe_name(file_name):
    return Path(file_name).stem.replace(" ", "_").replace(".", "p")


def plot_sensitivity_curves(df, file_name, output_dir):
    """
    Plot reference SOC and several representative sensitivity cases.
    """

    safe_name = make_safe_name(file_name)

    # Plot SOC curves
    plt.figure(figsize=(10, 6))

    plt.plot(
        df["time_s"],
        df["soc_ref_ah"],
        label="Reference SOC",
        linewidth=2.5,
    )

    plot_cols = [
        "soc_cc__ideal_capacity_2p9_soc0_100",
        "soc_cc__low_capacity_2p8_soc0_100",
        "soc_cc__high_capacity_3p0_soc0_100",
        "soc_cc__capacity_2p9_soc0_95",
        "soc_cc__capacity_2p9_soc0_90",
    ]

    for col in plot_cols:
        if col in df.columns:
            plt.plot(df["time_s"], df[col], linestyle="--", label=col.replace("soc_cc__", ""))

    plt.xlabel("Time (s)")
    plt.ylabel("SOC (%)")
    plt.title(f"Coulomb Counting Sensitivity: {file_name}")
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / f"sensitivity_soc_curve_{safe_name}.png", dpi=200)
    plt.close()

    # Plot error curves
    plt.figure(figsize=(10, 6))

    for col in plot_cols:
        if col in df.columns:
            error = df[col] - df["soc_ref_ah"]
            plt.plot(df["time_s"], error, label=col.replace("soc_cc__", ""))

    plt.xlabel("Time (s)")
    plt.ylabel("SOC Error (%)")
    plt.title(f"Coulomb Counting Sensitivity Error: {file_name}")
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / f"sensitivity_error_curve_{safe_name}.png", dpi=200)
    plt.close()


def plot_rmse_bar(summary_df, output_dir):
    """
    Plot average RMSE for each sensitivity case.
    """

    avg_rmse = (
        summary_df
        .groupby("case_name")["RMSE_percent"]
        .mean()
        .sort_values()
    )

    plt.figure(figsize=(12, 6))
    avg_rmse.plot(kind="bar")
    plt.xlabel("Test Case")
    plt.ylabel("Average RMSE (%)")
    plt.title("Average RMSE of Coulomb Counting Sensitivity Cases")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(output_dir / "average_rmse_by_sensitivity_case.png", dpi=200)
    plt.close()


# =========================
# 5. Load manifest and select 25degC drive cycles
# =========================

manifest = pd.read_csv(MANIFEST_PATH)

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


# =========================
# 6. Run sensitivity test
# =========================

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

    # Fixed Ah-based reference SOC
    df["soc_ref_ah"] = soc_from_ah(
        df,
        ah_col="ah",
        capacity_ah=REF_CAPACITY_AH,
        soc0=REF_SOC0,
    )

    duration_s = df["time_s"].iloc[-1] - df["time_s"].iloc[0]
    final_ah = df["ah"].iloc[-1]
    final_soc_ref = df["soc_ref_ah"].iloc[-1]

    for case in TEST_CASES:
        case_name = case["case_name"]
        capacity_ah = case["capacity_ah"]
        soc0 = case["soc0"]

        soc_col = f"soc_cc__{case_name}"

        df[soc_col] = coulomb_counting(
            df,
            time_col="time_s",
            current_col="current_A",
            capacity_ah=capacity_ah,
            soc0=soc0,
            discharge_current_positive=False,
        )

        metrics = evaluate_soc(
            y_true=df["soc_ref_ah"].to_numpy(),
            y_pred=df[soc_col].to_numpy(),
        )

        final_soc_cc = df[soc_col].iloc[-1]

        record = {
            "file_name": file_name,
            "cycle_name": row.get("cycle_name", ""),
            "ambient_temp_C": row.get("ambient_temp_C", ""),
            "duration_s": duration_s,
            "final_ah": final_ah,
            "reference_capacity_ah": REF_CAPACITY_AH,
            "reference_soc0_percent": REF_SOC0,
            "case_name": case_name,
            "test_capacity_ah": capacity_ah,
            "test_soc0_percent": soc0,
            "final_soc_ref_percent": final_soc_ref,
            "final_soc_cc_percent": final_soc_cc,
            **metrics,
        }

        records.append(record)

        print(
            f"  {case_name}: "
            f"MAE={metrics['MAE_percent']:.4f}%, "
            f"RMSE={metrics['RMSE_percent']:.4f}%, "
            f"MAX={metrics['MAX_ERROR_percent']:.4f}%, "
            f"FINAL={metrics['FINAL_ERROR_percent']:.4f}%"
        )

    # Save per-cycle result CSV with all SOC columns
    safe_name = make_safe_name(file_name)
    result_csv_path = OUTPUT_DIR / f"cc_sensitivity_result_{safe_name}.csv"
    df.to_csv(result_csv_path, index=False)

    # Save plots
    plot_sensitivity_curves(df, file_name, OUTPUT_DIR)


# =========================
# 7. Save summary
# =========================

summary_df = pd.DataFrame(records)

summary_path = OUTPUT_DIR / "cc_sensitivity_25degC_drive_cycle_summary.csv"
summary_df.to_csv(summary_path, index=False)

plot_rmse_bar(summary_df, OUTPUT_DIR)

print("\nCoulomb Counting sensitivity test finished.")
print("Summary saved to:")
print(summary_path)

print("\nSummary preview:")
print(summary_df.head())

print("\nAverage metrics by case:")
avg_metrics = (
    summary_df
    .groupby("case_name")[["MAE_percent", "RMSE_percent", "MAX_ERROR_percent", "FINAL_ERROR_percent"]]
    .mean()
    .sort_values("RMSE_percent")
)

print(avg_metrics)

avg_metrics_path = OUTPUT_DIR / "cc_sensitivity_average_metrics_by_case.csv"
avg_metrics.to_csv(avg_metrics_path)

print("\nAverage metrics saved to:")
print(avg_metrics_path)
