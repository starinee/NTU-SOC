from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = PROJECT_ROOT / "dataset/processed/panasonic_raw_csv/manifest.csv"
OUTPUT_DIR = PROJECT_ROOT / "dataset/processed/traditional_baselines/cc_current_noise_sensitivity_25C"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CAPACITY_AH = 2.9
SOC0 = 100.0
RANDOM_SEED = 42
TEST_CYCLES = ["UDDS", "LA92", "NN", "US06"]

SENSITIVITY_CASES = [
    {
        "case_name": "ideal_capacity_2p9_soc0_100",
        "capacity_ah": 2.9,
        "soc0": 100.0,
        "current_scale": 1.0,
        "current_offset_A": 0.0,
        "current_noise_std_A": 0.0,
        "current_drift_final_A": 0.0,
    },
    {
        "case_name": "capacity_2p8_soc0_100",
        "capacity_ah": 2.8,
        "soc0": 100.0,
        "current_scale": 1.0,
        "current_offset_A": 0.0,
        "current_noise_std_A": 0.0,
        "current_drift_final_A": 0.0,
    },
    {
        "case_name": "capacity_3p0_soc0_100",
        "capacity_ah": 3.0,
        "soc0": 100.0,
        "current_scale": 1.0,
        "current_offset_A": 0.0,
        "current_noise_std_A": 0.0,
        "current_drift_final_A": 0.0,
    },
    {
        "case_name": "capacity_2p9_soc0_95",
        "capacity_ah": 2.9,
        "soc0": 95.0,
        "current_scale": 1.0,
        "current_offset_A": 0.0,
        "current_noise_std_A": 0.0,
        "current_drift_final_A": 0.0,
    },
    {
        "case_name": "current_scale_plus_1pct",
        "capacity_ah": 2.9,
        "soc0": 100.0,
        "current_scale": 1.01,
        "current_offset_A": 0.0,
        "current_noise_std_A": 0.0,
        "current_drift_final_A": 0.0,
    },
    {
        "case_name": "current_scale_minus_1pct",
        "capacity_ah": 2.9,
        "soc0": 100.0,
        "current_scale": 0.99,
        "current_offset_A": 0.0,
        "current_noise_std_A": 0.0,
        "current_drift_final_A": 0.0,
    },
    {
        "case_name": "current_offset_plus_50mA",
        "capacity_ah": 2.9,
        "soc0": 100.0,
        "current_scale": 1.0,
        "current_offset_A": 0.05,
        "current_noise_std_A": 0.0,
        "current_drift_final_A": 0.0,
    },
    {
        "case_name": "current_offset_minus_50mA",
        "capacity_ah": 2.9,
        "soc0": 100.0,
        "current_scale": 1.0,
        "current_offset_A": -0.05,
        "current_noise_std_A": 0.0,
        "current_drift_final_A": 0.0,
    },
    {
        "case_name": "current_noise_50mA_rms",
        "capacity_ah": 2.9,
        "soc0": 100.0,
        "current_scale": 1.0,
        "current_offset_A": 0.0,
        "current_noise_std_A": 0.05,
        "current_drift_final_A": 0.0,
    },
    {
        "case_name": "current_noise_100mA_rms",
        "capacity_ah": 2.9,
        "soc0": 100.0,
        "current_scale": 1.0,
        "current_offset_A": 0.0,
        "current_noise_std_A": 0.10,
        "current_drift_final_A": 0.0,
    },
    {
        "case_name": "current_drift_to_plus_100mA",
        "capacity_ah": 2.9,
        "soc0": 100.0,
        "current_scale": 1.0,
        "current_offset_A": 0.0,
        "current_noise_std_A": 0.0,
        "current_drift_final_A": 0.10,
    },
    {
        "case_name": "combined_soc0_95_scale_plus_1pct_noise_50mA",
        "capacity_ah": 2.9,
        "soc0": 95.0,
        "current_scale": 1.01,
        "current_offset_A": 0.0,
        "current_noise_std_A": 0.05,
        "current_drift_final_A": 0.0,
    },
]


def safe_name(text):
    return Path(str(text)).stem.replace(" ", "_").replace(".", "p")


def soc_from_ah(df):
    return np.clip(SOC0 + df["ah"].to_numpy(dtype=float) / CAPACITY_AH * 100.0, 0.0, 100.0)


def evaluate(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_pred - y_true
    abs_error = np.abs(error)
    return {
        "MAE_percent": float(np.mean(abs_error)),
        "RMSE_percent": float(np.sqrt(np.mean(error**2))),
        "MAX_ERROR_percent": float(np.max(abs_error)),
        "P95_ABS_ERROR_percent": float(np.quantile(abs_error, 0.95)),
        "P99_ABS_ERROR_percent": float(np.quantile(abs_error, 0.99)),
        "FINAL_ERROR_percent": float(error[-1]),
    }


def select_25c_test_rows(manifest):
    rows = manifest[
        manifest["file_name"].astype(str).str.contains("25degC", case=False, na=False)
        & manifest["cycle_name"].astype(str).isin(TEST_CYCLES)
        & manifest["test_type"].eq("drive_cycle")
    ].copy()
    rows = rows[~rows["file_name"].astype(str).str.contains("HWFET|HWFT", case=False, regex=True, na=False)]
    return rows.sort_values("file_name").reset_index(drop=True)


def perturb_discharge_current(current_for_cc, case, rng):
    current = np.asarray(current_for_cc, dtype=float).copy()
    current = current * case["current_scale"]
    current = current + case["current_offset_A"]
    if case["current_noise_std_A"] > 0:
        current = current + rng.normal(0.0, case["current_noise_std_A"], size=len(current))
    if case["current_drift_final_A"] != 0:
        current = current + np.linspace(0.0, case["current_drift_final_A"], len(current))
    return current


def coulomb_counting_with_current_nonidealities(df, case, seed):
    raw_current = -df["current_A"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    current = perturb_discharge_current(raw_current, case, rng)
    time_s = df["time_s"].to_numpy(dtype=float)
    dt = np.diff(time_s, prepend=time_s[0])
    dt[0] = 0.0
    delta_soc = current * dt / (case["capacity_ah"] * 3600.0) * 100.0
    return np.clip(case["soc0"] - np.cumsum(delta_soc), 0.0, 100.0), current


def plot_case_error_curves(result_frames, file_name):
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for case_name, df in result_frames.items():
        if case_name in [
            "ideal_capacity_2p9_soc0_100",
            "capacity_2p9_soc0_95",
            "current_scale_plus_1pct",
            "current_noise_50mA_rms",
            "current_drift_to_plus_100mA",
            "combined_soc0_95_scale_plus_1pct_noise_50mA",
        ]:
            elapsed_min = (df["time_s"] - df["time_s"].iloc[0]) / 60.0
            ax.plot(elapsed_min, df["soc_error_percent"], linewidth=1.2, label=case_name)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title(f"CC sensitivity with current nonidealities: {file_name}")
    ax.set_xlabel("Elapsed time (min)")
    ax.set_ylabel("SOC error (%SOC)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"cc_current_noise_error_{safe_name(file_name)}.png", dpi=220)
    plt.close(fig)


def markdown_table(df, path):
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda v: "" if pd.isna(v) else f"{v:.4f}".rstrip("0").rstrip("."))
        else:
            display[col] = display[col].map(lambda v: "" if pd.isna(v) else str(v))
    lines = [
        "| " + " | ".join(display.columns) + " |",
        "| " + " | ".join(["---"] * len(display.columns)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(row.astype(str).tolist()) + " |")
    Path(path).write_text("\n".join(lines) + "\n")


def plot_average_rmse(avg):
    ordered = avg.sort_values("RMSE_percent", ascending=True)
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    ax.barh(ordered["case_name"], ordered["RMSE_percent"], color="#2f6f73")
    ax.set_xlabel("Average RMSE (%SOC)")
    ax.set_title("CC sensitivity: capacity, initial SOC, and current nonidealities")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cc_current_noise_average_rmse_by_case.png", dpi=220)
    plt.close(fig)


def main():
    manifest = pd.read_csv(MANIFEST_PATH)
    test_rows = select_25c_test_rows(manifest)
    test_rows.to_csv(OUTPUT_DIR / "selected_25C_test_rows.csv", index=False)
    print("Selected files:")
    print(test_rows[["file_name", "cycle_name", "output_csv"]].to_string(index=False))

    records = []
    for _, row in test_rows.iterrows():
        source = pd.read_csv(row["output_csv"]).sort_values("time_s").reset_index(drop=True)
        source = source.dropna(subset=["time_s", "current_A", "voltage_V", "ah"]).reset_index(drop=True)
        y = soc_from_ah(source)
        result_frames = {}
        for case_index, case in enumerate(SENSITIVITY_CASES):
            pred, perturbed_current = coulomb_counting_with_current_nonidealities(
                source,
                case,
                seed=RANDOM_SEED + case_index,
            )
            df = source.copy()
            df["soc_ref_ah"] = y
            df["cc_case_name"] = case["case_name"]
            df["current_for_cc_A"] = perturbed_current
            df["soc_cc_nonideal"] = pred
            df["soc_error_percent"] = pred - y
            result_frames[case["case_name"]] = df
            df.to_csv(OUTPUT_DIR / f"cc_current_noise_{case['case_name']}_{safe_name(row['file_name'])}.csv", index=False)
            records.append(
                {
                    "case_name": case["case_name"],
                    "file_name": row["file_name"],
                    "cycle_name": row["cycle_name"],
                    "sample_count": len(df),
                    "capacity_ah": case["capacity_ah"],
                    "soc0": case["soc0"],
                    "current_scale": case["current_scale"],
                    "current_offset_A": case["current_offset_A"],
                    "current_noise_std_A": case["current_noise_std_A"],
                    "current_drift_final_A": case["current_drift_final_A"],
                    **evaluate(y, pred),
                }
            )
        plot_case_error_curves(result_frames, row["file_name"])

    summary = pd.DataFrame(records)
    avg = (
        summary.groupby(
            [
                "case_name",
                "capacity_ah",
                "soc0",
                "current_scale",
                "current_offset_A",
                "current_noise_std_A",
                "current_drift_final_A",
            ],
            as_index=False,
        )[["sample_count", "MAE_percent", "RMSE_percent", "MAX_ERROR_percent", "P95_ABS_ERROR_percent", "P99_ABS_ERROR_percent", "FINAL_ERROR_percent"]]
        .mean()
    )
    summary.to_csv(OUTPUT_DIR / "cc_current_noise_sensitivity_25C_profilewise.csv", index=False)
    avg.to_csv(OUTPUT_DIR / "cc_current_noise_sensitivity_25C_average_by_case.csv", index=False)
    markdown_table(avg, OUTPUT_DIR / "cc_current_noise_sensitivity_25C_average_by_case.md")
    plot_average_rmse(avg)
    print(avg.sort_values("RMSE_percent").to_string(index=False))
    print(f"\nSaved current-noise CC sensitivity outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
