import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# =========================
# Paths
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[4]
MANIFEST_PATH = PROJECT_ROOT / "dataset/processed/panasonic_raw_csv/manifest.csv"
OCV_LOOKUP_PATH = PROJECT_ROOT / "dataset/processed/ocv_lookup/ocv_soc_lookup_25degC.csv"

OUTPUT_DIR = PROJECT_ROOT / "dataset/processed/baseline_results_ocv_corrected"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Config
# =========================
CAPACITY_AH = 2.9
SOC0 = 100.0

# OCV 修正参数
# 只有电流接近 0，并且持续一段时间，才认为 terminal voltage 可以近似 OCV
REST_CURRENT_THRESHOLD_A = 0.05
MIN_REST_DURATION_S = 30.0
MIN_CORRECTION_INTERVAL_S = 300.0

# 修正不要太激进，否则动态工况下容易被极化电压带偏
CORRECTION_GAIN = 0.20
MAX_CORRECTION_STEP_PERCENT = 1.0


# =========================
# Basic functions
# =========================
def evaluate_soc(y_true, y_pred):
    error = np.asarray(y_pred) - np.asarray(y_true)

    return {
        "MAE_percent": float(np.mean(np.abs(error))),
        "RMSE_percent": float(np.sqrt(np.mean(error ** 2))),
        "MAX_ERROR_percent": float(np.max(np.abs(error))),
        "FINAL_ERROR_percent": float(error[-1]),
    }


def soc_from_ah(df, ah_col="ah", capacity_ah=2.9, soc0=100.0):
    ah = df[ah_col].to_numpy(dtype=float)

    # Panasonic 这里放电 Ah 通常为负值，所以 soc0 + ah / capacity * 100
    soc_ref = soc0 + ah / capacity_ah * 100.0
    soc_ref = np.clip(soc_ref, 0.0, 100.0)

    return soc_ref


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

    # Panasonic drive cycle 中，放电电流通常为负，充电/回充电流为正
    # 为了使用 SOC(k)=SOC(k-1)-I_discharge*dt/C，需要把放电电流转成正值
    if not discharge_current_positive:
        current_a = -current_a

    dt = np.diff(time_s, prepend=time_s[0])
    dt[0] = 0.0
    dt = np.maximum(dt, 0.0)

    capacity_coulomb = capacity_ah * 3600.0
    delta_soc_percent = current_a * dt / capacity_coulomb * 100.0

    soc = soc0 - np.cumsum(delta_soc_percent)
    soc = np.clip(soc, 0.0, 100.0)

    return soc


def load_ocv_lookup(lookup_path):
    lookup = pd.read_csv(lookup_path)

    required_cols = ["soc_percent", "ocv_V"]
    for col in required_cols:
        if col not in lookup.columns:
            raise ValueError(f"OCV lookup table missing column: {col}")

    lookup = lookup[required_cols].dropna().copy()

    # OCV -> SOC 插值需要以 OCV 从小到大排序
    lookup_for_interp = lookup.sort_values("ocv_V").drop_duplicates("ocv_V")

    ocv_values = lookup_for_interp["ocv_V"].to_numpy(dtype=float)
    soc_values = lookup_for_interp["soc_percent"].to_numpy(dtype=float)

    return lookup, ocv_values, soc_values


def voltage_to_soc_by_ocv_lookup(voltage_v, ocv_values, soc_values):
    voltage_v = np.asarray(voltage_v, dtype=float)

    # 超出 lookup table 范围的电压会自动 clip 到 0% 或 100%
    soc_est = np.interp(
        voltage_v,
        ocv_values,
        soc_values,
        left=soc_values[0],
        right=soc_values[-1],
    )

    soc_est = np.clip(soc_est, 0.0, 100.0)
    return soc_est


def calculate_rest_duration(time_s, current_a, threshold_a=0.05):
    time_s = np.asarray(time_s, dtype=float)
    current_a = np.asarray(current_a, dtype=float)

    dt = np.diff(time_s, prepend=time_s[0])
    dt[0] = 0.0
    dt = np.maximum(dt, 0.0)

    low_current = np.abs(current_a) <= threshold_a

    rest_duration = np.zeros_like(time_s, dtype=float)

    for k in range(1, len(time_s)):
        if low_current[k]:
            rest_duration[k] = rest_duration[k - 1] + dt[k]
        else:
            rest_duration[k] = 0.0

    return rest_duration, low_current


def ocv_corrected_coulomb_counting(
    df,
    ocv_values,
    soc_values,
    time_col="time_s",
    voltage_col="voltage_V",
    current_col="current_A",
    capacity_ah=2.9,
    soc0=100.0,
    discharge_current_positive=False,
    rest_current_threshold_a=0.05,
    min_rest_duration_s=30.0,
    min_correction_interval_s=300.0,
    correction_gain=0.20,
    max_correction_step_percent=1.0,
):
    df = df.copy().sort_values(time_col).reset_index(drop=True)

    time_s = df[time_col].to_numpy(dtype=float)
    voltage_v = df[voltage_col].to_numpy(dtype=float)
    current_raw = df[current_col].to_numpy(dtype=float)

    current_for_cc = current_raw.copy()
    if not discharge_current_positive:
        current_for_cc = -current_for_cc

    dt = np.diff(time_s, prepend=time_s[0])
    dt[0] = 0.0
    dt = np.maximum(dt, 0.0)

    capacity_coulomb = capacity_ah * 3600.0

    rest_duration_s, low_current_flag = calculate_rest_duration(
        time_s,
        current_raw,
        threshold_a=rest_current_threshold_a,
    )

    soc_ocv_raw = voltage_to_soc_by_ocv_lookup(voltage_v, ocv_values, soc_values)

    soc_corrected = np.zeros_like(time_s, dtype=float)
    soc_corrected[0] = soc0

    correction_applied = np.zeros_like(time_s, dtype=bool)
    ocv_innovation_percent = np.zeros_like(time_s, dtype=float)
    applied_correction_percent = np.zeros_like(time_s, dtype=float)

    last_correction_time = -1e18

    for k in range(1, len(time_s)):
        delta_soc = current_for_cc[k] * dt[k] / capacity_coulomb * 100.0
        soc_pred = soc_corrected[k - 1] - delta_soc
        soc_pred = np.clip(soc_pred, 0.0, 100.0)

        can_correct = (
            low_current_flag[k]
            and rest_duration_s[k] >= min_rest_duration_s
            and (time_s[k] - last_correction_time) >= min_correction_interval_s
        )

        if can_correct:
            soc_from_ocv = soc_ocv_raw[k]
            innovation = soc_from_ocv - soc_pred

            # 限制单次修正幅度，避免动态极化电压把 SOC 拉飞
            innovation_limited = np.clip(
                innovation,
                -max_correction_step_percent,
                max_correction_step_percent,
            )

            correction = correction_gain * innovation_limited
            soc_new = soc_pred + correction

            soc_corrected[k] = np.clip(soc_new, 0.0, 100.0)

            correction_applied[k] = True
            ocv_innovation_percent[k] = innovation
            applied_correction_percent[k] = correction
            last_correction_time = time_s[k]
        else:
            soc_corrected[k] = soc_pred

    diagnostic = pd.DataFrame({
        "rest_duration_s": rest_duration_s,
        "low_current_flag": low_current_flag,
        "soc_ocv_lookup_raw": soc_ocv_raw,
        "ocv_correction_applied": correction_applied,
        "ocv_innovation_percent": ocv_innovation_percent,
        "applied_correction_percent": applied_correction_percent,
    })

    return soc_corrected, soc_ocv_raw, diagnostic


# =========================
# Plot functions
# =========================
def plot_soc_comparison(df, file_name, output_dir):
    safe_name = Path(file_name).stem.replace(" ", "_")

    plt.figure(figsize=(10, 6))
    plt.plot(df["time_s"], df["soc_ref_ah"], label="Ah-based Reference SOC", linewidth=2)
    plt.plot(df["time_s"], df["soc_cc"], label="Coulomb Counting SOC", linestyle="--")
    plt.plot(df["time_s"], df["soc_ocv_lookup_raw"], label="Voltage-to-SOC Lookup", linestyle=":")
    plt.plot(df["time_s"], df["soc_ocv_corrected_cc"], label="OCV-corrected CC", linestyle="-.")

    correction_points = df[df["ocv_correction_applied"] == True]
    if len(correction_points) > 0:
        plt.scatter(
            correction_points["time_s"],
            correction_points["soc_ocv_corrected_cc"],
            s=20,
            label="OCV correction points",
            zorder=5,
        )

    plt.xlabel("Time (s)")
    plt.ylabel("SOC (%)")
    plt.title(f"SOC Comparison: {file_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / f"soc_comparison_{safe_name}.png", dpi=200)
    plt.close()


def plot_error_comparison(df, file_name, output_dir):
    safe_name = Path(file_name).stem.replace(" ", "_")

    plt.figure(figsize=(10, 6))
    plt.plot(df["time_s"], df["soc_cc"] - df["soc_ref_ah"], label="CC Error")
    plt.plot(df["time_s"], df["soc_ocv_lookup_raw"] - df["soc_ref_ah"], label="Voltage-to-SOC Lookup Error")
    plt.plot(df["time_s"], df["soc_ocv_corrected_cc"] - df["soc_ref_ah"], label="OCV-corrected CC Error")

    plt.xlabel("Time (s)")
    plt.ylabel("SOC Error (%)")
    plt.title(f"SOC Error Comparison: {file_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / f"error_comparison_{safe_name}.png", dpi=200)
    plt.close()


def plot_average_rmse(summary_df, output_dir):
    avg_rmse = (
        summary_df
        .groupby("method")["RMSE_percent"]
        .mean()
        .sort_values()
    )

    plt.figure(figsize=(8, 5))
    avg_rmse.plot(kind="bar")
    plt.ylabel("Average RMSE (%)")
    plt.xlabel("Method")
    plt.title("Average RMSE: CC vs OCV Lookup vs OCV-corrected CC")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(output_dir / "average_rmse_method_comparison.png", dpi=200)
    plt.close()


# =========================
# Main
# =========================
def main():
    print("Loading OCV-SOC lookup table:")
    print(OCV_LOOKUP_PATH)

    lookup_df, ocv_values, soc_values = load_ocv_lookup(OCV_LOOKUP_PATH)

    print("\nOCV lookup table preview:")
    print(lookup_df.head())
    print("...")
    print(lookup_df.tail())

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

    print("\n25degC drive cycle files selected:")
    print(drive_25[["file_name", "cycle_name", "output_csv"]])

    if len(drive_25) == 0:
        raise ValueError("No 25degC drive cycle files found.")

    records = []

    for idx, row in drive_25.iterrows():
        file_name = row["file_name"]
        file_path = row["output_csv"]

        print("\n========================================")
        print("Processing:", file_name)
        print(file_path)

        df = pd.read_csv(file_path)
        df = df.sort_values("time_s").reset_index(drop=True)

        required_cols = ["time_s", "voltage_V", "current_A", "ah"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"{file_name}: missing column {col}")

        # 1. Reference SOC from measured Ah
        df["soc_ref_ah"] = soc_from_ah(
            df,
            ah_col="ah",
            capacity_ah=CAPACITY_AH,
            soc0=SOC0,
        )

        # 2. Pure Coulomb Counting
        df["soc_cc"] = coulomb_counting(
            df,
            time_col="time_s",
            current_col="current_A",
            capacity_ah=CAPACITY_AH,
            soc0=SOC0,
            discharge_current_positive=False,
        )

        # 3. OCV lookup + OCV-corrected CC
        soc_corrected, soc_ocv_raw, diagnostic = ocv_corrected_coulomb_counting(
            df,
            ocv_values=ocv_values,
            soc_values=soc_values,
            time_col="time_s",
            voltage_col="voltage_V",
            current_col="current_A",
            capacity_ah=CAPACITY_AH,
            soc0=SOC0,
            discharge_current_positive=False,
            rest_current_threshold_a=REST_CURRENT_THRESHOLD_A,
            min_rest_duration_s=MIN_REST_DURATION_S,
            min_correction_interval_s=MIN_CORRECTION_INTERVAL_S,
            correction_gain=CORRECTION_GAIN,
            max_correction_step_percent=MAX_CORRECTION_STEP_PERCENT,
        )

        df["soc_ocv_lookup_raw"] = soc_ocv_raw
        df["soc_ocv_corrected_cc"] = soc_corrected

        for col in diagnostic.columns:
            df[col] = diagnostic[col]

        # Metrics
        method_outputs = {
            "coulomb_counting": df["soc_cc"],
            "voltage_to_soc_lookup_raw": df["soc_ocv_lookup_raw"],
            "ocv_corrected_coulomb_counting": df["soc_ocv_corrected_cc"],
        }

        correction_count = int(df["ocv_correction_applied"].sum())

        for method_name, y_pred in method_outputs.items():
            metrics = evaluate_soc(df["soc_ref_ah"], y_pred)

            record = {
                "file_name": file_name,
                "cycle_name": row.get("cycle_name", ""),
                "ambient_temp_C": row.get("ambient_temp_C", ""),
                "method": method_name,
                "capacity_ah": CAPACITY_AH,
                "initial_soc_percent": SOC0,
                "rest_current_threshold_A": REST_CURRENT_THRESHOLD_A,
                "min_rest_duration_s": MIN_REST_DURATION_S,
                "min_correction_interval_s": MIN_CORRECTION_INTERVAL_S,
                "correction_gain": CORRECTION_GAIN,
                "max_correction_step_percent": MAX_CORRECTION_STEP_PERCENT,
                "correction_count": correction_count,
                "duration_s": df["time_s"].iloc[-1] - df["time_s"].iloc[0],
                "final_soc_ref_percent": df["soc_ref_ah"].iloc[-1],
                "final_soc_pred_percent": float(y_pred.iloc[-1] if hasattr(y_pred, "iloc") else y_pred[-1]),
                **metrics,
            }

            records.append(record)

            print(
                f"{method_name}: "
                f"MAE={metrics['MAE_percent']:.4f}%, "
                f"RMSE={metrics['RMSE_percent']:.4f}%, "
                f"MAX={metrics['MAX_ERROR_percent']:.4f}%, "
                f"FINAL={metrics['FINAL_ERROR_percent']:.4f}%"
            )

        print(f"OCV correction count: {correction_count}")

        if correction_count == 0:
            print(
                "Note: No OCV correction was applied under the current rest-condition settings. "
                "This is possible for dynamic drive cycles."
            )

        # Save per-file result
        safe_name = Path(file_name).stem.replace(" ", "_")
        result_csv_path = OUTPUT_DIR / f"ocv_corrected_cc_result_{safe_name}.csv"
        df.to_csv(result_csv_path, index=False)

        plot_soc_comparison(df, file_name, OUTPUT_DIR)
        plot_error_comparison(df, file_name, OUTPUT_DIR)

        print("Saved result CSV to:")
        print(result_csv_path)

    summary_df = pd.DataFrame(records)

    summary_path = OUTPUT_DIR / "ocv_corrected_cc_25degC_drive_cycle_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    avg_metrics = (
        summary_df
        .groupby("method")[["MAE_percent", "RMSE_percent", "MAX_ERROR_percent", "FINAL_ERROR_percent"]]
        .mean()
        .reset_index()
    )

    avg_metrics_path = OUTPUT_DIR / "ocv_corrected_cc_average_metrics_by_method.csv"
    avg_metrics.to_csv(avg_metrics_path, index=False)

    plot_average_rmse(summary_df, OUTPUT_DIR)

    print("\n========================================")
    print("OCV-corrected Coulomb Counting finished.")
    print("Summary saved to:")
    print(summary_path)

    print("\nAverage metrics saved to:")
    print(avg_metrics_path)

    print("\nAverage metrics by method:")
    print(avg_metrics)

    print("\nPlots saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
