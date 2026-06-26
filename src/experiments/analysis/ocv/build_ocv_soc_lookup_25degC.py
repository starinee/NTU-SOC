import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# =========================
# Path settings
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[4]
MANIFEST_PATH = PROJECT_ROOT / "dataset/processed/panasonic_raw_csv/manifest.csv"

OUTPUT_DIR = PROJECT_ROOT / "dataset/processed/ocv_lookup"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# We identified this file as the best C20 OCV-like source
OCV_FILE_KEYWORD = "C20 OCV Test_C20_25dC"

# Threshold for selecting C20 discharge branch
# C20 current is around 2.9Ah / 20 = 0.145A
DISCHARGE_CURRENT_THRESHOLD = -0.05


def find_ocv_source_file(manifest: pd.DataFrame) -> pd.Series:
    """
    Find the 25degC C20 OCV test file from manifest.
    """
    candidates = manifest[
        manifest["output_csv"].astype(str).str.contains("25degC", case=False, na=False)
        &
        manifest["file_name"].astype(str).str.contains(OCV_FILE_KEYWORD, case=False, na=False)
    ].copy()

    print("OCV source candidates:")
    if len(candidates) > 0:
        print(candidates[["file_name", "test_type", "output_csv"]].to_string())
    else:
        print("No candidates found by file_name. Trying output_csv keyword search...")

        candidates = manifest[
            manifest["output_csv"].astype(str).str.contains("25degC", case=False, na=False)
            &
            manifest["output_csv"].astype(str).str.contains("C20", case=False, na=False)
            &
            manifest["output_csv"].astype(str).str.contains("OCV", case=False, na=False)
        ].copy()

        print(candidates[["file_name", "test_type", "output_csv"]].to_string())

    if len(candidates) == 0:
        raise ValueError("No 25degC C20 OCV source file found.")

    if len(candidates) > 1:
        print("\nWarning: multiple candidates found. Using the first one.")

    return candidates.iloc[0]


def extract_c20_discharge_branch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract only the C20 discharge branch.

    In this Panasonic file:
    - current_A < 0 means discharge
    - current_A around -0.145 A means C20 discharge
    """
    required_cols = ["time_s", "voltage_V", "current_A", "ah"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df = df.sort_values("time_s").reset_index(drop=True)

    # keep valid rows
    df = df.dropna(subset=required_cols).reset_index(drop=True)

    # select C20 discharge period
    dis = df[df["current_A"] < DISCHARGE_CURRENT_THRESHOLD].copy()

    if len(dis) == 0:
        raise ValueError(
            "No discharge branch found. Please check current sign or current threshold."
        )

    dis = dis.reset_index(drop=True)

    # normalize time
    dis["time_from_discharge_start_s"] = dis["time_s"] - dis["time_s"].iloc[0]

    # Ah in the file decreases during discharge.
    # Use measured C20 discharged capacity to normalize SOC.
    ah_start = dis["ah"].iloc[0]
    dis["discharged_ah"] = ah_start - dis["ah"]

    measured_capacity_ah = dis["discharged_ah"].max()

    if measured_capacity_ah <= 0:
        raise ValueError("Measured discharged capacity is non-positive. Check ah column.")

    # Define SOC from 100% to 0% using measured C20 capacity
    dis["soc_percent"] = 100.0 * (1.0 - dis["discharged_ah"] / measured_capacity_ah)
    dis["soc_percent"] = dis["soc_percent"].clip(0.0, 100.0)

    # In C20 low-current condition, terminal voltage is used as quasi-OCV
    dis["ocv_V"] = dis["voltage_V"]

    return dis


def build_lookup_table(dis: pd.DataFrame) -> pd.DataFrame:
    """
    Build 1%-resolution SOC-OCV lookup table.

    Output:
    soc_percent: 0, 1, 2, ..., 100
    ocv_V: interpolated OCV at each SOC point
    """
    temp = dis[["soc_percent", "ocv_V"]].copy()

    # For bin averaging
    temp["soc_bin"] = temp["soc_percent"].round().astype(int)
    temp["soc_bin"] = temp["soc_bin"].clip(0, 100)

    binned = (
        temp
        .groupby("soc_bin")
        .agg(
            ocv_V_mean=("ocv_V", "mean"),
            ocv_V_std=("ocv_V", "std"),
            sample_count=("ocv_V", "count"),
        )
        .reset_index()
        .rename(columns={"soc_bin": "soc_percent"})
        .sort_values("soc_percent")
    )

    # Interpolate to complete 0-100 grid
    grid = np.arange(0, 101, 1)

    x = binned["soc_percent"].to_numpy(dtype=float)
    y = binned["ocv_V_mean"].to_numpy(dtype=float)

    # Ensure x is increasing and unique
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    ocv_interp = np.interp(grid, x, y)

    lookup = pd.DataFrame({
        "soc_percent": grid,
        "ocv_V": ocv_interp,
    })

    # Add raw bin information where available
    lookup = lookup.merge(
        binned,
        on="soc_percent",
        how="left"
    )

    return lookup


def plot_discharge_segment(dis: pd.DataFrame, output_dir: Path):
    """
    Save checking plots for selected C20 discharge branch.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(dis["time_from_discharge_start_s"], dis["voltage_V"])
    plt.xlabel("Time from discharge start (s)")
    plt.ylabel("Voltage (V)")
    plt.title("Selected C20 Discharge Branch: Voltage vs Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "c20_discharge_voltage_vs_time_25degC.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(dis["time_from_discharge_start_s"], dis["current_A"])
    plt.xlabel("Time from discharge start (s)")
    plt.ylabel("Current (A)")
    plt.title("Selected C20 Discharge Branch: Current vs Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "c20_discharge_current_vs_time_25degC.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(dis["time_from_discharge_start_s"], dis["soc_percent"])
    plt.xlabel("Time from discharge start (s)")
    plt.ylabel("SOC (%)")
    plt.title("Selected C20 Discharge Branch: SOC vs Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "c20_discharge_soc_vs_time_25degC.png", dpi=200)
    plt.close()


def plot_ocv_soc_curve(dis: pd.DataFrame, lookup: pd.DataFrame, output_dir: Path):
    """
    Save OCV-SOC curve.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(dis["soc_percent"], dis["ocv_V"], s=8, alpha=0.4, label="C20 discharge raw points")
    plt.plot(lookup["soc_percent"], lookup["ocv_V"], linewidth=2, label="1% SOC lookup table")
    plt.xlabel("SOC (%)")
    plt.ylabel("OCV / quasi-OCV (V)")
    plt.title("OCV-SOC Lookup Curve at 25°C")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "ocv_soc_curve_25degC.png", dpi=200)
    plt.close()

    # Also save reversed x-axis version, often used in battery papers
    plt.figure(figsize=(8, 6))
    plt.plot(lookup["soc_percent"], lookup["ocv_V"], linewidth=2)
    plt.gca().invert_xaxis()
    plt.xlabel("SOC (%)")
    plt.ylabel("OCV / quasi-OCV (V)")
    plt.title("OCV-SOC Lookup Curve at 25°C")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "ocv_soc_curve_25degC_reversed_x.png", dpi=200)
    plt.close()


def main():
    manifest = pd.read_csv(MANIFEST_PATH)

    source_row = find_ocv_source_file(manifest)

    file_name = source_row["file_name"]
    file_path = source_row["output_csv"]

    print("\nSelected OCV source file:")
    print(file_name)
    print(file_path)

    df = pd.read_csv(file_path)

    print("\nOriginal file shape:", df.shape)
    print("Original columns:")
    print(df.columns.tolist())

    dis = extract_c20_discharge_branch(df)

    measured_capacity_ah = dis["discharged_ah"].max()

    print("\nSelected C20 discharge branch shape:", dis.shape)
    print("\nC20 discharge branch description:")
    print(dis[["time_from_discharge_start_s", "voltage_V", "current_A", "ah", "discharged_ah", "soc_percent"]].describe())

    print("\nEstimated measured C20 discharge capacity:")
    print(f"{measured_capacity_ah:.6f} Ah")

    print("\nVoltage range in selected discharge branch:")
    print(f"{dis['voltage_V'].min():.6f} V to {dis['voltage_V'].max():.6f} V")

    lookup = build_lookup_table(dis)

    # Add metadata
    lookup["temperature_C"] = 25
    lookup["source_file"] = file_name
    lookup["measured_capacity_ah"] = measured_capacity_ah
    lookup["note"] = "OCV approximated using C20 low-current discharge branch"

    # Save outputs
    raw_segment_path = OUTPUT_DIR / "c20_discharge_segment_25degC.csv"
    lookup_path = OUTPUT_DIR / "ocv_soc_lookup_25degC.csv"

    dis.to_csv(raw_segment_path, index=False)
    lookup.to_csv(lookup_path, index=False)

    plot_discharge_segment(dis, OUTPUT_DIR)
    plot_ocv_soc_curve(dis, lookup, OUTPUT_DIR)

    print("\nSaved C20 discharge segment to:")
    print(raw_segment_path)

    print("\nSaved OCV-SOC lookup table to:")
    print(lookup_path)

    print("\nLookup table preview:")
    print(lookup[["soc_percent", "ocv_V", "sample_count"]].head(10))
    print("...")
    print(lookup[["soc_percent", "ocv_V", "sample_count"]].tail(10))

    print("\nPlots saved to:")
    print(OUTPUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
