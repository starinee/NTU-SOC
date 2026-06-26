#!/usr/bin/env python3
"""Compare voltage-lookup baselines and filtered-feature models under dynamic load.

This script answers the paper-analysis question:
does terminal-voltage lookup degrade more when current magnitude or dynamic
intensity increases, while filtered-feature learning models stay more stable?

It uses existing 25degC prediction files and writes summary tables/figures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]
PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed"
OUTPUT_DIR = (
    PROCESSED_DIR
    / "traditional_baselines"
    / "voltage_lookup_vs_filtered_dynamic_25C"
)

OCV_DIR = PROCESSED_DIR / "baseline_results_ocv_corrected"
MLP_FILTERED_FILE = (
    PROCESSED_DIR
    / "mlp_filtered_features_25degC"
    / "mlp_filtered_features_25degC_test_predictions.csv"
)
CNN_FILTERED_FILE = (
    PROCESSED_DIR
    / "cnn_lstm_teacher_filtered_features_25degC"
    / "soc_cnn_lstm_teacher_filtered_test_predictions.csv"
)

METHOD_ORDER = [
    "Terminal-voltage lookup",
    "OCV-corrected CC",
    "Filtered-feature MLP",
    "Filtered CNN-LSTM Teacher",
]


def rmse(values: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def load_ocv_baselines() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    files = sorted(OCV_DIR.glob("ocv_corrected_cc_result_*_25degC_*.csv"))
    if not files:
        raise FileNotFoundError(f"No OCV-corrected CC prediction files found in {OCV_DIR}")

    for csv_path in files:
        df = pd.read_csv(csv_path).sort_values("time_s").reset_index(drop=True)

        # The learned-model predictions are stored after the same 10x downsampling
        # used in the 25degC pipeline. Downsampling this high-rate baseline keeps
        # dynamic/current bin comparisons on a similar sampling density.
        df = df.iloc[::10].copy()

        for method, pred_col in [
            ("Terminal-voltage lookup", "soc_ocv_lookup_raw"),
            ("OCV-corrected CC", "soc_ocv_corrected_cc"),
        ]:
            part = df[
                [
                    "file_name",
                    "cycle_name",
                    "time_s",
                    "current_A",
                    "voltage_V",
                    "soc_ref_ah",
                    pred_col,
                ]
            ].copy()
            part = part.rename(columns={pred_col: "soc_pred"})
            part["method"] = method
            frames.append(part)

    return pd.concat(frames, ignore_index=True)


def load_learning_model(csv_path: Path, pred_col: str, method: str) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    required = ["file_name", "cycle_name", "time_s", "current_A", "voltage_V", "soc_ref_ah", pred_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing columns: {missing}")

    part = df[required].copy()
    part = part.rename(columns={pred_col: "soc_pred"})
    part = part.dropna(subset=["soc_pred", "soc_ref_ah", "current_A", "time_s"])
    part["method"] = method
    return part


def add_dynamic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["abs_current_A"] = df["current_A"].abs()
    df["dynamic_intensity_A_per_s"] = 0.0

    group_cols = ["method", "file_name"]
    for _, idx in df.groupby(group_cols, sort=False).groups.items():
        idx = list(idx)
        part = df.loc[idx].sort_values("time_s")
        dt = part["time_s"].diff().replace(0, np.nan)
        di = part["current_A"].diff().abs()
        slew = (di / dt).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        dynamic = slew.rolling(window=30, min_periods=1).mean()
        df.loc[part.index, "dynamic_intensity_A_per_s"] = dynamic.to_numpy()

    df["soc_error_percent"] = df["soc_pred"] - df["soc_ref_ah"]
    df["abs_soc_error_percent"] = df["soc_error_percent"].abs()
    return df


def add_quantile_bins(df: pd.DataFrame, source_col: str, target_col: str) -> pd.DataFrame:
    labels = ["Low", "Medium", "High"]
    ranked = df[source_col].rank(method="first")
    df[target_col] = pd.qcut(ranked, q=3, labels=labels)
    return df


def summarize_by_bin(df: pd.DataFrame, bin_col: str) -> pd.DataFrame:
    rows = []
    for (method, bin_name), part in df.groupby(["method", bin_col], observed=False):
        rows.append(
            {
                "method": method,
                "bin": str(bin_name),
                "sample_count": int(len(part)),
                "rmse_percent": rmse(part["soc_error_percent"]),
                "mae_percent": float(part["abs_soc_error_percent"].mean()),
                "p95_abs_error_percent": float(part["abs_soc_error_percent"].quantile(0.95)),
                "mean_abs_current_A": float(part["abs_current_A"].mean()),
                "mean_dynamic_intensity_A_per_s": float(part["dynamic_intensity_A_per_s"].mean()),
            }
        )

    out = pd.DataFrame(rows)
    out["method"] = pd.Categorical(out["method"], METHOD_ORDER, ordered=True)
    out["bin"] = pd.Categorical(out["bin"], ["Low", "Medium", "High"], ordered=True)
    return out.sort_values(["method", "bin"]).reset_index(drop=True)


def write_markdown_table(df: pd.DataFrame, path: Path, title: str) -> None:
    display = df.copy()
    for col in display.select_dtypes(include=[np.number]).columns:
        if col != "sample_count":
            display[col] = display[col].round(4)

    headers = list(display.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    path.write_text(f"# {title}\n\n" + "\n".join(lines) + "\n", encoding="utf-8")


def plot_grouped_rmse(df: pd.DataFrame, path: Path, title: str, xlabel: str) -> None:
    pivot = df.pivot(index="bin", columns="method", values="rmse_percent").loc[
        ["Low", "Medium", "High"], METHOD_ORDER
    ]
    ax = pivot.plot(kind="bar", figsize=(10.5, 5.5), width=0.78)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("SOC RMSE (%)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="", ncols=2, fontsize=8)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    long_df = pd.concat(
        [
            load_ocv_baselines(),
            load_learning_model(
                MLP_FILTERED_FILE,
                pred_col="soc_mlp_filtered",
                method="Filtered-feature MLP",
            ),
            load_learning_model(
                CNN_FILTERED_FILE,
                pred_col="soc_cnn_lstm_teacher_filtered",
                method="Filtered CNN-LSTM Teacher",
            ),
        ],
        ignore_index=True,
    )
    long_df = add_dynamic_features(long_df)
    long_df = add_quantile_bins(long_df, "abs_current_A", "current_magnitude_bin")
    long_df = add_quantile_bins(long_df, "dynamic_intensity_A_per_s", "dynamic_intensity_bin")

    long_df.to_csv(OUTPUT_DIR / "sample_level_comparison_25C.csv", index=False)

    current_summary = summarize_by_bin(long_df, "current_magnitude_bin")
    dynamic_summary = summarize_by_bin(long_df, "dynamic_intensity_bin")

    current_summary.to_csv(OUTPUT_DIR / "current_magnitude_comparison_25C.csv", index=False)
    dynamic_summary.to_csv(OUTPUT_DIR / "dynamic_intensity_comparison_25C.csv", index=False)
    write_markdown_table(
        current_summary,
        OUTPUT_DIR / "current_magnitude_comparison_25C.md",
        "25degC Error vs Current Magnitude",
    )
    write_markdown_table(
        dynamic_summary,
        OUTPUT_DIR / "dynamic_intensity_comparison_25C.md",
        "25degC Error vs Dynamic Intensity",
    )

    plot_grouped_rmse(
        current_summary,
        OUTPUT_DIR / "current_magnitude_rmse_comparison_25C.png",
        "25degC SOC RMSE by Current Magnitude",
        "Current magnitude bin",
    )
    plot_grouped_rmse(
        dynamic_summary,
        OUTPUT_DIR / "dynamic_intensity_rmse_comparison_25C.png",
        "25degC SOC RMSE by Dynamic Intensity",
        "Dynamic intensity bin",
    )

    readme = """# Voltage Lookup vs Filtered Dynamic Analysis

This folder compares traditional voltage-based baselines against filtered-feature
learning models on the same 25degC test profiles.

- `Terminal-voltage lookup`: directly maps loaded terminal voltage to SOC using the
  OCV-SOC curve.
- `OCV-corrected CC`: propagates SOC by Coulomb Counting and applies OCV-based
  corrections during low-current/rest-like periods.
- `Filtered-feature MLP`: lightweight tabular model with moving-average voltage,
  current, power, and current-change features.
- `Filtered CNN-LSTM Teacher`: sequence model using the filtered feature set.

The current-magnitude and dynamic-intensity bins are quantile bins over the
sample-level comparison table. Dynamic intensity is measured as a rolling mean of
absolute current slew rate.
"""
    (OUTPUT_DIR / "README.md").write_text(readme, encoding="utf-8")

    print(f"Wrote analysis outputs to {OUTPUT_DIR}")
    print(dynamic_summary.to_string(index=False))


if __name__ == "__main__":
    main()
