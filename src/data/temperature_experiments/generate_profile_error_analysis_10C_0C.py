from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = PROJECT_ROOT / "dataset/processed/temperature_experiments/within_temperature_full_pipeline_10C_0C"
ANALYSIS_DIR = PROJECT_ROOT / "dataset/processed/temperature_experiments/profile_error_analysis_10C_0C"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

TEMPERATURES = [10, 0]
SOC_BINS = [0, 20, 40, 60, 80, 100]
CURRENT_BINS = [0, 0.5, 1.0, 2.0, 3.0, 5.0, np.inf]
CURRENT_LABELS = ["0-0.5A", "0.5-1A", "1-2A", "2-3A", "3-5A", ">5A"]

METHOD_FILES = {
    "Coulomb Counting": ("traditional_cc/cc_prediction_*.csv", "soc_cc"),
    "Instantaneous MLP": ("instantaneous_mlp/soc_mlp_test_predictions.csv", "soc_mlp"),
    "Filtered-feature MLP": ("filtered-feature_mlp/soc_mlp_filtered_test_predictions.csv", "soc_mlp_filtered"),
    "LSTM": ("lstm/soc_lstm_test_predictions.csv", "soc_lstm"),
    "Filtered CNN-LSTM Teacher": (
        "filtered_cnn-lstm_teacher/soc_cnn_lstm_teacher_filtered_test_predictions.csv",
        "soc_cnn_lstm_teacher_filtered",
    ),
    "Filtered Tiny CNN-LSTM Student": (
        "filtered_tiny_cnn-lstm_student/soc_cnn_lstm_student_filtered_test_predictions.csv",
        "soc_cnn_lstm_student_filtered",
    ),
    "Filtered Distilled Tiny CNN-LSTM": (
        "filtered_distilled_tiny_cnn-lstm/soc_cnn_lstm_distilled_student_filtered_test_predictions.csv",
        "soc_cnn_lstm_distilled_student_filtered",
    ),
}

TIME_PLOT_METHODS = [
    "Coulomb Counting",
    "Instantaneous MLP",
    "Filtered-feature MLP",
    "LSTM",
    "Filtered CNN-LSTM Teacher",
    "Filtered Tiny CNN-LSTM Student",
]

METHOD_COLORS = {
    "Coulomb Counting": "#6b7280",
    "Instantaneous MLP": "#d97706",
    "Filtered-feature MLP": "#0f766e",
    "LSTM": "#2563eb",
    "Filtered CNN-LSTM Teacher": "#7c3aed",
    "Filtered Tiny CNN-LSTM Student": "#db2777",
    "Filtered Distilled Tiny CNN-LSTM": "#991b1b",
}


def rmse(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    return float(np.sqrt(np.mean(values**2)))


def short_method(method):
    return (
        method.replace("Filtered-feature MLP", "Filtered MLP")
        .replace("Filtered CNN-LSTM Teacher", "Teacher")
        .replace("Filtered Tiny CNN-LSTM Student", "Tiny Student")
        .replace("Filtered Distilled Tiny CNN-LSTM", "Distilled Tiny")
        .replace("Coulomb Counting", "CC")
        .replace("Instantaneous MLP", "Instant MLP")
    )


def write_markdown_table(df, path):
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


def collect_predictions():
    frames = []
    for temp in TEMPERATURES:
        temp_dir = OUTPUT_ROOT / f"{temp}degC"
        for method, (pattern, pred_col) in METHOD_FILES.items():
            for path in sorted(temp_dir.glob(pattern)):
                df = pd.read_csv(path)
                if pred_col not in df.columns:
                    continue
                keep = [
                    "time_s",
                    "voltage_V",
                    "current_A",
                    "battery_temp_C",
                    "cycle_name",
                    "file_name",
                    "soc_ref_ah",
                    pred_col,
                    "soc_error_percent",
                ]
                missing = [c for c in keep if c not in df.columns]
                if missing:
                    raise ValueError(f"{path} missing columns: {missing}")
                part = df[keep].rename(columns={pred_col: "soc_pred"}).copy()
                part["temperature_C"] = temp
                part["method"] = method
                part = part.dropna(subset=["soc_pred", "soc_error_percent", "soc_ref_ah", "current_A"])
                frames.append(part)
    if not frames:
        raise RuntimeError("No prediction files found.")
    all_pred = pd.concat(frames, ignore_index=True)
    all_pred["abs_error_percent"] = all_pred["soc_error_percent"].abs()
    all_pred["abs_current_A"] = all_pred["current_A"].abs()
    all_pred["elapsed_min"] = all_pred.groupby(["temperature_C", "method", "file_name"])["time_s"].transform(
        lambda s: (s - s.iloc[0]) / 60.0
    )
    all_pred["dt_s"] = all_pred.groupby(["temperature_C", "method", "file_name"])["time_s"].diff()
    all_pred["dcurrent_A"] = all_pred.groupby(["temperature_C", "method", "file_name"])["current_A"].diff()
    all_pred["dynamic_intensity_A_per_s"] = (all_pred["dcurrent_A"] / all_pred["dt_s"]).abs().replace([np.inf, -np.inf], np.nan)
    all_pred["dynamic_intensity_A_per_s"] = all_pred["dynamic_intensity_A_per_s"].fillna(0.0)
    all_pred.to_csv(ANALYSIS_DIR / "all_test_predictions_long.csv", index=False)
    return all_pred


def build_profilewise_tables():
    rows = []
    for temp in TEMPERATURES:
        summary = pd.read_csv(OUTPUT_ROOT / f"{temp}degC" / "combined_method_summary.csv")
        summary = summary[summary["split"].eq("test")].copy()
        summary["temperature_C"] = temp
        rows.append(summary)
    table = pd.concat(rows, ignore_index=True)
    cols = [
        "temperature_C",
        "method",
        "cycle_name",
        "sample_count",
        "MAE_percent",
        "RMSE_percent",
        "MAX_ERROR_percent",
        "P95_ABS_ERROR_percent",
        "P99_ABS_ERROR_percent",
        "FINAL_ERROR_percent",
    ]
    table = table[cols].sort_values(["temperature_C", "cycle_name", "method"])
    table.to_csv(ANALYSIS_DIR / "profilewise_metrics.csv", index=False)
    write_markdown_table(table, ANALYSIS_DIR / "profilewise_metrics.md")
    return table


def plot_profilewise_rmse(profile_table):
    for temp in TEMPERATURES:
        temp_table = profile_table[profile_table["temperature_C"].eq(temp)].copy()
        pivot = temp_table.pivot(index="cycle_name", columns="method", values="RMSE_percent")
        pivot = pivot[[m for m in METHOD_FILES if m in pivot.columns]]

        fig, ax = plt.subplots(figsize=(13, 5.8))
        x = np.arange(len(pivot.index))
        width = 0.11
        offsets = (np.arange(len(pivot.columns)) - (len(pivot.columns) - 1) / 2) * width
        for i, method in enumerate(pivot.columns):
            ax.bar(
                x + offsets[i],
                pivot[method].to_numpy(),
                width=width,
                label=short_method(method),
                color=METHOD_COLORS.get(method),
            )
        ax.set_title(f"Profile-wise RMSE at {temp}C")
        ax.set_ylabel("RMSE (%SOC)")
        ax.set_xlabel("Test profile")
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(ncol=4, fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(ANALYSIS_DIR / f"profilewise_rmse_{temp}degC.png", dpi=220)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    for ax, temp in zip(axes, TEMPERATURES):
        temp_table = profile_table[profile_table["temperature_C"].eq(temp)].copy()
        pivot = temp_table.pivot(index="method", columns="cycle_name", values="RMSE_percent")
        pivot = pivot.loc[[m for m in METHOD_FILES if m in pivot.index]]
        im = ax.imshow(pivot.to_numpy(), cmap="YlGnBu", aspect="auto")
        ax.set_title(f"{temp}C")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([short_method(m) for m in pivot.index], fontsize=8)
        for y in range(pivot.shape[0]):
            for x in range(pivot.shape[1]):
                ax.text(x, y, f"{pivot.iloc[y, x]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=axes.ravel().tolist(), label="RMSE (%SOC)", shrink=0.82)
    fig.suptitle("Profile-wise RMSE heatmap")
    fig.savefig(ANALYSIS_DIR / "profilewise_rmse_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_error_vs_time(all_pred):
    for temp in TEMPERATURES:
        for cycle in ["UDDS", "LA92", "NN"]:
            data = all_pred[
                all_pred["temperature_C"].eq(temp)
                & all_pred["cycle_name"].eq(cycle)
                & all_pred["method"].isin(TIME_PLOT_METHODS)
            ].copy()
            if data.empty:
                continue
            fig, ax = plt.subplots(figsize=(12, 5.5))
            for method in TIME_PLOT_METHODS:
                part = data[data["method"].eq(method)].sort_values("elapsed_min")
                if part.empty:
                    continue
                smoothed = part["soc_error_percent"].rolling(31, min_periods=1, center=True).mean()
                ax.plot(
                    part["elapsed_min"],
                    smoothed,
                    label=short_method(method),
                    linewidth=1.35,
                    color=METHOD_COLORS.get(method),
                )
            ax.axhline(0, color="black", linewidth=0.8, alpha=0.55)
            ax.set_title(f"Error vs time, {temp}C {cycle}")
            ax.set_xlabel("Elapsed time (min)")
            ax.set_ylabel("SOC error (%SOC)")
            ax.grid(alpha=0.22)
            ax.legend(ncol=3, fontsize=8, frameon=False)
            fig.tight_layout()
            fig.savefig(ANALYSIS_DIR / f"error_vs_time_{temp}degC_{cycle}.png", dpi=220)
            plt.close(fig)


def summarize_bins(all_pred):
    data = all_pred.copy()
    data["soc_range"] = pd.cut(data["soc_ref_ah"], SOC_BINS, include_lowest=True, labels=["0-20", "20-40", "40-60", "60-80", "80-100"])
    data["current_bin"] = pd.cut(data["abs_current_A"], CURRENT_BINS, include_lowest=True, labels=CURRENT_LABELS)

    q = data["dynamic_intensity_A_per_s"].quantile([0.33, 0.66]).to_numpy()
    if not np.isfinite(q).all() or q[0] == q[1]:
        q = np.array([0.02, 0.1])
    dynamic_bins = [-np.inf, q[0], q[1], np.inf]
    data["dynamic_bin"] = pd.cut(data["dynamic_intensity_A_per_s"], dynamic_bins, labels=["low", "medium", "high"])

    tables = {}
    for name, col in [("soc_range", "soc_range"), ("current_bin", "current_bin"), ("dynamic_bin", "dynamic_bin")]:
        table = (
            data.groupby(["temperature_C", "method", col], observed=True)
            .agg(
                sample_count=("soc_error_percent", "size"),
                MAE_percent=("abs_error_percent", "mean"),
                RMSE_percent=("soc_error_percent", rmse),
                P95_ABS_ERROR_percent=("abs_error_percent", lambda s: float(np.quantile(s, 0.95))),
            )
            .reset_index()
        )
        table.to_csv(ANALYSIS_DIR / f"error_by_{name}.csv", index=False)
        write_markdown_table(table, ANALYSIS_DIR / f"error_by_{name}.md")
        tables[name] = table
    return tables


def plot_bin_bars(table, bin_col, title, filename):
    for temp in TEMPERATURES:
        temp_table = table[table["temperature_C"].eq(temp)].copy()
        if temp_table.empty:
            continue
        pivot = temp_table.pivot(index=bin_col, columns="method", values="RMSE_percent")
        pivot = pivot[[m for m in METHOD_FILES if m in pivot.columns]]
        fig, ax = plt.subplots(figsize=(13, 5.8))
        x = np.arange(len(pivot.index))
        width = min(0.11, 0.8 / max(len(pivot.columns), 1))
        offsets = (np.arange(len(pivot.columns)) - (len(pivot.columns) - 1) / 2) * width
        for i, method in enumerate(pivot.columns):
            ax.bar(
                x + offsets[i],
                pivot[method].to_numpy(),
                width=width,
                label=short_method(method),
                color=METHOD_COLORS.get(method),
            )
        ax.set_title(f"{title} at {temp}C")
        ax.set_xlabel(bin_col.replace("_", " "))
        ax.set_ylabel("RMSE (%SOC)")
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index.astype(str))
        ax.grid(axis="y", alpha=0.25)
        ax.legend(ncol=4, fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(ANALYSIS_DIR / f"{filename}_{temp}degC.png", dpi=220)
        plt.close(fig)


def main():
    print(f"Writing analysis figures to {ANALYSIS_DIR}")
    all_pred = collect_predictions()
    profile_table = build_profilewise_tables()
    plot_profilewise_rmse(profile_table)
    plot_error_vs_time(all_pred)
    tables = summarize_bins(all_pred)
    plot_bin_bars(tables["soc_range"], "soc_range", "Error vs SOC range", "error_vs_soc_range")
    plot_bin_bars(tables["current_bin"], "current_bin", "Error vs current magnitude", "error_vs_current_magnitude")
    plot_bin_bars(tables["dynamic_bin"], "dynamic_bin", "Error vs dynamic intensity", "error_vs_dynamic_intensity")
    print("Done.")


if __name__ == "__main__":
    main()
