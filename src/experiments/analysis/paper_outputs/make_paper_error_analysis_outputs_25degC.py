from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]
PROCESSED_DIR = PROJECT_ROOT / "dataset/processed"
OUTPUT_DIR = PROCESSED_DIR / "paper_analysis_outputs_25degC"
TABLE_DIR = OUTPUT_DIR / "tables"
PLOT_DIR = OUTPUT_DIR / "plots"
PREDICTION_DIR = PLOT_DIR / "prediction_curves"
ABS_ERROR_DIR = PLOT_DIR / "absolute_error_over_time"
DIST_DIR = PLOT_DIR / "error_distribution"
REFERENCE_FIG_DIR = PLOT_DIR / "reference_existing_figures"

for directory in [TABLE_DIR, PREDICTION_DIR, ABS_ERROR_DIR, DIST_DIR, REFERENCE_FIG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


CYCLES = {
    "UDDS": {
        "summary_file": "03-21-17_00.29 25degC_UDDS_Pan18650PF.mat",
        "safe_p": "03-21-17_00p29_25degC_UDDS_Pan18650PF",
        "safe_dot": "03-21-17_00.29_25degC_UDDS_Pan18650PF",
    },
    "LA92": {
        "summary_file": "03-21-17_09.38 25degC_LA92_Pan18650PF.mat",
        "safe_p": "03-21-17_09p38_25degC_LA92_Pan18650PF",
        "safe_dot": "03-21-17_09.38_25degC_LA92_Pan18650PF",
    },
    "NN": {
        "summary_file": "03-21-17_16.27 25degC_NN_Pan18650PF.mat",
        "safe_p": "03-21-17_16p27_25degC_NN_Pan18650PF",
        "safe_dot": "03-21-17_16.27_25degC_NN_Pan18650PF",
    },
}


SUMMARY_SOURCES = [
    {
        "method": "Coulomb Counting",
        "group": "Traditional",
        "summary": PROCESSED_DIR / "baseline_results_batch/coulomb_counting_25degC_drive_cycle_summary.csv",
    },
    {
        "method": "OCV-corrected CC",
        "group": "Traditional",
        "summary": PROCESSED_DIR
        / "baseline_results_ocv_corrected/ocv_corrected_cc_25degC_drive_cycle_summary.csv",
        "filter_method": "ocv_corrected_coulomb_counting",
    },
    {
        "method": "Raw voltage lookup",
        "group": "Traditional",
        "summary": PROCESSED_DIR
        / "baseline_results_ocv_corrected/ocv_corrected_cc_25degC_drive_cycle_summary.csv",
        "filter_method": "voltage_to_soc_lookup_raw",
    },
    {
        "method": "Instantaneous MLP",
        "group": "Data-driven",
        "summary": PROCESSED_DIR / "mlp_baseline_25degC/mlp_baseline_25degC_summary.csv",
    },
    {
        "method": "Filtered-feature MLP",
        "group": "Data-driven",
        "summary": PROCESSED_DIR
        / "mlp_filtered_features_25degC/mlp_filtered_features_25degC_summary.csv",
    },
    {
        "method": "LSTM",
        "group": "Data-driven",
        "summary": PROCESSED_DIR / "lstm_baseline_25degC/lstm_baseline_25degC_summary.csv",
    },
    {
        "method": "Filtered CNN-LSTM Teacher",
        "group": "Data-driven",
        "summary": PROCESSED_DIR
        / "cnn_lstm_teacher_filtered_features_25degC/cnn_lstm_teacher_filtered_features_25degC_summary.csv",
    },
    {
        "method": "Filtered Tiny CNN-LSTM Student",
        "group": "Data-driven",
        "summary": PROCESSED_DIR
        / "cnn_lstm_student_filtered_features_25degC/cnn_lstm_student_filtered_features_25degC_summary.csv",
    },
    {
        "method": "Filtered Distilled Tiny CNN-LSTM",
        "group": "Data-driven",
        "summary": PROCESSED_DIR
        / "cnn_lstm_distilled_student_filtered_features_25degC/cnn_lstm_distilled_student_filtered_features_25degC_summary.csv",
    },
]


PREDICTION_SOURCES = {
    "Coulomb Counting": {
        "path_template": PROCESSED_DIR / "baseline_results_batch/cc_result_{safe_dot}.csv",
        "prediction_col": "soc_cc",
    },
    "Instantaneous MLP": {
        "path_template": PROCESSED_DIR / "mlp_baseline_25degC/mlp_prediction_{safe_p}.csv",
        "prediction_col": "soc_mlp",
    },
    "Filtered-feature MLP": {
        "path_template": PROCESSED_DIR / "mlp_filtered_features_25degC/mlp_filtered_prediction_{safe_p}.csv",
        "prediction_col": "soc_mlp_filtered",
    },
    "LSTM": {
        "path_template": PROCESSED_DIR / "lstm_baseline_25degC/lstm_prediction_{safe_p}.csv",
        "prediction_col": "soc_lstm",
    },
    "Filtered CNN-LSTM Teacher": {
        "path_template": PROCESSED_DIR
        / "cnn_lstm_teacher_filtered_features_25degC/soc_cnn_lstm_teacher_filtered_prediction_{safe_p}.csv",
        "prediction_col": "soc_cnn_lstm_teacher_filtered",
    },
    "Filtered Tiny CNN-LSTM Student": {
        "path_template": PROCESSED_DIR
        / "cnn_lstm_student_filtered_features_25degC/soc_cnn_lstm_student_filtered_prediction_{safe_p}.csv",
        "prediction_col": "soc_cnn_lstm_student_filtered",
    },
    "Filtered Distilled Tiny CNN-LSTM": {
        "path_template": PROCESSED_DIR
        / "cnn_lstm_distilled_student_filtered_features_25degC/soc_cnn_lstm_distilled_student_filtered_prediction_{safe_p}.csv",
        "prediction_col": "soc_cnn_lstm_distilled_student_filtered",
    },
}


PLOT_METHODS = [
    "Coulomb Counting",
    "Instantaneous MLP",
    "Filtered-feature MLP",
    "LSTM",
    "Filtered CNN-LSTM Teacher",
    "Filtered Distilled Tiny CNN-LSTM",
]


def to_markdown_table(df):
    display_df = df.copy().where(pd.notna(df), "")
    columns = list(display_df.columns)
    rows = display_df.astype(str).values.tolist()
    widths = []
    for col_idx, col in enumerate(columns):
        widths.append(max(len(str(col)), *[len(row[col_idx]) for row in rows]))

    def format_row(values):
        return "| " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(values)) + " |"

    return "\n".join(
        [
            format_row(columns),
            "| " + " | ".join("-" * w for w in widths) + " |",
            *[format_row(row) for row in rows],
        ]
    ) + "\n"


def save_table(df, stem):
    csv_path = TABLE_DIR / f"{stem}.csv"
    md_path = TABLE_DIR / f"{stem}.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(to_markdown_table(df), encoding="utf-8")
    return csv_path, md_path


def build_per_cycle_error_table():
    records = []
    for source in SUMMARY_SOURCES:
        df = pd.read_csv(source["summary"])
        if "filter_method" in source:
            df = df[df["method"].eq(source["filter_method"])]
        df = df[df["split"].eq("test")] if "split" in df.columns else df

        for cycle, meta in CYCLES.items():
            row = df[df["file_name"].eq(meta["summary_file"])]
            if row.empty:
                continue
            row = row.iloc[0]
            records.append(
                {
                    "group": source["group"],
                    "method": source["method"],
                    "cycle": cycle,
                    "sample_count": int(row.get("sample_count", np.nan))
                    if pd.notna(row.get("sample_count", np.nan))
                    else "",
                    "MAE_percent": row["MAE_percent"],
                    "RMSE_percent": row["RMSE_percent"],
                    "MAX_ERROR_percent": row["MAX_ERROR_percent"],
                    "FINAL_ERROR_percent": row.get("FINAL_ERROR_percent", np.nan),
                }
            )

    table = pd.DataFrame(records)
    table["MAE_fraction"] = table["MAE_percent"] / 100.0
    table["RMSE_fraction"] = table["RMSE_percent"] / 100.0
    table["MAX_ERROR_fraction"] = table["MAX_ERROR_percent"] / 100.0

    cycle_order = {cycle: i for i, cycle in enumerate(CYCLES)}
    method_order = {source["method"]: i for i, source in enumerate(SUMMARY_SOURCES)}
    table["_cycle_order"] = table["cycle"].map(cycle_order)
    table["_method_order"] = table["method"].map(method_order)
    table = table.sort_values(["_cycle_order", "_method_order"]).drop(
        columns=["_cycle_order", "_method_order"]
    )

    paper = table.copy()
    for col in [
        "MAE_percent",
        "RMSE_percent",
        "MAX_ERROR_percent",
        "FINAL_ERROR_percent",
    ]:
        paper[col] = pd.to_numeric(paper[col], errors="coerce").round(3)
    for col in ["MAE_fraction", "RMSE_fraction", "MAX_ERROR_fraction"]:
        paper[col] = paper[col].round(5)

    save_table(table, "per_drive_cycle_error_table_full_precision")
    save_table(paper, "per_drive_cycle_error_table_paper")
    return paper


def path_for(template, meta):
    return Path(str(template).format(**meta))


def load_prediction(cycle, method):
    meta = CYCLES[cycle]
    spec = PREDICTION_SOURCES[method]
    path = path_for(spec["path_template"], meta)
    df = pd.read_csv(path)
    pred_col = spec["prediction_col"]
    keep = df[["time_s", "soc_ref_ah", pred_col]].copy()
    keep = keep.rename(columns={pred_col: "soc_pred"})
    keep["method"] = method
    keep["cycle"] = cycle
    keep = keep.dropna(subset=["soc_ref_ah", "soc_pred"]).reset_index(drop=True)
    keep["error_percent"] = keep["soc_pred"] - keep["soc_ref_ah"]
    keep["abs_error_percent"] = keep["error_percent"].abs()
    return keep


def downsample_for_plot(df, max_points=2500):
    if len(df) <= max_points:
        return df
    step = int(np.ceil(len(df) / max_points))
    return df.iloc[::step].copy()


def make_prediction_plots():
    for cycle in CYCLES:
        first = load_prediction(cycle, "Filtered-feature MLP")
        ref = downsample_for_plot(first[["time_s", "soc_ref_ah"]].drop_duplicates())

        plt.figure(figsize=(11, 6))
        plt.plot(ref["time_s"], ref["soc_ref_ah"], color="black", linewidth=2.2, label="Reference SOC")
        for method in PLOT_METHODS:
            pred = downsample_for_plot(load_prediction(cycle, method))
            plt.plot(pred["time_s"], pred["soc_pred"], linewidth=1.35, alpha=0.9, label=method)
        plt.xlabel("Time (s)")
        plt.ylabel("SOC (%)")
        plt.title(f"SOC Prediction Comparison on Unseen {cycle} Drive Cycle")
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(PREDICTION_DIR / f"core_models_soc_prediction_{cycle}.png", dpi=220)
        plt.close()


def make_absolute_error_plots():
    for cycle in CYCLES:
        plt.figure(figsize=(11, 5.5))
        for method in PLOT_METHODS:
            pred = downsample_for_plot(load_prediction(cycle, method))
            plt.plot(pred["time_s"], pred["abs_error_percent"], linewidth=1.2, alpha=0.9, label=method)
        plt.xlabel("Time (s)")
        plt.ylabel("Absolute SOC Error (%)")
        plt.title(f"Absolute Error over Time on Unseen {cycle} Drive Cycle")
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(ABS_ERROR_DIR / f"core_models_absolute_error_over_time_{cycle}.png", dpi=220)
        plt.close()


def make_error_distribution_and_max_error_table():
    all_errors = []
    max_records = []
    for cycle in CYCLES:
        for method in PLOT_METHODS:
            pred = load_prediction(cycle, method)
            all_errors.append(pred[["cycle", "method", "abs_error_percent", "error_percent"]])
            max_idx = pred["abs_error_percent"].idxmax()
            max_row = pred.loc[max_idx]
            max_records.append(
                {
                    "cycle": cycle,
                    "method": method,
                    "max_abs_error_percent": max_row["abs_error_percent"],
                    "signed_error_percent": max_row["error_percent"],
                    "time_s": max_row["time_s"],
                    "reference_soc_percent": max_row["soc_ref_ah"],
                    "predicted_soc_percent": max_row["soc_pred"],
                }
            )

    error_df = pd.concat(all_errors, ignore_index=True)
    max_df = pd.DataFrame(max_records)

    max_paper = max_df.copy()
    for col in [
        "max_abs_error_percent",
        "signed_error_percent",
        "time_s",
        "reference_soc_percent",
        "predicted_soc_percent",
    ]:
        max_paper[col] = max_paper[col].round(3)
    save_table(max_df, "max_error_locations_full_precision")
    save_table(max_paper, "max_error_locations_paper")

    plt.figure(figsize=(12, 6))
    ordered_methods = PLOT_METHODS
    data = [
        error_df.loc[error_df["method"].eq(method), "abs_error_percent"].to_numpy()
        for method in ordered_methods
    ]
    plt.boxplot(data, labels=ordered_methods, showfliers=False)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Absolute SOC Error (%)")
    plt.title("Absolute Error Distribution across UDDS, LA92, and NN")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(DIST_DIR / "core_models_absolute_error_distribution_boxplot.png", dpi=220)
    plt.close()

    plt.figure(figsize=(12, 6))
    for method in ordered_methods:
        vals = error_df.loc[error_df["method"].eq(method), "abs_error_percent"].to_numpy()
        vals = np.sort(vals)
        cumulative = np.arange(1, len(vals) + 1) / len(vals)
        plt.plot(vals, cumulative, linewidth=1.8, label=method)
    plt.xlabel("Absolute SOC Error (%)")
    plt.ylabel("Cumulative Probability")
    plt.title("Cumulative Distribution of Absolute SOC Error")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(DIST_DIR / "core_models_absolute_error_cdf.png", dpi=220)
    plt.close()

    return max_paper


def copy_reference_existing_figures():
    existing = [
        (
            PROCESSED_DIR
            / "baseline_results_sensitivity/sensitivity_soc_curve_03-21-17_00p29_25degC_UDDS_Pan18650PF.png",
            "cc_sensitivity_soc_curve_UDDS.png",
        ),
        (
            PROCESSED_DIR
            / "baseline_results_sensitivity/sensitivity_error_curve_03-21-17_00p29_25degC_UDDS_Pan18650PF.png",
            "cc_sensitivity_error_curve_UDDS.png",
        ),
        (
            PROCESSED_DIR
            / "baseline_results_sensitivity/average_rmse_by_sensitivity_case.png",
            "cc_sensitivity_average_rmse_by_case.png",
        ),
    ]
    for src, name in existing:
        if src.exists():
            shutil.copy2(src, REFERENCE_FIG_DIR / name)


def main():
    per_cycle = build_per_cycle_error_table()
    make_prediction_plots()
    make_absolute_error_plots()
    max_errors = make_error_distribution_and_max_error_table()
    copy_reference_existing_figures()

    readme = f"""Paper analysis outputs for 25degC SOC estimation experiments.

Generated contents:
- tables/per_drive_cycle_error_table_paper.csv and .md
- tables/max_error_locations_paper.csv and .md
- plots/prediction_curves/core_models_soc_prediction_UDDS/LA92/NN.png
- plots/absolute_error_over_time/core_models_absolute_error_over_time_UDDS/LA92/NN.png
- plots/error_distribution/core_models_absolute_error_distribution_boxplot.png
- plots/error_distribution/core_models_absolute_error_cdf.png
- plots/reference_existing_figures/ copied CC sensitivity figures

Core models used in comparison:
{', '.join(PLOT_METHODS)}
"""
    (OUTPUT_DIR / "README.txt").write_text(readme, encoding="utf-8")

    print("Saved paper analysis outputs to:")
    print(OUTPUT_DIR)
    print("\nPer-drive-cycle table preview:")
    print(per_cycle.to_string(index=False))
    print("\nMax-error location preview:")
    print(max_errors.to_string(index=False))


if __name__ == "__main__":
    main()
