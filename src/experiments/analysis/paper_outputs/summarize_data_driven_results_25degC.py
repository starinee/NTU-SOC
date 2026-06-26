from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]
PROCESSED_DIR = PROJECT_ROOT / "dataset/processed"
OUTPUT_DIR = PROCESSED_DIR / "data_driven_comparison_25degC"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_FILES = [
    (
        "Instantaneous MLP",
        PROCESSED_DIR / "mlp_baseline_25degC/mlp_baseline_25degC_summary.csv",
    ),
    (
        "Filtered-feature MLP",
        PROCESSED_DIR / "mlp_filtered_features_25degC/mlp_filtered_features_25degC_summary.csv",
    ),
    (
        "LSTM",
        PROCESSED_DIR / "lstm_baseline_25degC/lstm_baseline_25degC_summary.csv",
    ),
    (
        "CNN-LSTM Teacher",
        PROCESSED_DIR / "cnn_lstm_teacher_25degC/cnn_lstm_teacher_25degC_summary.csv",
    ),
    (
        "Tiny CNN-LSTM Student",
        PROCESSED_DIR / "cnn_lstm_student_25degC/cnn_lstm_student_25degC_summary.csv",
    ),
    (
        "Distilled Tiny CNN-LSTM",
        PROCESSED_DIR
        / "cnn_lstm_distilled_student_25degC/cnn_lstm_distilled_student_25degC_summary.csv",
    ),
    (
        "Filtered-feature CNN-LSTM Teacher",
        PROCESSED_DIR
        / "cnn_lstm_teacher_filtered_features_25degC/cnn_lstm_teacher_filtered_features_25degC_summary.csv",
    ),
    (
        "Filtered-feature Tiny CNN-LSTM Student",
        PROCESSED_DIR
        / "cnn_lstm_student_filtered_features_25degC/cnn_lstm_student_filtered_features_25degC_summary.csv",
    ),
    (
        "Filtered-feature Distilled Tiny CNN-LSTM",
        PROCESSED_DIR
        / "cnn_lstm_distilled_student_filtered_features_25degC/cnn_lstm_distilled_student_filtered_features_25degC_summary.csv",
    ),
]

PERCENT_COLUMNS = [
    "MAE_percent",
    "RMSE_percent",
    "MAX_ERROR_percent",
    "P95_ABS_ERROR_percent",
    "P99_ABS_ERROR_percent",
    "FINAL_ERROR_percent",
]


def add_fraction_columns(df):
    df = df.copy()
    for col in PERCENT_COLUMNS:
        if col in df.columns:
            df[col.replace("_percent", "_fraction")] = df[col] / 100.0
    return df


def main():
    rows = []
    missing = []

    for method_name, path in SUMMARY_FILES:
        if not path.exists():
            missing.append(str(path))
            continue

        summary_df = add_fraction_columns(pd.read_csv(path))
        summary_with_fraction_path = path.with_name(path.stem + "_with_fraction.csv")
        summary_df.to_csv(summary_with_fraction_path, index=False)

        average_row = summary_df[summary_df["split"].eq("test_average")]
        if average_row.empty:
            continue

        record = average_row.iloc[0].to_dict()
        record["method"] = method_name
        record["source_summary"] = str(path)
        rows.append(record)

    if not rows:
        raise ValueError("No summary rows found.")

    comparison_df = pd.DataFrame(rows)
    ordered_columns = [
        "method",
        "sample_count",
        "MAE_percent",
        "MAE_fraction",
        "RMSE_percent",
        "RMSE_fraction",
        "MAX_ERROR_percent",
        "MAX_ERROR_fraction",
        "P95_ABS_ERROR_percent",
        "P95_ABS_ERROR_fraction",
        "P99_ABS_ERROR_percent",
        "P99_ABS_ERROR_fraction",
        "FINAL_ERROR_percent",
        "FINAL_ERROR_fraction",
        "source_summary",
    ]
    existing_columns = [col for col in ordered_columns if col in comparison_df.columns]
    comparison_df = comparison_df[existing_columns].sort_values("RMSE_percent")

    output_path = OUTPUT_DIR / "data_driven_methods_with_fraction_25degC_comparison.csv"
    paper_table_path = OUTPUT_DIR / "paper_table_data_driven_methods_with_fraction_25degC.csv"
    comparison_df.to_csv(output_path, index=False)

    paper_table = comparison_df[
        [
            "method",
            "MAE_percent",
            "MAE_fraction",
            "RMSE_percent",
            "RMSE_fraction",
            "MAX_ERROR_percent",
            "MAX_ERROR_fraction",
        ]
    ].copy()
    paper_table.to_csv(paper_table_path, index=False)

    print("Saved:")
    print(output_path)
    print(paper_table_path)
    if missing:
        print("\nSkipped missing summary files:")
        for path in missing:
            print(path)

    print("\nComparison:")
    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()
