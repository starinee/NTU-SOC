from pathlib import Path

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[4]
PROCESSED_DIR = PROJECT_ROOT / "dataset/processed"
OUTPUT_DIR = PROCESSED_DIR / "final_paper_tables_25degC"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


METHODS = [
    {
        "method": "Instantaneous MLP",
        "group": "Basic baseline",
        "feature_set": "Raw V, I, T",
        "model_type": "MLP",
        "summary": PROCESSED_DIR / "mlp_baseline_25degC/mlp_baseline_25degC_summary.csv",
        "parameter_count": "mlp_3_64_32",
        "reference_teacher": None,
        "paper_note": "Simple data-driven baseline without temporal feature engineering.",
    },
    {
        "method": "Filtered-feature MLP",
        "group": "Strong lightweight baseline",
        "feature_set": "Raw + EMA filtered V/I",
        "model_type": "MLP",
        "summary": PROCESSED_DIR
        / "mlp_filtered_features_25degC/mlp_filtered_features_25degC_summary.csv",
        "parameter_count": "mlp_9_128_64_32",
        "reference_teacher": None,
        "paper_note": "Best lightweight model in the current 25degC split.",
    },
    {
        "method": "LSTM",
        "group": "Sequence baseline",
        "feature_set": "Raw V, I, T",
        "model_type": "LSTM",
        "summary": PROCESSED_DIR / "lstm_baseline_25degC/lstm_baseline_25degC_summary.csv",
        "checkpoint": PROCESSED_DIR / "lstm_baseline_25degC/lstm_baseline_25degC_model.pt",
        "reference_teacher": None,
        "paper_note": "Strong recurrent sequence baseline using 60-s windows.",
    },
    {
        "method": "CNN-LSTM Teacher",
        "group": "CNN-LSTM compression",
        "feature_set": "Raw V, I, T",
        "model_type": "CNN-LSTM teacher",
        "summary": PROCESSED_DIR
        / "cnn_lstm_teacher_25degC/cnn_lstm_teacher_25degC_summary.csv",
        "checkpoint": PROCESSED_DIR / "cnn_lstm_teacher_25degC/cnn_lstm_teacher_25degC_model.pt",
        "reference_teacher": "CNN-LSTM Teacher",
        "paper_note": "Original teacher model without filtered features.",
    },
    {
        "method": "Tiny CNN-LSTM Student",
        "group": "CNN-LSTM compression",
        "feature_set": "Raw V, I, T",
        "model_type": "Tiny CNN-LSTM student",
        "summary": PROCESSED_DIR
        / "cnn_lstm_student_25degC/cnn_lstm_student_25degC_summary.csv",
        "checkpoint": PROCESSED_DIR / "cnn_lstm_student_25degC/cnn_lstm_student_25degC_model.pt",
        "reference_teacher": "CNN-LSTM Teacher",
        "paper_note": "Compact student trained directly with ground-truth SOC labels.",
    },
    {
        "method": "Distilled Tiny CNN-LSTM",
        "group": "CNN-LSTM compression",
        "feature_set": "Raw V, I, T",
        "model_type": "Distilled tiny CNN-LSTM",
        "summary": PROCESSED_DIR
        / "cnn_lstm_distilled_student_25degC/cnn_lstm_distilled_student_25degC_summary.csv",
        "checkpoint": PROCESSED_DIR
        / "cnn_lstm_distilled_student_25degC/cnn_lstm_distilled_student_25degC_model.pt",
        "reference_teacher": "CNN-LSTM Teacher",
        "paper_note": "Knowledge-distilled compact student from the original teacher.",
    },
    {
        "method": "Filtered-feature CNN-LSTM Teacher",
        "group": "Filtered CNN-LSTM compression",
        "feature_set": "Raw + EMA filtered V/I",
        "model_type": "Filtered CNN-LSTM teacher",
        "summary": PROCESSED_DIR
        / "cnn_lstm_teacher_filtered_features_25degC/cnn_lstm_teacher_filtered_features_25degC_summary.csv",
        "checkpoint": PROCESSED_DIR
        / "cnn_lstm_teacher_filtered_features_25degC/cnn_lstm_teacher_filtered_features_25degC_model.pt",
        "reference_teacher": "Filtered-feature CNN-LSTM Teacher",
        "paper_note": "Improved teacher using filtered dynamic features.",
    },
    {
        "method": "Filtered-feature Tiny CNN-LSTM Student",
        "group": "Filtered CNN-LSTM compression",
        "feature_set": "Raw + EMA filtered V/I",
        "model_type": "Filtered tiny CNN-LSTM student",
        "summary": PROCESSED_DIR
        / "cnn_lstm_student_filtered_features_25degC/cnn_lstm_student_filtered_features_25degC_summary.csv",
        "checkpoint": PROCESSED_DIR
        / "cnn_lstm_student_filtered_features_25degC/cnn_lstm_student_filtered_features_25degC_model.pt",
        "reference_teacher": "Filtered-feature CNN-LSTM Teacher",
        "paper_note": "Compact filtered-feature student trained directly.",
    },
    {
        "method": "Filtered-feature Distilled Tiny CNN-LSTM",
        "group": "Filtered CNN-LSTM compression",
        "feature_set": "Raw + EMA filtered V/I",
        "model_type": "Filtered distilled tiny CNN-LSTM",
        "summary": PROCESSED_DIR
        / "cnn_lstm_distilled_student_filtered_features_25degC/cnn_lstm_distilled_student_filtered_features_25degC_summary.csv",
        "checkpoint": PROCESSED_DIR
        / "cnn_lstm_distilled_student_filtered_features_25degC/cnn_lstm_distilled_student_filtered_features_25degC_model.pt",
        "reference_teacher": "Filtered-feature CNN-LSTM Teacher",
        "paper_note": "Distilled compact model using the filtered-feature teacher.",
    },
]


def mlp_parameter_count(input_size, hidden_layers):
    sizes = [input_size, *hidden_layers, 1]
    return int(sum((sizes[i] + 1) * sizes[i + 1] for i in range(len(sizes) - 1)))


def checkpoint_parameter_count(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "parameter_count" in payload:
        return int(payload["parameter_count"])
    state_dict = payload["model_state_dict"]
    return int(sum(tensor.numel() for tensor in state_dict.values()))


def get_parameter_count(method):
    spec = method.get("parameter_count")
    if spec == "mlp_3_64_32":
        return mlp_parameter_count(3, [64, 32])
    if spec == "mlp_9_128_64_32":
        return mlp_parameter_count(9, [128, 64, 32])
    checkpoint = method.get("checkpoint")
    if checkpoint:
        return checkpoint_parameter_count(checkpoint)
    return None


def read_test_average(path):
    df = pd.read_csv(path)
    row = df[df["split"].eq("test_average")]
    if row.empty:
        raise ValueError(f"No test_average row in {path}")
    return row.iloc[0].to_dict()


def add_fraction_metrics(record):
    for col in [
        "MAE_percent",
        "RMSE_percent",
        "MAX_ERROR_percent",
        "P95_ABS_ERROR_percent",
        "P99_ABS_ERROR_percent",
        "FINAL_ERROR_percent",
    ]:
        if col in record and pd.notna(record[col]):
            record[col.replace("_percent", "_fraction")] = record[col] / 100.0
    return record


def build_tables():
    rows = []
    for method in METHODS:
        metrics = add_fraction_metrics(read_test_average(method["summary"]))
        row = {
            "group": method["group"],
            "method": method["method"],
            "feature_set": method["feature_set"],
            "model_type": method["model_type"],
            "sample_count": int(metrics["sample_count"]),
            "MAE_percent": metrics.get("MAE_percent"),
            "MAE_fraction": metrics.get("MAE_fraction"),
            "RMSE_percent": metrics.get("RMSE_percent"),
            "RMSE_fraction": metrics.get("RMSE_fraction"),
            "MAX_ERROR_percent": metrics.get("MAX_ERROR_percent"),
            "MAX_ERROR_fraction": metrics.get("MAX_ERROR_fraction"),
            "P95_ABS_ERROR_percent": metrics.get("P95_ABS_ERROR_percent"),
            "P95_ABS_ERROR_fraction": metrics.get("P95_ABS_ERROR_fraction"),
            "P99_ABS_ERROR_percent": metrics.get("P99_ABS_ERROR_percent"),
            "P99_ABS_ERROR_fraction": metrics.get("P99_ABS_ERROR_fraction"),
            "FINAL_ERROR_percent": metrics.get("FINAL_ERROR_percent"),
            "FINAL_ERROR_fraction": metrics.get("FINAL_ERROR_fraction"),
            "parameter_count": get_parameter_count(method),
            "reference_teacher": method["reference_teacher"],
            "paper_note": method["paper_note"],
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    teacher_params = dict(zip(df["method"], df["parameter_count"]))

    df["params_vs_reference_teacher_percent"] = pd.NA
    df["parameter_reduction_vs_teacher_percent"] = pd.NA
    for idx, row in df.iterrows():
        ref_name = row["reference_teacher"]
        if not ref_name or ref_name not in teacher_params:
            continue
        ref_params = teacher_params[ref_name]
        if pd.isna(row["parameter_count"]) or not ref_params:
            continue
        ratio = row["parameter_count"] / ref_params * 100.0
        df.loc[idx, "params_vs_reference_teacher_percent"] = ratio
        df.loc[idx, "parameter_reduction_vs_teacher_percent"] = 100.0 - ratio

    performance_columns = [
        "group",
        "method",
        "feature_set",
        "sample_count",
        "MAE_percent",
        "MAE_fraction",
        "RMSE_percent",
        "RMSE_fraction",
        "MAX_ERROR_percent",
        "MAX_ERROR_fraction",
        "P95_ABS_ERROR_percent",
        "P99_ABS_ERROR_percent",
        "paper_note",
    ]
    complexity_columns = [
        "group",
        "method",
        "model_type",
        "feature_set",
        "parameter_count",
        "reference_teacher",
        "params_vs_reference_teacher_percent",
        "parameter_reduction_vs_teacher_percent",
        "RMSE_percent",
        "MAE_percent",
        "MAX_ERROR_percent",
    ]

    performance_df = df[performance_columns].copy()
    complexity_df = df[complexity_columns].copy()
    complexity_df = complexity_df.sort_values(["group", "parameter_count", "RMSE_percent"])
    return performance_df, complexity_df


def rounded_for_paper(df):
    out = df.copy()
    for col in out.columns:
        if col.endswith("_percent"):
            out[col] = pd.to_numeric(out[col], errors="coerce").round(3)
        elif col.endswith("_fraction"):
            out[col] = pd.to_numeric(out[col], errors="coerce").round(5)
        elif col == "params_vs_reference_teacher_percent":
            out[col] = pd.to_numeric(out[col], errors="coerce").round(2)
        elif col == "parameter_reduction_vs_teacher_percent":
            out[col] = pd.to_numeric(out[col], errors="coerce").round(2)
    return out


def save_table(df, stem):
    csv_path = OUTPUT_DIR / f"{stem}.csv"
    md_path = OUTPUT_DIR / f"{stem}.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(to_markdown_table(df), encoding="utf-8")
    return csv_path, md_path


def to_markdown_table(df):
    display_df = df.copy()
    display_df = display_df.where(pd.notna(display_df), "")
    columns = list(display_df.columns)
    rows = display_df.astype(str).values.tolist()

    widths = []
    for col_idx, col in enumerate(columns):
        cell_widths = [len(row[col_idx]) for row in rows] if rows else [0]
        widths.append(max(len(str(col)), *cell_widths))

    def format_row(values):
        return "| " + " | ".join(
            str(value).ljust(widths[idx]) for idx, value in enumerate(values)
        ) + " |"

    header = format_row(columns)
    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [format_row(row) for row in rows]
    return "\n".join([header, separator, *body]) + "\n"


def main():
    performance_df, complexity_df = build_tables()
    performance_paper = rounded_for_paper(performance_df)
    complexity_paper = rounded_for_paper(complexity_df)

    paths = []
    paths.extend(save_table(performance_df, "final_data_driven_performance_table_full_precision"))
    paths.extend(save_table(performance_paper, "final_data_driven_performance_table_paper"))
    paths.extend(save_table(complexity_df, "model_complexity_and_compression_table_full_precision"))
    paths.extend(save_table(complexity_paper, "model_complexity_and_compression_table_paper"))

    print("Saved final paper tables:")
    for path in paths:
        print(path)

    print("\nFinal performance table:")
    print(to_markdown_table(performance_paper))

    print("\nModel complexity and compression table:")
    print(to_markdown_table(complexity_paper))


if __name__ == "__main__":
    main()
