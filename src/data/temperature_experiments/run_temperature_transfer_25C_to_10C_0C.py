from pathlib import Path
import importlib.util

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


BASE_SCRIPT = Path(__file__).resolve().with_name("run_within_temperature_full_pipeline_10C_0C.py")
spec = importlib.util.spec_from_file_location("temperature_pipeline", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = PROJECT_ROOT / "dataset/processed/panasonic_raw_csv/manifest.csv"
OUTPUT_ROOT = PROJECT_ROOT / "dataset/processed/temperature_experiments/temperature_transfer_25C_to_10C_0C"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

TRAIN_TEMPERATURES = [25]
TEST_TEMPERATURES = [25, 10, 0]
WITHIN_TEMPERATURE_SUMMARY = PROJECT_ROOT / "dataset/processed/temperature_experiments/within_temperature_full_pipeline_10C_0C"


def select_temperature_rows(manifest, temp_c):
    file_name = manifest["file_name"].astype(str)
    if temp_c == 0:
        temp_mask = file_name.str.contains("0degC", case=False, na=False) & ~file_name.str.contains(
            "10degC|20degC", case=False, na=False
        )
    elif temp_c == 10:
        temp_mask = file_name.str.contains("10degC", case=False, na=False) & ~file_name.str.contains(
            "n10degC|n20degC|-10degC|-20degC", case=False, na=False
        )
    else:
        temp_mask = file_name.str.contains(f"{temp_c}degC", case=False, na=False)

    rows = manifest[temp_mask & manifest["test_type"].isin(["cycle", "drive_cycle"])].copy()
    if "temperature_profile" in rows.columns:
        rows = rows[rows["temperature_profile"].eq("constant")].copy()
    rows = rows[
        ~rows["file_name"].astype(str).str.contains(
            "HWFT|HWFET|LA92_NN|US06_HWFET|HWFET_UDDS",
            case=False,
            regex=True,
            na=False,
        )
    ].copy()
    rows["split"] = rows.apply(
        lambda r: "train"
        if base.contains_any(r["file_name"], base.TRAIN_KEYWORDS)
        else ("test" if str(r.get("cycle_name", "")) in base.TEST_CYCLES else "unused"),
        axis=1,
    )
    rows = rows[rows["split"].isin(["train", "test"])].sort_values(["split", "file_name"])
    train_rows = rows[rows["split"].eq("train")].reset_index(drop=True)
    test_rows = rows[rows["split"].eq("test")].reset_index(drop=True)
    if len(train_rows) < 5 or len(test_rows) < 3:
        raise ValueError(f"{temp_c}C selected train={len(train_rows)}, test={len(test_rows)}")
    return train_rows, test_rows


def load_frames(rows, split):
    return [base.load_cycle(row, split, use_filtered=True) for _, row in rows.iterrows()]


def metric_row(train_temp, test_temp, method, frame, y, pred, parameter_count=np.nan):
    return {
        "train_temperature_C": train_temp,
        "test_temperature_C": test_temp,
        "split": "test",
        "method": method,
        "file_name": frame["file_name"].iloc[0],
        "cycle_name": frame["cycle_name"].iloc[0],
        "sample_count": len(y),
        "parameter_count": parameter_count,
        **base.evaluate(y, pred),
    }


def add_average(df):
    rows = []
    for keys, group in df.groupby(["train_temperature_C", "test_temperature_C", "method"], dropna=False):
        train_temp, test_temp, method = keys
        avg = {
            "train_temperature_C": train_temp,
            "test_temperature_C": test_temp,
            "split": "test_average",
            "method": method,
            "file_name": "AVERAGE_TEST_CYCLES",
            "cycle_name": "test_average",
            "sample_count": int(group["sample_count"].sum()),
            "parameter_count": group["parameter_count"].dropna().iloc[0] if group["parameter_count"].notna().any() else np.nan,
        }
        for col in [
            "MAE_percent",
            "RMSE_percent",
            "MAX_ERROR_percent",
            "P95_ABS_ERROR_percent",
            "P99_ABS_ERROR_percent",
            "FINAL_ERROR_percent",
        ]:
            avg[col] = float(group[col].mean())
        rows.append(avg)
    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True)


def evaluate_tabular_model(out_dir, train_temp, test_temp, method, model, test_frames, feature_cols, pred_col):
    rows = []
    pred_frames = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for frame in test_frames:
        cycle = frame.copy().reset_index(drop=True)
        y = cycle["soc_ref_ah"].to_numpy(dtype=float)
        pred = np.clip(model.predict(cycle[feature_cols].to_numpy(dtype=float)), 0.0, 100.0)
        cycle[pred_col] = pred
        cycle["soc_error_percent"] = pred - y
        rows.append(metric_row(train_temp, test_temp, method, cycle, y, pred))
        cycle.to_csv(out_dir / f"{pred_col}_prediction_{base.safe_name(cycle['file_name'].iloc[0])}.csv", index=False)
        pred_frames.append(cycle)
    pd.concat(pred_frames, ignore_index=True).to_csv(out_dir / f"{pred_col}_test_predictions.csv", index=False)
    return rows


def train_tabular_model(out_dir, method, train_frames, feature_cols, hidden_layers, max_iter, learning_rate):
    train_df = pd.concat(train_frames, ignore_index=True)
    model = base.make_mlp(hidden_layers, max_iter, learning_rate)
    print(f"Training {method} on 25C: samples={len(train_df)}, features={len(feature_cols)}")
    model.fit(train_df[feature_cols].to_numpy(dtype=float), train_df["soc_ref_ah"].to_numpy(dtype=float))
    out_dir.mkdir(parents=True, exist_ok=True)
    return model


def evaluate_torch_model(out_dir, train_temp, test_temp, method, model, test_frames, feature_cols, pred_col, mean, std, device):
    rows = []
    pred_frames = []
    out_dir.mkdir(parents=True, exist_ok=True)
    parameter_count = base.count_params(model)
    for frame in test_frames:
        cycle = frame.copy().reset_index(drop=True)
        x, y, idxs = base.build_windows(cycle, mean, std, base.TEST_STRIDE, feature_cols)
        pred = base.predict_torch(model, x, device)
        cycle[pred_col] = np.nan
        cycle.loc[idxs, pred_col] = pred
        cycle["soc_error_percent"] = cycle[pred_col] - cycle["soc_ref_ah"]
        rows.append(metric_row(train_temp, test_temp, method, cycle, y.reshape(-1), pred, parameter_count))
        cycle.to_csv(out_dir / f"{pred_col}_prediction_{base.safe_name(cycle['file_name'].iloc[0])}.csv", index=False)
        pred_frames.append(cycle)
    pd.concat(pred_frames, ignore_index=True).to_csv(out_dir / f"{pred_col}_test_predictions.csv", index=False)
    return rows


def train_torch_model(out_dir, method, train_frames, feature_cols, model_factory, device, teacher=None):
    train_df, mean, std = base.fit_scaler(train_frames, feature_cols)
    x_train, y_train = base.training_windows(train_frames, mean, std, feature_cols)
    model = model_factory()
    print(f"Training {method} on 25C: windows={x_train.shape}, params={base.count_params(model)}")
    model = base.train_torch(model, x_train, y_train, out_dir, device, max_epochs=60, patience=10, teacher=teacher)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_mean": mean,
            "feature_std": std,
            "feature_columns": feature_cols,
            "parameter_count": base.count_params(model),
        },
        out_dir / f"{method.replace(' ', '_').lower()}_25c_model.pt",
    )
    return model, mean, std


def run_coulomb_reference(test_frames_by_temp):
    rows = []
    for test_temp, test_frames in test_frames_by_temp.items():
        out_dir = OUTPUT_ROOT / "train_none" / f"test_{test_temp}degC" / "coulomb_counting"
        out_dir.mkdir(parents=True, exist_ok=True)
        pred_frames = []
        for frame in test_frames:
            cycle = frame.copy().reset_index(drop=True)
            y = cycle["soc_ref_ah"].to_numpy(dtype=float)
            pred = base.coulomb_counting(cycle)
            cycle["soc_cc"] = pred
            cycle["soc_error_percent"] = pred - y
            rows.append(metric_row("none", test_temp, "Coulomb Counting", cycle, y, pred))
            cycle.to_csv(out_dir / f"cc_prediction_{base.safe_name(cycle['file_name'].iloc[0])}.csv", index=False)
            pred_frames.append(cycle)
        pd.concat(pred_frames, ignore_index=True).to_csv(out_dir / "soc_cc_test_predictions.csv", index=False)
    return rows


def import_within_temperature_baselines(records):
    for temp in [10, 0]:
        path = WITHIN_TEMPERATURE_SUMMARY / f"{temp}degC" / "combined_method_summary.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df = df[df["split"].eq("test")].copy()
        df["train_temperature_C"] = temp
        df["test_temperature_C"] = temp
        records.append(df)
    return records


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


def plot_transfer_summary(avg):
    plot_dir = OUTPUT_ROOT / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)
    model_avg = avg[~avg["method"].eq("Coulomb Counting")].copy()
    model_avg["pair"] = model_avg["train_temperature_C"].astype(str) + "C->" + model_avg["test_temperature_C"].astype(str) + "C"

    fig, ax = plt.subplots(figsize=(13.5, 6.2))
    methods = [
        "Instantaneous MLP",
        "Filtered-feature MLP",
        "LSTM",
        "Filtered CNN-LSTM Teacher",
        "Filtered Tiny CNN-LSTM Student",
        "Filtered Distilled Tiny CNN-LSTM",
    ]
    pairs = ["25C->25C", "25C->10C", "25C->0C", "10C->10C", "0C->0C"]
    pivot = model_avg.pivot_table(index="pair", columns="method", values="RMSE_percent", aggfunc="mean").reindex(pairs)
    pivot = pivot[[m for m in methods if m in pivot.columns]]
    x = np.arange(len(pivot.index))
    width = min(0.12, 0.8 / max(len(pivot.columns), 1))
    offsets = (np.arange(len(pivot.columns)) - (len(pivot.columns) - 1) / 2) * width
    for i, method in enumerate(pivot.columns):
        ax.bar(x + offsets[i], pivot[method].to_numpy(), width=width, label=method)
    ax.set_title("Temperature transfer RMSE")
    ax.set_ylabel("RMSE (%SOC)")
    ax.set_xlabel("Train -> test temperature")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(plot_dir / "temperature_transfer_rmse_grouped.png", dpi=220)
    plt.close(fig)

    heat = model_avg.pivot_table(index="method", columns="pair", values="RMSE_percent", aggfunc="mean").reindex(methods)
    heat = heat[[p for p in pairs if p in heat.columns]]
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    im = ax.imshow(heat.to_numpy(), cmap="YlOrRd", aspect="auto")
    ax.set_title("Temperature transfer RMSE heatmap")
    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_xticklabels(heat.columns, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels(heat.index, fontsize=8)
    for y in range(heat.shape[0]):
        for x in range(heat.shape[1]):
            value = heat.iloc[y, x]
            if np.isfinite(value):
                ax.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="RMSE (%SOC)", shrink=0.82)
    fig.tight_layout()
    fig.savefig(plot_dir / "temperature_transfer_rmse_heatmap.png", dpi=220)
    plt.close(fig)


def main():
    manifest = pd.read_csv(MANIFEST_PATH)
    device = base.get_device()
    print(f"Using device: {device}")

    train_rows_by_temp = {}
    test_rows_by_temp = {}
    train_frames_by_temp = {}
    test_frames_by_temp = {}
    for temp in sorted(set(TRAIN_TEMPERATURES + TEST_TEMPERATURES)):
        train_rows, test_rows = select_temperature_rows(manifest, temp)
        temp_dir = OUTPUT_ROOT / "selected_rows"
        temp_dir.mkdir(parents=True, exist_ok=True)
        train_rows.to_csv(temp_dir / f"{temp}degC_train_rows.csv", index=False)
        test_rows.to_csv(temp_dir / f"{temp}degC_test_rows.csv", index=False)
        train_rows_by_temp[temp] = train_rows
        test_rows_by_temp[temp] = test_rows
        train_frames_by_temp[temp] = load_frames(train_rows, "train")
        test_frames_by_temp[temp] = load_frames(test_rows, "test")
        print(f"{temp}C: train={len(train_rows)}, test={len(test_rows)}")

    all_records = []
    all_records.extend(run_coulomb_reference(test_frames_by_temp))

    train_temp = 25
    train_frames = train_frames_by_temp[train_temp]
    train_dir = OUTPUT_ROOT / f"train_{train_temp}degC"

    instant = train_tabular_model(train_dir / "instantaneous_mlp", "Instantaneous MLP", train_frames, base.RAW_FEATURES, (64, 32), 300, 1e-3)
    filtered = train_tabular_model(train_dir / "filtered-feature_mlp", "Filtered-feature MLP", train_frames, base.FILTERED_INPUTS, (128, 64, 32), 500, 8e-4)

    lstm, lstm_mean, lstm_std = train_torch_model(
        train_dir / "lstm",
        "LSTM",
        train_frames,
        base.RAW_FEATURES,
        lambda: base.LSTMRegressor(len(base.RAW_FEATURES), hidden=32),
        device,
    )
    teacher, teacher_mean, teacher_std = train_torch_model(
        train_dir / "filtered_cnn-lstm_teacher",
        "Filtered CNN-LSTM Teacher",
        train_frames,
        base.FILTERED_INPUTS,
        lambda: base.CNNLSTMRegressor(len(base.FILTERED_INPUTS), conv_channels=64, lstm_hidden=64, dense_hidden=32),
        device,
    )
    tiny, tiny_mean, tiny_std = train_torch_model(
        train_dir / "filtered_tiny_cnn-lstm_student",
        "Filtered Tiny CNN-LSTM Student",
        train_frames,
        base.FILTERED_INPUTS,
        lambda: base.CNNLSTMRegressor(len(base.FILTERED_INPUTS), conv_channels=16, lstm_hidden=16, dense_hidden=8),
        device,
    )
    distilled, distilled_mean, distilled_std = train_torch_model(
        train_dir / "filtered_distilled_tiny_cnn-lstm",
        "Filtered Distilled Tiny CNN-LSTM",
        train_frames,
        base.FILTERED_INPUTS,
        lambda: base.CNNLSTMRegressor(len(base.FILTERED_INPUTS), conv_channels=16, lstm_hidden=16, dense_hidden=8),
        device,
        teacher=teacher,
    )

    for test_temp in TEST_TEMPERATURES:
        test_frames = test_frames_by_temp[test_temp]
        test_dir = train_dir / f"test_{test_temp}degC"
        all_records.extend(
            evaluate_tabular_model(
                test_dir / "instantaneous_mlp",
                train_temp,
                test_temp,
                "Instantaneous MLP",
                instant,
                test_frames,
                base.RAW_FEATURES,
                "soc_mlp",
            )
        )
        all_records.extend(
            evaluate_tabular_model(
                test_dir / "filtered-feature_mlp",
                train_temp,
                test_temp,
                "Filtered-feature MLP",
                filtered,
                test_frames,
                base.FILTERED_INPUTS,
                "soc_mlp_filtered",
            )
        )
        all_records.extend(
            evaluate_torch_model(
                test_dir / "lstm",
                train_temp,
                test_temp,
                "LSTM",
                lstm,
                test_frames,
                base.RAW_FEATURES,
                "soc_lstm",
                lstm_mean,
                lstm_std,
                device,
            )
        )
        all_records.extend(
            evaluate_torch_model(
                test_dir / "filtered_cnn-lstm_teacher",
                train_temp,
                test_temp,
                "Filtered CNN-LSTM Teacher",
                teacher,
                test_frames,
                base.FILTERED_INPUTS,
                "soc_cnn_lstm_teacher_filtered",
                teacher_mean,
                teacher_std,
                device,
            )
        )
        all_records.extend(
            evaluate_torch_model(
                test_dir / "filtered_tiny_cnn-lstm_student",
                train_temp,
                test_temp,
                "Filtered Tiny CNN-LSTM Student",
                tiny,
                test_frames,
                base.FILTERED_INPUTS,
                "soc_cnn_lstm_student_filtered",
                tiny_mean,
                tiny_std,
                device,
            )
        )
        all_records.extend(
            evaluate_torch_model(
                test_dir / "filtered_distilled_tiny_cnn-lstm",
                train_temp,
                test_temp,
                "Filtered Distilled Tiny CNN-LSTM",
                distilled,
                test_frames,
                base.FILTERED_INPUTS,
                "soc_cnn_lstm_distilled_student_filtered",
                distilled_mean,
                distilled_std,
                device,
            )
        )

    pieces = [pd.DataFrame(all_records)]
    pieces = import_within_temperature_baselines(pieces)
    summary = pd.concat(pieces, ignore_index=True, sort=False)
    cols = [
        "train_temperature_C",
        "test_temperature_C",
        "split",
        "method",
        "file_name",
        "cycle_name",
        "sample_count",
        "parameter_count",
        "MAE_percent",
        "RMSE_percent",
        "MAX_ERROR_percent",
        "P95_ABS_ERROR_percent",
        "P99_ABS_ERROR_percent",
        "FINAL_ERROR_percent",
    ]
    summary = summary[cols]
    summary = add_average(summary[summary["split"].eq("test")].copy())
    for col in [
        "MAE_percent",
        "RMSE_percent",
        "MAX_ERROR_percent",
        "P95_ABS_ERROR_percent",
        "P99_ABS_ERROR_percent",
        "FINAL_ERROR_percent",
    ]:
        summary[col] = pd.to_numeric(summary[col], errors="coerce").round(4)
    summary.to_csv(OUTPUT_ROOT / "temperature_transfer_full_summary.csv", index=False)

    avg = summary[summary["split"].eq("test_average")].copy()
    avg = avg.sort_values(["method", "train_temperature_C", "test_temperature_C"])
    avg.to_csv(OUTPUT_ROOT / "temperature_transfer_test_average.csv", index=False)
    write_markdown_table(avg, OUTPUT_ROOT / "temperature_transfer_test_average.md")

    profile = summary[summary["split"].eq("test")].copy()
    profile.to_csv(OUTPUT_ROOT / "temperature_transfer_profilewise_metrics.csv", index=False)
    write_markdown_table(profile, OUTPUT_ROOT / "temperature_transfer_profilewise_metrics.md")

    plot_transfer_summary(avg)
    print("\nTemperature transfer test averages:")
    print(avg.to_string(index=False))
    print(f"\nSaved outputs to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
