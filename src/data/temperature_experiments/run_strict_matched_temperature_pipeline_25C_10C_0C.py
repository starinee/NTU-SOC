from pathlib import Path
import importlib.util

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


BASE_SCRIPT = Path(__file__).resolve().with_name("run_within_temperature_full_pipeline_10C_0C.py")
spec = importlib.util.spec_from_file_location("temperature_base", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = PROJECT_ROOT / "dataset/processed/panasonic_raw_csv/manifest.csv"
OUTPUT_ROOT = PROJECT_ROOT / "dataset/processed/temperature_experiments/strict_matched_temperature_pipeline_25C_10C_0C"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

TEMPERATURES = [25, 10, 0]
TRANSFER_TEST_TEMPERATURES = [25, 10, 0]


MODEL_SPECS = [
    {
        "method": "Instantaneous MLP",
        "kind": "tabular",
        "feature_cols": base.RAW_FEATURES,
        "pred_col": "soc_mlp",
        "hidden_layers": (64, 32),
        "max_iter": 300,
        "learning_rate": 1e-3,
    },
    {
        "method": "Filtered-feature MLP",
        "kind": "tabular",
        "feature_cols": base.FILTERED_INPUTS,
        "pred_col": "soc_mlp_filtered",
        "hidden_layers": (128, 64, 32),
        "max_iter": 500,
        "learning_rate": 8e-4,
    },
    {
        "method": "LSTM",
        "kind": "torch",
        "feature_cols": base.RAW_FEATURES,
        "pred_col": "soc_lstm",
        "factory": lambda: base.LSTMRegressor(len(base.RAW_FEATURES), hidden=32),
        "max_epochs": 80,
        "patience": 12,
        "learning_rate": 1e-3,
        "teacher": None,
    },
    {
        "method": "CNN-LSTM Teacher",
        "kind": "torch",
        "feature_cols": base.RAW_FEATURES,
        "pred_col": "soc_cnn_lstm_teacher",
        "factory": lambda: base.CNNLSTMRegressor(len(base.RAW_FEATURES), conv_channels=64, lstm_hidden=64, dense_hidden=32),
        "max_epochs": 80,
        "patience": 12,
        "learning_rate": 1e-3,
        "teacher": None,
    },
    {
        "method": "Tiny CNN-LSTM Student",
        "kind": "torch",
        "feature_cols": base.RAW_FEATURES,
        "pred_col": "soc_cnn_lstm_student",
        "factory": lambda: base.CNNLSTMRegressor(len(base.RAW_FEATURES), conv_channels=16, lstm_hidden=16, dense_hidden=8),
        "max_epochs": 80,
        "patience": 12,
        "learning_rate": 1e-3,
        "teacher": None,
    },
    {
        "method": "Distilled Tiny CNN-LSTM",
        "kind": "torch",
        "feature_cols": base.RAW_FEATURES,
        "pred_col": "soc_cnn_lstm_distilled_student",
        "factory": lambda: base.CNNLSTMRegressor(len(base.RAW_FEATURES), conv_channels=16, lstm_hidden=16, dense_hidden=8),
        "max_epochs": 80,
        "patience": 12,
        "learning_rate": 1e-3,
        "teacher": "CNN-LSTM Teacher",
    },
    {
        "method": "Filtered CNN-LSTM Teacher",
        "kind": "torch",
        "feature_cols": base.FILTERED_INPUTS,
        "pred_col": "soc_cnn_lstm_teacher_filtered",
        "factory": lambda: base.CNNLSTMRegressor(len(base.FILTERED_INPUTS), conv_channels=64, lstm_hidden=64, dense_hidden=32),
        "max_epochs": 100,
        "patience": 15,
        "learning_rate": 8e-4,
        "teacher": None,
    },
    {
        "method": "Filtered Tiny CNN-LSTM Student",
        "kind": "torch",
        "feature_cols": base.FILTERED_INPUTS,
        "pred_col": "soc_cnn_lstm_student_filtered",
        "factory": lambda: base.CNNLSTMRegressor(len(base.FILTERED_INPUTS), conv_channels=16, lstm_hidden=16, dense_hidden=8),
        "max_epochs": 100,
        "patience": 15,
        "learning_rate": 8e-4,
        "teacher": None,
    },
    {
        "method": "Filtered Distilled Tiny CNN-LSTM",
        "kind": "torch",
        "feature_cols": base.FILTERED_INPUTS,
        "pred_col": "soc_cnn_lstm_distilled_student_filtered",
        "factory": lambda: base.CNNLSTMRegressor(len(base.FILTERED_INPUTS), conv_channels=16, lstm_hidden=16, dense_hidden=8),
        "max_epochs": 100,
        "patience": 15,
        "learning_rate": 8e-4,
        "teacher": "Filtered CNN-LSTM Teacher",
    },
]


def serializable_spec(spec):
    return {
        key: value
        for key, value in spec.items()
        if key not in {"factory"}
    }


def to_markdown(df):
    display = df.copy().where(pd.notna(df), "")
    cols = list(display.columns)
    rows = display.astype(str).values.tolist()
    widths = [max(len(str(c)), *[len(r[i]) for r in rows]) for i, c in enumerate(cols)]

    def fmt(vals):
        return "| " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(vals)) + " |"

    return "\n".join([fmt(cols), "| " + " | ".join("-" * w for w in widths) + " |", *[fmt(r) for r in rows]]) + "\n"


def select_temperature_rows(manifest, temp_c):
    if temp_c == 25:
        file_name = manifest["file_name"].astype(str)
        temp_mask = file_name.str.contains("25degC", case=False, na=False)
        rows = manifest[temp_mask & manifest["test_type"].isin(["cycle", "drive_cycle"])].copy()
        if "temperature_profile" in rows.columns:
            rows = rows[rows["temperature_profile"].eq("constant")].copy()
        rows = rows[
            rows["file_name"].astype(str).str.contains(
                "Cycle_1|Cycle_2|Cycle_3|Cycle_4|US06|UDDS|LA92|NN",
                case=False,
                regex=True,
                na=False,
            )
        ].copy()
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
        return rows[rows["split"].eq("train")].reset_index(drop=True), rows[rows["split"].eq("test")].reset_index(drop=True)
    return base.select_temperature_rows(manifest, temp_c)


def load_temperature_frames(manifest, temp_c, out_dir):
    train_rows, test_rows = select_temperature_rows(manifest, temp_c)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_rows.to_csv(out_dir / f"selected_train_rows_{temp_c}degC.csv", index=False)
    test_rows.to_csv(out_dir / f"selected_test_rows_{temp_c}degC.csv", index=False)
    train_frames = [base.load_cycle(row, "train", use_filtered=True) for _, row in train_rows.iterrows()]
    test_frames = [base.load_cycle(row, "test", use_filtered=True) for _, row in test_rows.iterrows()]
    return train_rows, test_rows, train_frames, test_frames


def train_torch_strict(model, x_train, y_train, out_dir, device, spec, teacher=None):
    return base.train_torch(
        model,
        x_train,
        y_train,
        out_dir,
        device,
        max_epochs=spec["max_epochs"],
        patience=spec["patience"],
        batch_size=256,
        lr=spec["learning_rate"],
        weight_decay=1e-5,
        teacher=teacher,
        alpha=0.6,
    )


def evaluate_torch_on_frames(model, frames, mean, std, feature_cols, pred_col, method, out_dir, device, train_temp, test_temp, split_label):
    rows = []
    pred_frames = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for frame in frames:
        cycle = frame.copy().reset_index(drop=True)
        x, y, idxs = base.build_windows(cycle, mean, std, base.TEST_STRIDE, feature_cols)
        pred = base.predict_torch(model, x, device)
        cycle[pred_col] = np.nan
        cycle.loc[idxs, pred_col] = pred
        cycle["soc_error_percent"] = cycle[pred_col] - cycle["soc_ref_ah"]
        rows.append(
            {
                "train_temperature_C": train_temp,
                "test_temperature_C": test_temp,
                "split": split_label,
                "method": method,
                "file_name": cycle["file_name"].iloc[0],
                "cycle_name": cycle["cycle_name"].iloc[0],
                "sample_count": len(y),
                "parameter_count": base.count_params(model),
                **base.evaluate(y.reshape(-1), pred),
            }
        )
        cycle.to_csv(out_dir / f"{pred_col}_prediction_{base.safe_name(cycle['file_name'].iloc[0])}.csv", index=False)
        pred_frames.append(cycle)
    pd.concat(pred_frames, ignore_index=True).to_csv(out_dir / f"{pred_col}_{split_label}_predictions.csv", index=False)
    return rows


def evaluate_tabular_on_frames(model, frames, feature_cols, pred_col, method, out_dir, train_temp, test_temp, split_label):
    rows = []
    pred_frames = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for frame in frames:
        cycle = frame.copy().reset_index(drop=True)
        y = cycle["soc_ref_ah"].to_numpy(dtype=float)
        pred = np.clip(model.predict(cycle[feature_cols].to_numpy(dtype=float)), 0.0, 100.0)
        cycle[pred_col] = pred
        cycle["soc_error_percent"] = pred - y
        rows.append(
            {
                "train_temperature_C": train_temp,
                "test_temperature_C": test_temp,
                "split": split_label,
                "method": method,
                "file_name": cycle["file_name"].iloc[0],
                "cycle_name": cycle["cycle_name"].iloc[0],
                "sample_count": len(y),
                "parameter_count": np.nan,
                **base.evaluate(y, pred),
            }
        )
        cycle.to_csv(out_dir / f"{pred_col}_prediction_{base.safe_name(cycle['file_name'].iloc[0])}.csv", index=False)
        pred_frames.append(cycle)
    pd.concat(pred_frames, ignore_index=True).to_csv(out_dir / f"{pred_col}_{split_label}_predictions.csv", index=False)
    return rows


def add_averages(df):
    rows = []
    for keys, group in df[df["split"].eq("test")].groupby(["experiment", "train_temperature_C", "test_temperature_C", "method"], dropna=False):
        experiment, train_temp, test_temp, method = keys
        avg = {
            "experiment": experiment,
            "train_temperature_C": train_temp,
            "test_temperature_C": test_temp,
            "split": "test_average",
            "method": method,
            "file_name": "AVERAGE_TEST_CYCLES",
            "cycle_name": "test_average",
            "sample_count": int(group["sample_count"].sum()),
            "parameter_count": group["parameter_count"].dropna().iloc[0] if group["parameter_count"].notna().any() else np.nan,
        }
        for col in ["MAE_percent", "RMSE_percent", "MAX_ERROR_percent", "P95_ABS_ERROR_percent", "P99_ABS_ERROR_percent", "FINAL_ERROR_percent"]:
            avg[col] = float(group[col].mean())
        rows.append(avg)
    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True)


def run_training_suite(train_temp, train_frames, test_frames_by_temp, device, experiment_name, output_dir):
    all_rows = []
    trained = {}
    for spec in MODEL_SPECS:
        method_dir = output_dir / f"train_{train_temp}degC" / spec["method"].replace(" ", "_").lower()
        if spec["kind"] == "tabular":
            train_df = pd.concat(train_frames, ignore_index=True)
            model = base.make_mlp(spec["hidden_layers"], spec["max_iter"], spec["learning_rate"])
            print(f"Training {experiment_name} {spec['method']} at {train_temp}C: samples={len(train_df)}")
            model.fit(train_df[spec["feature_cols"]].to_numpy(dtype=float), train_df["soc_ref_ah"].to_numpy(dtype=float))
            trained[spec["method"]] = {"model": model, "spec": spec}
            for test_temp, test_frames in test_frames_by_temp.items():
                rows = evaluate_tabular_on_frames(
                    model,
                    test_frames,
                    spec["feature_cols"],
                    spec["pred_col"],
                    spec["method"],
                    method_dir / f"test_{test_temp}degC",
                    train_temp,
                    test_temp,
                    "test",
                )
                all_rows.extend(rows)
        else:
            train_df, mean, std = base.fit_scaler(train_frames, spec["feature_cols"])
            x_train, y_train = base.training_windows(train_frames, mean, std, spec["feature_cols"])
            model = spec["factory"]()
            teacher_pack = trained.get(spec["teacher"]) if spec.get("teacher") else None
            teacher_model = teacher_pack["model"] if teacher_pack else None
            print(
                f"Training {experiment_name} {spec['method']} at {train_temp}C: "
                f"windows={x_train.shape}, params={base.count_params(model)}"
            )
            model = train_torch_strict(model, x_train, y_train, method_dir, device, spec, teacher=teacher_model)
            trained[spec["method"]] = {"model": model, "mean": mean, "std": std, "spec": spec}
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "feature_mean": mean,
                    "feature_std": std,
                    "feature_columns": spec["feature_cols"],
                    "parameter_count": base.count_params(model),
                    "strict_matched_spec": serializable_spec(spec),
                },
                method_dir / f"{spec['method'].replace(' ', '_').lower()}_{train_temp}degC_model.pt",
            )
            for test_temp, test_frames in test_frames_by_temp.items():
                rows = evaluate_torch_on_frames(
                    model,
                    test_frames,
                    mean,
                    std,
                    spec["feature_cols"],
                    spec["pred_col"],
                    spec["method"],
                    method_dir / f"test_{test_temp}degC",
                    device,
                    train_temp,
                    test_temp,
                    "test",
                )
                all_rows.extend(rows)
    df = pd.DataFrame(all_rows)
    df["experiment"] = experiment_name
    return df


def write_summaries(df):
    df = add_averages(df)
    for col in ["MAE_percent", "RMSE_percent", "MAX_ERROR_percent", "P95_ABS_ERROR_percent", "P99_ABS_ERROR_percent", "FINAL_ERROR_percent"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").round(4)
    df.to_csv(OUTPUT_ROOT / "strict_matched_full_summary.csv", index=False)
    avg = df[df["split"].eq("test_average")].copy()
    avg.to_csv(OUTPUT_ROOT / "strict_matched_test_average.csv", index=False)
    (OUTPUT_ROOT / "strict_matched_test_average.md").write_text(to_markdown(avg), encoding="utf-8")
    profile = df[df["split"].eq("test")].copy()
    profile.to_csv(OUTPUT_ROOT / "strict_matched_profilewise_metrics.csv", index=False)
    (OUTPUT_ROOT / "strict_matched_profilewise_metrics.md").write_text(to_markdown(profile), encoding="utf-8")
    return df, avg


def plot_degradation(avg):
    plot_dir = OUTPUT_ROOT / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)
    transfer = avg[avg["experiment"].eq("transfer_from_25degC")].copy()
    transfer["pair"] = transfer["train_temperature_C"].astype(str) + "C->" + transfer["test_temperature_C"].astype(str) + "C"
    methods = [m["method"] for m in MODEL_SPECS]
    pairs = ["25C->25C", "25C->10C", "25C->0C"]
    pivot = transfer.pivot_table(index="method", columns="pair", values="RMSE_percent", aggfunc="mean").reindex(methods)
    pivot = pivot[[p for p in pairs if p in pivot.columns]]
    fig, ax = plt.subplots(figsize=(9.5, 6))
    im = ax.imshow(pivot.to_numpy(), cmap="YlOrRd", aspect="auto")
    ax.set_title("Strict matched direct temperature transfer RMSE")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    for y in range(pivot.shape[0]):
        for x in range(pivot.shape[1]):
            val = pivot.iloc[y, x]
            if np.isfinite(val):
                ax.text(x, y, f"{val:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="RMSE (%SOC)")
    fig.tight_layout()
    fig.savefig(plot_dir / "strict_matched_transfer_rmse_heatmap.png", dpi=220)
    plt.close(fig)

    degradation = transfer.pivot_table(index="method", columns="test_temperature_C", values="RMSE_percent", aggfunc="mean")
    if 25 in degradation.columns:
        for temp in [10, 0]:
            if temp in degradation.columns:
                degradation[f"degradation_x_25_to_{temp}"] = degradation[temp] / degradation[25]
                degradation[f"delta_RMSE_25_to_{temp}"] = degradation[temp] - degradation[25]
    degradation.reset_index().to_csv(OUTPUT_ROOT / "strict_matched_transfer_degradation_factors.csv", index=False)


def main():
    manifest = pd.read_csv(MANIFEST_PATH)
    device = base.get_device()
    print(f"Using device: {device}")
    rows_dir = OUTPUT_ROOT / "selected_rows"
    frames_by_temp = {}
    for temp in TEMPERATURES:
        train_rows, test_rows, train_frames, test_frames = load_temperature_frames(manifest, temp, rows_dir)
        frames_by_temp[temp] = {"train": train_frames, "test": test_frames}
        print(f"{temp}C selected train={len(train_rows)}, test={len(test_rows)}")
        print(train_rows[["file_name", "cycle_name"]].to_string(index=False))
        print(test_rows[["file_name", "cycle_name"]].to_string(index=False))

    all_parts = []
    for temp in TEMPERATURES:
        all_parts.append(
            run_training_suite(
                temp,
                frames_by_temp[temp]["train"],
                {temp: frames_by_temp[temp]["test"]},
                device,
                f"within_{temp}degC",
                OUTPUT_ROOT / "within_temperature",
            )
        )
    all_parts.append(
        run_training_suite(
            25,
            frames_by_temp[25]["train"],
            {temp: frames_by_temp[temp]["test"] for temp in TRANSFER_TEST_TEMPERATURES},
            device,
            "transfer_from_25degC",
            OUTPUT_ROOT / "temperature_transfer",
        )
    )
    all_df, avg = write_summaries(pd.concat(all_parts, ignore_index=True))
    plot_degradation(avg)
    (OUTPUT_ROOT / "README.md").write_text(
        "Strict matched temperature pipeline. All temperatures use the same split logic, downsampling, "
        "Ah-based SOC reference, feature definitions, model set, and per-model 25degC hyperparameters. "
        "Training profiles are Cycle_1-Cycle_4 and US06; test profiles are UDDS, LA92, and NN.\n",
        encoding="utf-8",
    )
    print("\nStrict matched test averages:")
    print(avg.to_string(index=False))
    print(f"\nSaved outputs to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
