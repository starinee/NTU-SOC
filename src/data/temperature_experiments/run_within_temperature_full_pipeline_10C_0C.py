from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = PROJECT_ROOT / "dataset/processed/panasonic_raw_csv/manifest.csv"
OUTPUT_ROOT = PROJECT_ROOT / "dataset/processed/temperature_experiments/within_temperature_full_pipeline_10C_0C"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

TEMPERATURES_TO_RUN = [10, 0]
CAPACITY_AH = 2.9
SOC0 = 100.0
DOWNSAMPLE_STEP = 10
SEQUENCE_LENGTH = 60
TRAIN_STRIDE = 5
TEST_STRIDE = 1
RANDOM_SEED = 42

TRAIN_KEYWORDS = ["Cycle_1", "Cycle_2", "Cycle_3", "Cycle_4", "US06"]
TEST_CYCLES = ["UDDS", "LA92", "NN"]

RAW_FEATURES = ["voltage_V", "current_A", "battery_temp_C"]
FILTERED_FEATURES = [
    "voltage_ema_5s",
    "current_ema_5s",
    "voltage_ema_30s",
    "current_ema_30s",
    "voltage_ema_120s",
    "current_ema_120s",
]
FILTERED_INPUTS = RAW_FEATURES + FILTERED_FEATURES
EMA_TIME_CONSTANTS_S = [5.0, 30.0, 120.0]

CC_SENSITIVITY_CASES = [
    ("ideal_capacity_2p9_soc0_100", 2.9, 100.0),
    ("low_capacity_2p8_soc0_100", 2.8, 100.0),
    ("high_capacity_3p0_soc0_100", 3.0, 100.0),
    ("capacity_2p9_soc0_95", 2.9, 95.0),
    ("capacity_2p9_soc0_90", 2.9, 90.0),
    ("low_capacity_2p8_soc0_95", 2.8, 95.0),
    ("high_capacity_3p0_soc0_95", 3.0, 95.0),
    ("low_capacity_2p8_soc0_90", 2.8, 90.0),
    ("high_capacity_3p0_soc0_90", 3.0, 90.0),
]


def safe_name(text):
    return Path(str(text)).stem.replace(" ", "_").replace(".", "p")


def contains_any(text, keywords):
    text = str(text).lower()
    return any(k.lower() in text for k in keywords)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def soc_from_ah(df):
    return np.clip(SOC0 + df["ah"].to_numpy(dtype=float) / CAPACITY_AH * 100.0, 0.0, 100.0)


def coulomb_counting(df, capacity_ah=CAPACITY_AH, soc0=SOC0):
    current_a = -df["current_A"].to_numpy(dtype=float)
    time_s = df["time_s"].to_numpy(dtype=float)
    dt = np.diff(time_s, prepend=time_s[0])
    dt[0] = 0.0
    delta_soc = current_a * dt / (capacity_ah * 3600.0) * 100.0
    return np.clip(soc0 - np.cumsum(delta_soc), 0.0, 100.0)


def evaluate(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    abs_err = np.abs(err)
    return {
        "MAE_percent": float(mean_absolute_error(y_true, y_pred)),
        "RMSE_percent": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAX_ERROR_percent": float(np.max(abs_err)),
        "P95_ABS_ERROR_percent": float(np.quantile(abs_err, 0.95)),
        "P99_ABS_ERROR_percent": float(np.quantile(abs_err, 0.99)),
        "FINAL_ERROR_percent": float(err[-1]),
    }


def add_filtered_features(df):
    df = df.copy()
    median_dt = float(df["time_s"].diff().dropna().median())
    if not np.isfinite(median_dt) or median_dt <= 0:
        median_dt = 1.0
    for tau in EMA_TIME_CONSTANTS_S:
        alpha = 1.0 - np.exp(-median_dt / tau)
        label = str(int(tau))
        df[f"voltage_ema_{label}s"] = df["voltage_V"].ewm(alpha=alpha, adjust=False).mean()
        df[f"current_ema_{label}s"] = df["current_A"].ewm(alpha=alpha, adjust=False).mean()
    return df


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
        temp_mask = manifest["ambient_temp_C"].eq(float(temp_c))

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
        if contains_any(r["file_name"], TRAIN_KEYWORDS)
        else ("test" if str(r.get("cycle_name", "")) in TEST_CYCLES else "unused"),
        axis=1,
    )
    rows = rows[rows["split"].isin(["train", "test"])].sort_values(["split", "file_name"])
    train_rows = rows[rows["split"].eq("train")].reset_index(drop=True)
    test_rows = rows[rows["split"].eq("test")].reset_index(drop=True)
    if len(train_rows) < 5 or len(test_rows) < 3:
        raise ValueError(f"{temp_c}C selected train={len(train_rows)}, test={len(test_rows)}")
    return train_rows, test_rows


def load_cycle(row, split, use_filtered=True):
    df = pd.read_csv(row["output_csv"]).sort_values("time_s").reset_index(drop=True)
    required = ["time_s", "voltage_V", "current_A", "battery_temp_C", "ah"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{row['file_name']} missing {missing}")
    df = df.dropna(subset=required).reset_index(drop=True)
    df = df.iloc[::DOWNSAMPLE_STEP].reset_index(drop=True)
    if use_filtered:
        df = add_filtered_features(df)
    df["soc_ref_ah"] = soc_from_ah(df)
    df["split"] = split
    df["cycle_name"] = row.get("cycle_name", "")
    df["file_name"] = row["file_name"]
    df["ambient_temp_C"] = row.get("ambient_temp_C", np.nan)
    return df


def make_mlp(hidden_layers, max_iter, learning_rate):
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=hidden_layers,
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    batch_size=512,
                    learning_rate_init=learning_rate,
                    max_iter=max_iter,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=20,
                    random_state=RANDOM_SEED,
                    verbose=False,
                ),
            ),
        ]
    )


def run_tabular_model(temp_dir, method_name, train_frames, test_frames, feature_cols, pred_col, hidden_layers, max_iter, learning_rate):
    out_dir = temp_dir / method_name.replace(" ", "_").lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df = pd.concat(train_frames, ignore_index=True)
    x_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df["soc_ref_ah"].to_numpy(dtype=float)
    model = make_mlp(hidden_layers, max_iter, learning_rate)
    print(f"Training {method_name}: samples={len(train_df)}, features={len(feature_cols)}")
    model.fit(x_train, y_train)

    records = []
    train_pred = np.clip(model.predict(x_train), 0.0, 100.0)
    records.append({"split": "train", "method": method_name, "file_name": "ALL_TRAIN", "cycle_name": "train", "sample_count": len(train_df), **evaluate(y_train, train_pred)})
    pred_frames = []
    for cycle_df in test_frames:
        cycle_df = cycle_df.copy().reset_index(drop=True)
        x = cycle_df[feature_cols].to_numpy(dtype=float)
        y = cycle_df["soc_ref_ah"].to_numpy(dtype=float)
        pred = np.clip(model.predict(x), 0.0, 100.0)
        cycle_df[pred_col] = pred
        cycle_df["soc_error_percent"] = pred - cycle_df["soc_ref_ah"]
        records.append({"split": "test", "method": method_name, "file_name": cycle_df["file_name"].iloc[0], "cycle_name": cycle_df["cycle_name"].iloc[0], "sample_count": len(cycle_df), **evaluate(y, pred)})
        cycle_df.to_csv(out_dir / f"{pred_col}_prediction_{safe_name(cycle_df['file_name'].iloc[0])}.csv", index=False)
        pred_frames.append(cycle_df)
    summary = add_average(pd.DataFrame(records), method_name)
    summary.to_csv(out_dir / f"{method_name.replace(' ', '_').lower()}_summary.csv", index=False)
    pd.concat(pred_frames, ignore_index=True).to_csv(out_dir / f"{pred_col}_test_predictions.csv", index=False)
    return summary


def build_windows(cycle_df, feature_mean, feature_std, stride, feature_cols):
    x_raw = cycle_df[feature_cols].to_numpy(dtype=np.float32)
    x_raw = (x_raw - feature_mean) / feature_std
    y_raw = cycle_df["soc_ref_ah"].to_numpy(dtype=np.float32)
    xs, ys, idxs = [], [], []
    for end in range(SEQUENCE_LENGTH - 1, len(cycle_df), stride):
        start = end - SEQUENCE_LENGTH + 1
        xs.append(x_raw[start : end + 1])
        ys.append(y_raw[end])
        idxs.append(end)
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32).reshape(-1, 1), np.asarray(idxs, dtype=int)


def fit_scaler(train_frames, feature_cols):
    train_df = pd.concat(train_frames, ignore_index=True)
    x = train_df[feature_cols].to_numpy(dtype=np.float32)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return train_df, mean, std


def training_windows(train_frames, mean, std, feature_cols):
    xs, ys = [], []
    for frame in train_frames:
        x, y, _ = build_windows(frame, mean, std, TRAIN_STRIDE, feature_cols)
        xs.append(x)
        ys.append(y)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class CNNLSTMRegressor(nn.Module):
    def __init__(self, input_size, conv_channels=64, lstm_hidden=64, dense_hidden=32, dropout=0.05):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, conv_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(input_size=conv_channels, hidden_size=lstm_hidden, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(lstm_hidden, dense_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dense_hidden, 1))

    def forward(self, x):
        z = self.conv(x.transpose(1, 2)).transpose(1, 2)
        out, _ = self.lstm(z)
        return self.head(out[:, -1, :])


def count_params(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def train_torch(model, x_train, y_train, out_dir, device, max_epochs=60, patience=10, batch_size=256, lr=8e-4, weight_decay=1e-5, teacher=None, alpha=0.6):
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    out_dir.mkdir(parents=True, exist_ok=True)

    teacher_targets = None
    if teacher is not None:
        teacher = teacher.to(device).eval()
        preds = []
        for (bx,) in DataLoader(TensorDataset(torch.from_numpy(x_train)), batch_size=batch_size):
            with torch.no_grad():
                preds.append(teacher(bx.to(device)).cpu().numpy())
        teacher_targets = np.concatenate(preds, axis=0).astype(np.float32)

    n = len(x_train)
    rng = np.random.default_rng(RANDOM_SEED)
    indices = rng.permutation(n)
    val_count = max(1, int(n * 0.15))
    val_idx = indices[:val_count]
    train_idx = indices[val_count:]

    if teacher_targets is None:
        dataset = TensorDataset(torch.from_numpy(x_train[train_idx]), torch.from_numpy(y_train[train_idx]))
    else:
        dataset = TensorDataset(
            torch.from_numpy(x_train[train_idx]),
            torch.from_numpy(y_train[train_idx]),
            torch.from_numpy(teacher_targets[train_idx]),
        )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_x = torch.from_numpy(x_train[val_idx]).to(device)
    val_y = torch.from_numpy(y_train[val_idx]).to(device)
    val_teacher = torch.from_numpy(teacher_targets[val_idx]).to(device) if teacher_targets is not None else None

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    best_loss = float("inf")
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for batch in loader:
            opt.zero_grad()
            bx = batch[0].to(device)
            by = batch[1].to(device)
            pred = model(bx)
            supervised = loss_fn(pred, by)
            if teacher_targets is None:
                loss = supervised
            else:
                bt = batch[2].to(device)
                loss = alpha * supervised + (1.0 - alpha) * loss_fn(pred, bt)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            val_pred = model(val_x)
            val_loss = loss_fn(val_pred, val_y)
            if val_teacher is not None:
                val_loss = alpha * val_loss + (1.0 - alpha) * loss_fn(val_pred, val_teacher)
            val_loss_value = float(val_loss.item())
        rmse = float(np.sqrt(np.mean(losses)))
        val_rmse = float(np.sqrt(val_loss_value))
        history.append({"epoch": epoch, "train_RMSE": rmse, "val_RMSE": val_rmse})
        print(f"Epoch {epoch:03d}: train_RMSE={rmse:.4f}, val_RMSE={val_rmse:.4f}")
        if val_loss_value < best_loss - 1e-4:
            best_loss = val_loss_value
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            print("Early stopping.")
            break
    if best_state:
        model.load_state_dict(best_state)
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    return model


def predict_torch(model, x, device, batch_size=256):
    model = model.to(device).eval()
    preds = []
    for (bx,) in DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size):
        with torch.no_grad():
            preds.append(model(bx.to(device)).cpu().numpy().reshape(-1))
    return np.clip(np.concatenate(preds), 0.0, 100.0)


def run_torch_model(temp_dir, method_name, train_frames, test_frames, feature_cols, pred_col, model_factory, device, teacher=None, max_epochs=60):
    out_dir = temp_dir / method_name.replace(" ", "_").lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df, mean, std = fit_scaler(train_frames, feature_cols)
    x_train, y_train = training_windows(train_frames, mean, std, feature_cols)
    model = model_factory()
    print(f"Training {method_name}: windows={x_train.shape}, params={count_params(model)}")
    model = train_torch(model, x_train, y_train, out_dir, device, max_epochs=max_epochs, patience=10, teacher=teacher)

    records = []
    train_pred = predict_torch(model, x_train, device)
    records.append({"split": "train", "method": method_name, "file_name": "ALL_TRAIN", "cycle_name": "train", "sample_count": len(y_train), "parameter_count": count_params(model), **evaluate(y_train.reshape(-1), train_pred)})
    pred_frames = []
    for frame in test_frames:
        cycle_df = frame.copy().reset_index(drop=True)
        x, y, idxs = build_windows(cycle_df, mean, std, TEST_STRIDE, feature_cols)
        pred = predict_torch(model, x, device)
        cycle_df[pred_col] = np.nan
        cycle_df.loc[idxs, pred_col] = pred
        cycle_df["soc_error_percent"] = cycle_df[pred_col] - cycle_df["soc_ref_ah"]
        records.append({"split": "test", "method": method_name, "file_name": cycle_df["file_name"].iloc[0], "cycle_name": cycle_df["cycle_name"].iloc[0], "sample_count": len(y), "parameter_count": count_params(model), **evaluate(y.reshape(-1), pred)})
        cycle_df.to_csv(out_dir / f"{pred_col}_prediction_{safe_name(cycle_df['file_name'].iloc[0])}.csv", index=False)
        pred_frames.append(cycle_df)
    summary = add_average(pd.DataFrame(records), method_name)
    summary.to_csv(out_dir / f"{method_name.replace(' ', '_').lower()}_summary.csv", index=False)
    pd.concat(pred_frames, ignore_index=True).to_csv(out_dir / f"{pred_col}_test_predictions.csv", index=False)
    torch.save(
        {"model_state_dict": model.state_dict(), "feature_mean": mean, "feature_std": std, "feature_columns": feature_cols, "parameter_count": count_params(model)},
        out_dir / f"{method_name.replace(' ', '_').lower()}_model.pt",
    )
    return summary, model


def add_average(df, method_name):
    tests = df[df["split"].eq("test")]
    avg = {"split": "test_average", "method": method_name, "file_name": "AVERAGE_TEST_CYCLES", "cycle_name": "test_average", "sample_count": int(tests["sample_count"].sum())}
    if "parameter_count" in df.columns:
        avg["parameter_count"] = tests["parameter_count"].dropna().iloc[0] if tests["parameter_count"].notna().any() else np.nan
    for col in ["MAE_percent", "RMSE_percent", "MAX_ERROR_percent", "P95_ABS_ERROR_percent", "P99_ABS_ERROR_percent", "FINAL_ERROR_percent"]:
        avg[col] = float(tests[col].mean())
    return pd.concat([df, pd.DataFrame([avg])], ignore_index=True)


def run_coulomb_methods(temp_dir, test_frames):
    out_dir = temp_dir / "traditional_cc"
    out_dir.mkdir(parents=True, exist_ok=True)
    cc_records = []
    sensitivity_records = []
    for frame in test_frames:
        df = frame.copy().reset_index(drop=True)
        y = df["soc_ref_ah"].to_numpy(dtype=float)
        pred = coulomb_counting(df)
        df["soc_cc"] = pred
        df["soc_error_percent"] = pred - y
        cc_records.append({"split": "test", "method": "Coulomb Counting", "file_name": df["file_name"].iloc[0], "cycle_name": df["cycle_name"].iloc[0], "sample_count": len(df), **evaluate(y, pred)})
        for case_name, cap, soc0 in CC_SENSITIVITY_CASES:
            case_pred = coulomb_counting(df, capacity_ah=cap, soc0=soc0)
            sensitivity_records.append({"split": "test", "method": "CC sensitivity", "case_name": case_name, "file_name": df["file_name"].iloc[0], "cycle_name": df["cycle_name"].iloc[0], "sample_count": len(df), **evaluate(y, case_pred)})
        df.to_csv(out_dir / f"cc_prediction_{safe_name(df['file_name'].iloc[0])}.csv", index=False)
    cc_summary = add_average(pd.DataFrame(cc_records), "Coulomb Counting")
    sens_summary = pd.DataFrame(sensitivity_records)
    sens_avg = (
        sens_summary.groupby("case_name", as_index=False)[["MAE_percent", "RMSE_percent", "MAX_ERROR_percent", "P95_ABS_ERROR_percent", "P99_ABS_ERROR_percent", "FINAL_ERROR_percent"]]
        .mean()
        .assign(split="test_average", method="CC sensitivity", file_name="AVERAGE_TEST_CYCLES", cycle_name="test_average", sample_count=int(sum(len(f) for f in test_frames)))
    )
    sens_summary = pd.concat([sens_summary, sens_avg], ignore_index=True)
    cc_summary.to_csv(out_dir / "coulomb_counting_summary.csv", index=False)
    sens_summary.to_csv(out_dir / "cc_sensitivity_summary.csv", index=False)
    return cc_summary, sens_summary


def to_markdown(df):
    display = df.copy().where(pd.notna(df), "")
    cols = list(display.columns)
    rows = display.astype(str).values.tolist()
    widths = [max(len(str(c)), *[len(r[i]) for r in rows]) for i, c in enumerate(cols)]
    def fmt(vals):
        return "| " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(vals)) + " |"
    return "\n".join([fmt(cols), "| " + " | ".join("-" * w for w in widths) + " |", *[fmt(r) for r in rows]]) + "\n"


def run_temperature(temp_c, manifest, device):
    temp_dir = OUTPUT_ROOT / f"{temp_c}degC"
    temp_dir.mkdir(parents=True, exist_ok=True)
    train_rows, test_rows = select_temperature_rows(manifest, temp_c)
    train_rows.to_csv(temp_dir / "selected_train_rows.csv", index=False)
    test_rows.to_csv(temp_dir / "selected_test_rows.csv", index=False)
    print(f"\n===== {temp_c}C selected files =====")
    print("Train:")
    print(train_rows[["file_name", "cycle_name", "output_csv"]].to_string(index=False))
    print("Test:")
    print(test_rows[["file_name", "cycle_name", "output_csv"]].to_string(index=False))

    train_frames = [load_cycle(row, "train", use_filtered=True) for _, row in train_rows.iterrows()]
    test_frames = [load_cycle(row, "test", use_filtered=True) for _, row in test_rows.iterrows()]

    summaries = []
    cc_summary, sens_summary = run_coulomb_methods(temp_dir, test_frames)
    summaries.append(cc_summary)

    summaries.append(run_tabular_model(temp_dir, "Instantaneous MLP", train_frames, test_frames, RAW_FEATURES, "soc_mlp", (64, 32), 300, 1e-3))
    summaries.append(run_tabular_model(temp_dir, "Filtered-feature MLP", train_frames, test_frames, FILTERED_INPUTS, "soc_mlp_filtered", (128, 64, 32), 500, 8e-4))
    summaries.append(run_torch_model(temp_dir, "LSTM", train_frames, test_frames, RAW_FEATURES, "soc_lstm", lambda: LSTMRegressor(len(RAW_FEATURES), hidden=32), device, max_epochs=60)[0])

    teacher_summary, teacher_model = run_torch_model(
        temp_dir,
        "Filtered CNN-LSTM Teacher",
        train_frames,
        test_frames,
        FILTERED_INPUTS,
        "soc_cnn_lstm_teacher_filtered",
        lambda: CNNLSTMRegressor(len(FILTERED_INPUTS), conv_channels=64, lstm_hidden=64, dense_hidden=32),
        device,
        max_epochs=60,
    )
    summaries.append(teacher_summary)
    summaries.append(
        run_torch_model(
            temp_dir,
            "Filtered Tiny CNN-LSTM Student",
            train_frames,
            test_frames,
            FILTERED_INPUTS,
            "soc_cnn_lstm_student_filtered",
            lambda: CNNLSTMRegressor(len(FILTERED_INPUTS), conv_channels=16, lstm_hidden=16, dense_hidden=8),
            device,
            max_epochs=60,
        )[0]
    )
    summaries.append(
        run_torch_model(
            temp_dir,
            "Filtered Distilled Tiny CNN-LSTM",
            train_frames,
            test_frames,
            FILTERED_INPUTS,
            "soc_cnn_lstm_distilled_student_filtered",
            lambda: CNNLSTMRegressor(len(FILTERED_INPUTS), conv_channels=16, lstm_hidden=16, dense_hidden=8),
            device,
            teacher=teacher_model,
            max_epochs=60,
        )[0]
    )

    combined = pd.concat(summaries, ignore_index=True)
    for col in ["MAE_percent", "RMSE_percent", "MAX_ERROR_percent", "P95_ABS_ERROR_percent", "P99_ABS_ERROR_percent", "FINAL_ERROR_percent"]:
        combined[col] = pd.to_numeric(combined[col], errors="coerce").round(4)
    combined.to_csv(temp_dir / "combined_method_summary.csv", index=False)
    combined[combined["split"].eq("test_average")].to_csv(temp_dir / "combined_method_test_average.csv", index=False)
    (temp_dir / "combined_method_test_average.md").write_text(to_markdown(combined[combined["split"].eq("test_average")]), encoding="utf-8")
    sens_summary.to_csv(temp_dir / "cc_sensitivity_full_summary.csv", index=False)
    return combined.assign(temperature_C=temp_c)


def main():
    manifest = pd.read_csv(MANIFEST_PATH)
    device = get_device()
    print(f"Using device: {device}")
    all_summaries = []
    for temp in TEMPERATURES_TO_RUN:
        all_summaries.append(run_temperature(temp, manifest, device))
    all_df = pd.concat(all_summaries, ignore_index=True)
    all_df.to_csv(OUTPUT_ROOT / "all_temperatures_combined_summary.csv", index=False)
    avg = all_df[all_df["split"].eq("test_average")].copy()
    avg = avg[["temperature_C", "method", "sample_count", "MAE_percent", "RMSE_percent", "MAX_ERROR_percent", "P95_ABS_ERROR_percent", "P99_ABS_ERROR_percent", "FINAL_ERROR_percent"]]
    avg.to_csv(OUTPUT_ROOT / "all_temperatures_test_average.csv", index=False)
    (OUTPUT_ROOT / "all_temperatures_test_average.md").write_text(to_markdown(avg), encoding="utf-8")
    (OUTPUT_ROOT / "README.md").write_text(
        "Full 10degC and 0degC reruns of the 25degC SOC-estimation pipeline. "
        "Each temperature uses Cycle_1-Cycle_4 and US06 for training, and UDDS/LA92/NN for testing. "
        "OCV lookup is not rerun because the converted data only expose a clear C20 OCV source for 25degC, not for the selected 10degC/0degC reruns.\n",
        encoding="utf-8",
    )
    print("\nAll temperature test averages:")
    print(avg.to_string(index=False))
    print(f"\nSaved all outputs to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
