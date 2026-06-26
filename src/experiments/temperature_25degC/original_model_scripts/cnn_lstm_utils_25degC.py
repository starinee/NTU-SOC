from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[4]
MANIFEST_PATH = PROJECT_ROOT / "dataset/processed/panasonic_raw_csv/manifest.csv"

CAPACITY_AH = 2.9
SOC0 = 100.0
DOWNSAMPLE_STEP = 10
SEQUENCE_LENGTH = 60
TRAIN_STRIDE = 5
TEST_STRIDE = 1
RANDOM_SEED = 42

FEATURE_COLUMNS = [
    "voltage_V",
    "current_A",
    "battery_temp_C",
]

EMA_TIME_CONSTANTS_S = [5.0, 30.0, 120.0]

FILTERED_FEATURE_COLUMNS = [
    "voltage_ema_5s",
    "current_ema_5s",
    "voltage_ema_30s",
    "current_ema_30s",
    "voltage_ema_120s",
    "current_ema_120s",
]

FEATURE_COLUMNS_FILTERED = FEATURE_COLUMNS + FILTERED_FEATURE_COLUMNS

TRAIN_CYCLE_KEYWORDS = [
    "Cycle_1",
    "Cycle_2",
    "Cycle_3",
    "Cycle_4",
    "US06",
]

TEST_CYCLE_KEYWORDS = [
    "UDDS",
    "LA92",
    "NN",
]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_safe_name(file_name: str) -> str:
    return Path(file_name).stem.replace(" ", "_").replace(".", "p")


def soc_from_ah(df, ah_col="ah", capacity_ah=CAPACITY_AH, soc0=SOC0):
    ah = df[ah_col].to_numpy(dtype=float)
    soc_ref = soc0 + ah / capacity_ah * 100.0
    return np.clip(soc_ref, 0.0, 100.0)


def evaluate_soc(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_pred - y_true
    abs_error = np.abs(error)
    return {
        "MAE_percent": float(mean_absolute_error(y_true, y_pred)),
        "RMSE_percent": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAX_ERROR_percent": float(np.max(abs_error)),
        "P95_ABS_ERROR_percent": float(np.quantile(abs_error, 0.95)),
        "P99_ABS_ERROR_percent": float(np.quantile(abs_error, 0.99)),
        "FINAL_ERROR_percent": float(error[-1]),
    }


def contains_any(text, keywords):
    text_lower = str(text).lower()
    return any(k.lower() in text_lower for k in keywords)


def select_25degC_drive_files(manifest):
    drive_25 = manifest[
        manifest["file_name"].astype(str).str.contains("25degC", case=False, na=False)
        & manifest["file_name"].astype(str).str.contains(
            "Cycle_1|Cycle_2|Cycle_3|Cycle_4|US06|UDDS|LA92|NN",
            case=False,
            na=False,
            regex=True,
        )
    ].copy()
    drive_25 = drive_25[
        ~drive_25["file_name"].astype(str).str.contains("HWFT|HWFET", case=False, na=False)
    ].copy()
    if len(drive_25) == 0:
        raise ValueError("No 25degC drive-cycle files found.")
    return drive_25.sort_values("file_name").reset_index(drop=True)


def assign_split(row):
    file_name = row["file_name"]
    if contains_any(file_name, TRAIN_CYCLE_KEYWORDS):
        return "train"
    if contains_any(file_name, TEST_CYCLE_KEYWORDS):
        return "test"
    return "unused"


def add_filtered_features(df):
    df = df.copy()
    if len(df) < 2:
        raise ValueError("Not enough samples to build filtered features.")

    median_dt = float(df["time_s"].diff().dropna().median())
    if not np.isfinite(median_dt) or median_dt <= 0:
        median_dt = 1.0

    for tau_s in EMA_TIME_CONSTANTS_S:
        alpha = 1.0 - np.exp(-median_dt / tau_s)
        label = str(int(tau_s))
        df[f"voltage_ema_{label}s"] = df["voltage_V"].ewm(alpha=alpha, adjust=False).mean()
        df[f"current_ema_{label}s"] = df["current_A"].ewm(alpha=alpha, adjust=False).mean()

    return df


def load_one_cycle(row, split, use_filtered_features=False, feature_columns=None):
    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS_FILTERED if use_filtered_features else FEATURE_COLUMNS

    file_path = row["output_csv"]
    file_name = row["file_name"]
    cycle_name = row.get("cycle_name", "")

    df = pd.read_csv(file_path)
    df = df.sort_values("time_s").reset_index(drop=True)

    required_cols = ["time_s", "voltage_V", "current_A", "battery_temp_C", "ah"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{file_name}: missing columns {missing}")

    df = df.dropna(subset=required_cols).reset_index(drop=True)
    if DOWNSAMPLE_STEP > 1:
        df = df.iloc[::DOWNSAMPLE_STEP].reset_index(drop=True)

    if use_filtered_features:
        df = add_filtered_features(df)

    df = df.dropna(subset=feature_columns).reset_index(drop=True)
    df["soc_ref_ah"] = soc_from_ah(df)
    df["split"] = split
    df["cycle_name"] = cycle_name
    df["file_name"] = file_name
    return df


def load_split_data(use_filtered_features=False, feature_columns=None):
    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS_FILTERED if use_filtered_features else FEATURE_COLUMNS

    manifest = pd.read_csv(MANIFEST_PATH)
    drive_25 = select_25degC_drive_files(manifest)
    drive_25["split"] = drive_25.apply(assign_split, axis=1)

    print("\nSelected 25degC drive-cycle files:")
    print(drive_25[["file_name", "cycle_name", "split", "output_csv"]].to_string(index=False))

    train_rows = drive_25[drive_25["split"] == "train"]
    test_rows = drive_25[drive_25["split"] == "test"]

    train_frames = [
        load_one_cycle(row, "train", use_filtered_features, feature_columns)
        for _, row in train_rows.iterrows()
    ]
    test_frames = [
        load_one_cycle(row, "test", use_filtered_features, feature_columns)
        for _, row in test_rows.iterrows()
    ]
    return drive_25, train_frames, test_frames


def fit_feature_scaler(train_frames, feature_columns=None):
    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS

    train_df = pd.concat(train_frames, ignore_index=True)
    feature_mean = train_df[feature_columns].to_numpy(dtype=np.float32).mean(axis=0)
    feature_std = train_df[feature_columns].to_numpy(dtype=np.float32).std(axis=0)
    feature_std = np.where(feature_std == 0, 1.0, feature_std)
    return train_df, feature_mean, feature_std


def build_windows_from_cycle(df, feature_mean, feature_std, stride, feature_columns=None):
    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS

    features = df[feature_columns].to_numpy(dtype=np.float32)
    features = (features - feature_mean) / feature_std
    targets = df["soc_ref_ah"].to_numpy(dtype=np.float32)

    x_windows = []
    y_values = []
    target_indices = []

    for end_idx in range(SEQUENCE_LENGTH - 1, len(df), stride):
        start_idx = end_idx - SEQUENCE_LENGTH + 1
        x_windows.append(features[start_idx : end_idx + 1])
        y_values.append(targets[end_idx])
        target_indices.append(end_idx)

    x = np.asarray(x_windows, dtype=np.float32)
    y = np.asarray(y_values, dtype=np.float32).reshape(-1, 1)
    return x, y, np.asarray(target_indices, dtype=int)


def build_training_windows(train_frames, feature_mean, feature_std, feature_columns=None):
    train_windows = []
    train_targets = []
    for cycle_df in train_frames:
        x, y, _ = build_windows_from_cycle(
            cycle_df,
            feature_mean=feature_mean,
            feature_std=feature_std,
            stride=TRAIN_STRIDE,
            feature_columns=feature_columns,
        )
        train_windows.append(x)
        train_targets.append(y)
    return np.concatenate(train_windows, axis=0), np.concatenate(train_targets, axis=0)


class CNNLSTMRegressor(nn.Module):
    def __init__(
        self,
        input_size,
        conv_channels=32,
        lstm_hidden=64,
        dense_hidden=32,
        dropout=0.05,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, conv_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, dense_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_hidden, 1),
        )

    def forward(self, x):
        # x: [batch, seq, features]
        z = x.transpose(1, 2)
        z = self.conv(z)
        z = z.transpose(1, 2)
        output, _ = self.lstm(z)
        return self.head(output[:, -1, :])


def make_model(config, input_size=None):
    if input_size is None:
        input_size = len(config.get("feature_columns", FEATURE_COLUMNS))
    return CNNLSTMRegressor(
        input_size=input_size,
        conv_channels=config["conv_channels"],
        lstm_hidden=config["lstm_hidden"],
        dense_hidden=config["dense_hidden"],
        dropout=config.get("dropout", 0.05),
    )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_supervised(
    model,
    x_train,
    y_train,
    output_dir,
    device,
    max_epochs=80,
    patience=12,
    batch_size=256,
    learning_rate=1e-3,
    weight_decay=1e-5,
):
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    n = len(x_train)
    rng = np.random.default_rng(RANDOM_SEED)
    indices = rng.permutation(n)
    val_count = max(1, int(n * 0.15))
    val_idx = indices[:val_count]
    train_idx = indices[val_count:]

    train_dataset = TensorDataset(
        torch.from_numpy(x_train[train_idx]),
        torch.from_numpy(y_train[train_idx]),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_x = torch.from_numpy(x_train[val_idx]).to(device)
    val_y = torch.from_numpy(y_train[val_idx]).to(device)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_pred = model(val_x)
            val_loss = loss_fn(val_pred, val_y).item()

        train_rmse = float(np.sqrt(np.mean(train_losses)))
        val_rmse = float(np.sqrt(val_loss))
        history.append({"epoch": epoch, "train_RMSE": train_rmse, "val_RMSE": val_rmse})
        print(f"Epoch {epoch:03d}: train_RMSE={train_rmse:.4f}, val_RMSE={val_rmse:.4f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    return model, history


def train_distilled(
    student_model,
    teacher_model,
    x_train,
    y_train,
    output_dir,
    device,
    alpha=0.6,
    max_epochs=80,
    patience=12,
    batch_size=256,
    learning_rate=1e-3,
    weight_decay=1e-5,
):
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    teacher_model = teacher_model.to(device)
    teacher_model.eval()
    teacher_targets = []
    teacher_loader = DataLoader(torch.from_numpy(x_train), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch_x in teacher_loader:
            teacher_pred = teacher_model(batch_x.to(device)).cpu().numpy()
            teacher_targets.append(teacher_pred)
    teacher_targets = np.concatenate(teacher_targets, axis=0).astype(np.float32)

    n = len(x_train)
    rng = np.random.default_rng(RANDOM_SEED)
    indices = rng.permutation(n)
    val_count = max(1, int(n * 0.15))
    val_idx = indices[:val_count]
    train_idx = indices[val_count:]

    train_dataset = TensorDataset(
        torch.from_numpy(x_train[train_idx]),
        torch.from_numpy(y_train[train_idx]),
        torch.from_numpy(teacher_targets[train_idx]),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_x = torch.from_numpy(x_train[val_idx]).to(device)
    val_y = torch.from_numpy(y_train[val_idx]).to(device)
    val_teacher = torch.from_numpy(teacher_targets[val_idx]).to(device)

    student_model = student_model.to(device)
    optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        student_model.train()
        supervised_losses = []
        distill_losses = []
        total_losses = []

        for batch_x, batch_y, batch_teacher in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_teacher = batch_teacher.to(device)

            optimizer.zero_grad()
            pred = student_model(batch_x)
            supervised_loss = loss_fn(pred, batch_y)
            distill_loss = loss_fn(pred, batch_teacher)
            loss = alpha * supervised_loss + (1.0 - alpha) * distill_loss
            loss.backward()
            nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
            optimizer.step()

            supervised_losses.append(supervised_loss.item())
            distill_losses.append(distill_loss.item())
            total_losses.append(loss.item())

        student_model.eval()
        with torch.no_grad():
            val_pred = student_model(val_x)
            val_supervised_loss = loss_fn(val_pred, val_y).item()
            val_distill_loss = loss_fn(val_pred, val_teacher).item()
            val_total_loss = alpha * val_supervised_loss + (1.0 - alpha) * val_distill_loss

        history.append(
            {
                "epoch": epoch,
                "train_total_RMSE": float(np.sqrt(np.mean(total_losses))),
                "train_supervised_RMSE": float(np.sqrt(np.mean(supervised_losses))),
                "train_distill_RMSE": float(np.sqrt(np.mean(distill_losses))),
                "val_total_RMSE": float(np.sqrt(val_total_loss)),
                "val_supervised_RMSE": float(np.sqrt(val_supervised_loss)),
                "val_distill_RMSE": float(np.sqrt(val_distill_loss)),
                "alpha": alpha,
            }
        )
        print(
            f"Epoch {epoch:03d}: "
            f"train_total_RMSE={history[-1]['train_total_RMSE']:.4f}, "
            f"val_supervised_RMSE={history[-1]['val_supervised_RMSE']:.4f}, "
            f"val_distill_RMSE={history[-1]['val_distill_RMSE']:.4f}"
        )

        if val_supervised_loss < best_val_loss - 1e-4:
            best_val_loss = val_supervised_loss
            best_state = {k: v.detach().cpu().clone() for k, v in student_model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping.")
            break

    if best_state is not None:
        student_model.load_state_dict(best_state)

    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    return student_model, history


def predict_windows(model, x, device, batch_size=256):
    model.eval()
    preds = []
    loader = DataLoader(torch.from_numpy(x), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch_x in loader:
            pred = model(batch_x.to(device)).cpu().numpy().reshape(-1)
            preds.append(pred)
    y_pred = np.concatenate(preds)
    return np.clip(y_pred, 0.0, 100.0)


def evaluate_model_on_splits(
    model,
    train_frames,
    test_frames,
    x_train,
    y_train,
    feature_mean,
    feature_std,
    output_dir,
    device,
    prediction_col,
    method_name,
    feature_columns=None,
):
    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS

    records = []
    prediction_frames = []

    train_pred = predict_windows(model, x_train, device)
    train_metrics = evaluate_soc(y_train.reshape(-1), train_pred)
    records.append(
        {
            "split": "train",
            "file_name": "ALL_TRAIN_FILES",
            "cycle_name": "train_combined",
            "sample_count": len(y_train),
            **train_metrics,
        }
    )

    print("\nTraining-window metrics:")
    for k, v in train_metrics.items():
        print(f"{k}: {v:.6f}")

    print("\nEvaluating test cycles...")
    for cycle_df in test_frames:
        cycle_df = cycle_df.copy().reset_index(drop=True)
        x_test, y_test, target_indices = build_windows_from_cycle(
            cycle_df,
            feature_mean=feature_mean,
            feature_std=feature_std,
            stride=TEST_STRIDE,
            feature_columns=feature_columns,
        )
        y_pred = predict_windows(model, x_test, device)

        cycle_df[prediction_col] = np.nan
        cycle_df.loc[target_indices, prediction_col] = y_pred
        cycle_df["soc_error_percent"] = cycle_df[prediction_col] - cycle_df["soc_ref_ah"]

        metrics = evaluate_soc(y_test.reshape(-1), y_pred)
        file_name = cycle_df["file_name"].iloc[0]
        records.append(
            {
                "split": "test",
                "file_name": file_name,
                "cycle_name": cycle_df["cycle_name"].iloc[0],
                "sample_count": len(y_test),
                **metrics,
            }
        )

        safe_name = make_safe_name(file_name)
        cycle_df.to_csv(output_dir / f"{prediction_col}_prediction_{safe_name}.csv", index=False)
        plot_prediction(cycle_df, metrics, output_dir, prediction_col, method_name)
        prediction_frames.append(cycle_df)

        print(
            f"{file_name}: "
            f"MAE={metrics['MAE_percent']:.4f}%, "
            f"RMSE={metrics['RMSE_percent']:.4f}%, "
            f"P95={metrics['P95_ABS_ERROR_percent']:.4f}%, "
            f"MAX={metrics['MAX_ERROR_percent']:.4f}%"
        )

    results_df = pd.DataFrame(records)
    test_only = results_df[results_df["split"] == "test"]
    average_record = {
        "split": "test_average",
        "file_name": "AVERAGE_TEST_CYCLES",
        "cycle_name": "test_average",
        "sample_count": int(test_only["sample_count"].sum()),
        "MAE_percent": float(test_only["MAE_percent"].mean()),
        "RMSE_percent": float(test_only["RMSE_percent"].mean()),
        "MAX_ERROR_percent": float(test_only["MAX_ERROR_percent"].mean()),
        "P95_ABS_ERROR_percent": float(test_only["P95_ABS_ERROR_percent"].mean()),
        "P99_ABS_ERROR_percent": float(test_only["P99_ABS_ERROR_percent"].mean()),
        "FINAL_ERROR_percent": float(test_only["FINAL_ERROR_percent"].mean()),
    }
    results_df = pd.concat([results_df, pd.DataFrame([average_record])], ignore_index=True)

    pd.concat(prediction_frames, ignore_index=True).to_csv(
        output_dir / f"{prediction_col}_test_predictions.csv",
        index=False,
    )
    return results_df


def plot_prediction(df, metrics, output_dir, prediction_col, method_name):
    file_name = df["file_name"].iloc[0]
    safe_name = make_safe_name(file_name)
    plot_df = df.dropna(subset=[prediction_col]).copy()

    plt.figure(figsize=(10, 6))
    plt.plot(df["time_s"], df["soc_ref_ah"], label="Ah-based Reference SOC", linewidth=2)
    plt.plot(plot_df["time_s"], plot_df[prediction_col], label=f"{method_name} SOC", linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel("SOC (%)")
    plt.title(f"{method_name} SOC Estimation: {file_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / f"{prediction_col}_soc_prediction_{safe_name}.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(plot_df["time_s"], plot_df["soc_error_percent"])
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xlabel("Time (s)")
    plt.ylabel("SOC Error (%)")
    plt.title(
        f"{method_name} Error: {file_name} "
        f"(RMSE={metrics['RMSE_percent']:.3f}%, MAE={metrics['MAE_percent']:.3f}%)"
    )
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / f"{prediction_col}_soc_error_{safe_name}.png", dpi=200)
    plt.close()


def save_checkpoint(path, model, config, feature_mean, feature_std, extra=None):
    payload = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "feature_columns": config.get("feature_columns", FEATURE_COLUMNS),
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "sequence_length": SEQUENCE_LENGTH,
        "downsample_step": DOWNSAMPLE_STEP,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
