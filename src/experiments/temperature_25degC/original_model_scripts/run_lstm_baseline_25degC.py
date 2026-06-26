from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# =========================
# Paths
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[4]
MANIFEST_PATH = PROJECT_ROOT / "dataset/processed/panasonic_raw_csv/manifest.csv"
OUTPUT_DIR = PROJECT_ROOT / "dataset/processed/lstm_baseline_25degC"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Experiment config
# =========================
CAPACITY_AH = 2.9
SOC0 = 100.0

# Raw drive-cycle data is usually 0.1 s. Step 10 gives about 1 Hz.
DOWNSAMPLE_STEP = 10

# At 1 Hz, this means the model sees the previous 60 seconds.
SEQUENCE_LENGTH = 60
TRAIN_STRIDE = 5
TEST_STRIDE = 1

FEATURE_COLUMNS = [
    "voltage_V",
    "current_A",
    "battery_temp_C",
]

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

RANDOM_SEED = 42
BATCH_SIZE = 256
MAX_EPOCHS = 80
PATIENCE = 12
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5


def make_safe_name(file_name: str) -> str:
    return Path(file_name).stem.replace(" ", "_").replace(".", "p")


def soc_from_ah(
    df: pd.DataFrame,
    ah_col: str = "ah",
    capacity_ah: float = CAPACITY_AH,
    soc0: float = SOC0,
) -> np.ndarray:
    ah = df[ah_col].to_numpy(dtype=float)
    soc_ref = soc0 + ah / capacity_ah * 100.0
    return np.clip(soc_ref, 0.0, 100.0)


def evaluate_soc(y_true, y_pred) -> dict:
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


def contains_any(text: str, keywords: list[str]) -> bool:
    text_lower = str(text).lower()
    return any(k.lower() in text_lower for k in keywords)


def select_25degC_drive_files(manifest: pd.DataFrame) -> pd.DataFrame:
    drive_25 = manifest[
        manifest["file_name"].astype(str).str.contains("25degC", case=False, na=False)
        & manifest["file_name"].astype(str).str.contains(
            "Cycle_1|Cycle_2|Cycle_3|Cycle_4|US06|UDDS|LA92|NN",
            case=False,
            na=False,
            regex=True,
        )
    ].copy()

    # Keep split files only for a clean first LSTM baseline.
    drive_25 = drive_25[
        ~drive_25["file_name"].astype(str).str.contains("HWFT|HWFET", case=False, na=False)
    ].copy()

    if len(drive_25) == 0:
        raise ValueError("No 25degC drive-cycle files found.")

    return drive_25.sort_values("file_name").reset_index(drop=True)


def assign_split(row: pd.Series) -> str:
    file_name = row["file_name"]
    if contains_any(file_name, TRAIN_CYCLE_KEYWORDS):
        return "train"
    if contains_any(file_name, TEST_CYCLE_KEYWORDS):
        return "test"
    return "unused"


def load_one_cycle(row: pd.Series, split: str) -> pd.DataFrame:
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

    df["soc_ref_ah"] = soc_from_ah(df)
    df["split"] = split
    df["cycle_name"] = cycle_name
    df["file_name"] = file_name

    return df


def build_windows_from_cycle(
    df: pd.DataFrame,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    stride: int,
):
    features = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
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


class SOCSequenceLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        output, _ = self.lstm(x)
        last_output = output[:, -1, :]
        return self.head(last_output)


def train_lstm(x_train, y_train, device):
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
    val_x = torch.from_numpy(x_train[val_idx]).to(device)
    val_y = torch.from_numpy(y_train[val_idx]).to(device)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SOCSequenceLSTM(input_size=len(FEATURE_COLUMNS)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    print("\nTraining LSTM baseline...")
    for epoch in range(1, MAX_EPOCHS + 1):
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
        print(f"Epoch {epoch:03d}: train_RMSE={train_rmse:.4f}, val_RMSE={val_rmse:.4f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def predict_windows(model, x, device) -> np.ndarray:
    model.eval()
    preds = []
    loader = DataLoader(torch.from_numpy(x), batch_size=BATCH_SIZE, shuffle=False)
    with torch.no_grad():
        for batch_x in loader:
            pred = model(batch_x.to(device)).cpu().numpy().reshape(-1)
            preds.append(pred)
    y_pred = np.concatenate(preds)
    return np.clip(y_pred, 0.0, 100.0)


def plot_prediction(df: pd.DataFrame, metrics: dict, output_dir: Path):
    file_name = df["file_name"].iloc[0]
    safe_name = make_safe_name(file_name)

    plot_df = df.dropna(subset=["soc_lstm"]).copy()

    plt.figure(figsize=(10, 6))
    plt.plot(df["time_s"], df["soc_ref_ah"], label="Ah-based Reference SOC", linewidth=2)
    plt.plot(plot_df["time_s"], plot_df["soc_lstm"], label="LSTM Predicted SOC", linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel("SOC (%)")
    plt.title(f"LSTM SOC Estimation: {file_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / f"lstm_soc_prediction_{safe_name}.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(plot_df["time_s"], plot_df["soc_error_percent"])
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xlabel("Time (s)")
    plt.ylabel("SOC Error (%)")
    plt.title(
        f"LSTM Error: {file_name} "
        f"(RMSE={metrics['RMSE_percent']:.3f}%, MAE={metrics['MAE_percent']:.3f}%)"
    )
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / f"lstm_soc_error_{safe_name}.png", dpi=200)
    plt.close()


def main():
    print("Loading manifest:")
    print(MANIFEST_PATH)
    manifest = pd.read_csv(MANIFEST_PATH)

    drive_25 = select_25degC_drive_files(manifest)
    drive_25["split"] = drive_25.apply(assign_split, axis=1)

    print("\nSelected 25degC drive-cycle files:")
    print(drive_25[["file_name", "cycle_name", "split", "output_csv"]].to_string(index=False))

    train_rows = drive_25[drive_25["split"] == "train"]
    test_rows = drive_25[drive_25["split"] == "test"]

    train_frames = [load_one_cycle(row, "train") for _, row in train_rows.iterrows()]
    test_frames = [load_one_cycle(row, "test") for _, row in test_rows.iterrows()]

    train_df = pd.concat(train_frames, ignore_index=True)

    feature_mean = train_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32).mean(axis=0)
    feature_std = train_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32).std(axis=0)
    feature_std = np.where(feature_std == 0, 1.0, feature_std)

    train_windows = []
    train_targets = []
    for cycle_df in train_frames:
        x, y, _ = build_windows_from_cycle(
            cycle_df,
            feature_mean=feature_mean,
            feature_std=feature_std,
            stride=TRAIN_STRIDE,
        )
        train_windows.append(x)
        train_targets.append(y)

    x_train = np.concatenate(train_windows, axis=0)
    y_train = np.concatenate(train_targets, axis=0)

    print("\nTraining samples:", len(train_df))
    print("Training windows:", len(x_train))
    print("Sequence length:", SEQUENCE_LENGTH)
    print("Features:", FEATURE_COLUMNS)

    device = torch.device("cpu")
    model = train_lstm(x_train, y_train, device)

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
        )
        y_pred = predict_windows(model, x_test, device)

        cycle_df["soc_lstm"] = np.nan
        cycle_df.loc[target_indices, "soc_lstm"] = y_pred
        cycle_df["soc_error_percent"] = cycle_df["soc_lstm"] - cycle_df["soc_ref_ah"]

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
        cycle_df.to_csv(OUTPUT_DIR / f"lstm_prediction_{safe_name}.csv", index=False)
        plot_prediction(cycle_df, metrics, OUTPUT_DIR)
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

    summary_path = OUTPUT_DIR / "lstm_baseline_25degC_summary.csv"
    predictions_path = OUTPUT_DIR / "lstm_baseline_25degC_test_predictions.csv"
    model_path = OUTPUT_DIR / "lstm_baseline_25degC_model.pt"

    results_df.to_csv(summary_path, index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(predictions_path, index=False)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_columns": FEATURE_COLUMNS,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "sequence_length": SEQUENCE_LENGTH,
            "downsample_step": DOWNSAMPLE_STEP,
        },
        model_path,
    )

    print("\nLSTM baseline finished.")
    print("Summary saved to:")
    print(summary_path)
    print("\nCombined test predictions saved to:")
    print(predictions_path)
    print("\nModel saved to:")
    print(model_path)
    print("\nOutput plots saved to:")
    print(OUTPUT_DIR)
    print("\nSummary:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
