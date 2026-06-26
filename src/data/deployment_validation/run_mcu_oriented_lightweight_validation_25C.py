from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_ROOT = PROJECT_ROOT / "dataset/processed"
OUTPUT_DIR = PROCESSED_ROOT / "deployment_validation/mcu_oriented_lightweight_validation_25C"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEQUENCE_LENGTH = 60
CPU_DEVICE = torch.device("cpu")
WARMUP_RUNS = 50
TIMED_RUNS = 300


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


def mlp_parameter_count(input_size, hidden_layers):
    total = 0
    prev = input_size
    for width in hidden_layers:
        total += prev * width + width
        prev = width
    total += prev * 1 + 1
    return int(total)


def count_parameters(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def model_size_mb(path):
    return Path(path).stat().st_size / (1024 * 1024)


def load_state_dict_payload(path):
    payload = torch.load(path, map_location=CPU_DEVICE, weights_only=False)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"], payload
    if isinstance(payload, dict):
        return payload, {}
    raise ValueError(f"Unsupported checkpoint format: {path}")


def benchmark_torch_model(model, input_size, batch_size):
    model = model.to(CPU_DEVICE).eval()
    x = torch.randn(batch_size, SEQUENCE_LENGTH, input_size, device=CPU_DEVICE)
    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            _ = model(x)
        start = time.perf_counter()
        for _ in range(TIMED_RUNS):
            _ = model(x)
        elapsed = time.perf_counter() - start
    per_batch_ms = elapsed / TIMED_RUNS * 1000.0
    per_sample_ms = per_batch_ms / batch_size
    return per_batch_ms, per_sample_ms


def quantized_checkpoint_size_estimate_mb(parameter_count, bytes_per_parameter):
    return parameter_count * bytes_per_parameter / (1024 * 1024)


def markdown_table(df, path):
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


TORCH_MODELS = [
    {
        "method": "LSTM",
        "feature_set": "Raw V/I/T",
        "input_size": 3,
        "checkpoint": PROCESSED_ROOT / "lstm_baseline_25degC/lstm_baseline_25degC_model.pt",
        "factory": lambda: LSTMRegressor(3, hidden=32),
    },
    {
        "method": "CNN-LSTM Teacher",
        "feature_set": "Raw V/I/T",
        "input_size": 3,
        "checkpoint": PROCESSED_ROOT / "cnn_lstm_teacher_25degC/cnn_lstm_teacher_25degC_model.pt",
        "factory": lambda: CNNLSTMRegressor(3, conv_channels=64, lstm_hidden=64, dense_hidden=32),
    },
    {
        "method": "Tiny CNN-LSTM Student",
        "feature_set": "Raw V/I/T",
        "input_size": 3,
        "checkpoint": PROCESSED_ROOT / "cnn_lstm_student_25degC/cnn_lstm_student_25degC_model.pt",
        "factory": lambda: CNNLSTMRegressor(3, conv_channels=16, lstm_hidden=16, dense_hidden=8),
    },
    {
        "method": "Distilled Tiny CNN-LSTM",
        "feature_set": "Raw V/I/T",
        "input_size": 3,
        "checkpoint": PROCESSED_ROOT / "cnn_lstm_distilled_student_25degC/cnn_lstm_distilled_student_25degC_model.pt",
        "factory": lambda: CNNLSTMRegressor(3, conv_channels=16, lstm_hidden=16, dense_hidden=8),
    },
    {
        "method": "Filtered CNN-LSTM Teacher",
        "feature_set": "Raw + EMA V/I",
        "input_size": 9,
        "checkpoint": PROCESSED_ROOT / "cnn_lstm_teacher_filtered_features_25degC/cnn_lstm_teacher_filtered_features_25degC_model.pt",
        "factory": lambda: CNNLSTMRegressor(9, conv_channels=64, lstm_hidden=64, dense_hidden=32),
    },
    {
        "method": "Filtered Tiny CNN-LSTM Student",
        "feature_set": "Raw + EMA V/I",
        "input_size": 9,
        "checkpoint": PROCESSED_ROOT / "cnn_lstm_student_filtered_features_25degC/cnn_lstm_student_filtered_features_25degC_model.pt",
        "factory": lambda: CNNLSTMRegressor(9, conv_channels=16, lstm_hidden=16, dense_hidden=8),
    },
    {
        "method": "Filtered Distilled Tiny CNN-LSTM",
        "feature_set": "Raw + EMA V/I",
        "input_size": 9,
        "checkpoint": PROCESSED_ROOT / "cnn_lstm_distilled_student_filtered_features_25degC/cnn_lstm_distilled_student_filtered_features_25degC_model.pt",
        "factory": lambda: CNNLSTMRegressor(9, conv_channels=16, lstm_hidden=16, dense_hidden=8),
    },
]

MLP_MODELS = [
    {
        "method": "Instantaneous MLP",
        "feature_set": "Raw V/I/T",
        "input_size": 3,
        "hidden_layers": [64, 32],
        "source": "architecture_only_no_checkpoint_saved",
    },
    {
        "method": "Filtered-feature MLP",
        "feature_set": "Raw + EMA V/I",
        "input_size": 9,
        "hidden_layers": [128, 64, 32],
        "source": "architecture_only_no_checkpoint_saved",
    },
]


def main():
    rows = []

    for spec in MLP_MODELS:
        params = mlp_parameter_count(spec["input_size"], spec["hidden_layers"])
        rows.append(
            {
                "method": spec["method"],
                "feature_set": spec["feature_set"],
                "source": spec["source"],
                "parameter_count": params,
                "checkpoint_size_MB": np.nan,
                "fp32_weight_size_MB_est": quantized_checkpoint_size_estimate_mb(params, 4),
                "int8_weight_size_MB_est": quantized_checkpoint_size_estimate_mb(params, 1),
                "cpu_batch1_latency_ms": np.nan,
                "cpu_batch256_latency_ms": np.nan,
                "cpu_batch256_per_sample_ms": np.nan,
            }
        )

    for spec in TORCH_MODELS:
        checkpoint = spec["checkpoint"]
        if not checkpoint.exists():
            print(f"Skipping missing checkpoint: {checkpoint}")
            continue
        state_dict, payload = load_state_dict_payload(checkpoint)
        model = spec["factory"]()
        model.load_state_dict(state_dict)
        params = count_parameters(model)
        batch1_ms, sample1_ms = benchmark_torch_model(model, spec["input_size"], batch_size=1)
        batch256_ms, sample256_ms = benchmark_torch_model(model, spec["input_size"], batch_size=256)
        rows.append(
            {
                "method": spec["method"],
                "feature_set": spec["feature_set"],
                "source": checkpoint.name,
                "parameter_count": params,
                "checkpoint_size_MB": model_size_mb(checkpoint),
                "fp32_weight_size_MB_est": quantized_checkpoint_size_estimate_mb(params, 4),
                "int8_weight_size_MB_est": quantized_checkpoint_size_estimate_mb(params, 1),
                "cpu_batch1_latency_ms": batch1_ms,
                "cpu_batch256_latency_ms": batch256_ms,
                "cpu_batch256_per_sample_ms": sample256_ms,
            }
        )

    df = pd.DataFrame(rows)
    teacher_params = dict(zip(df["method"], df["parameter_count"]))
    refs = {
        "Tiny CNN-LSTM Student": "CNN-LSTM Teacher",
        "Distilled Tiny CNN-LSTM": "CNN-LSTM Teacher",
        "Filtered Tiny CNN-LSTM Student": "Filtered CNN-LSTM Teacher",
        "Filtered Distilled Tiny CNN-LSTM": "Filtered CNN-LSTM Teacher",
    }
    df["reference_teacher"] = df["method"].map(refs)
    df["params_vs_teacher_percent"] = np.nan
    df["parameter_reduction_vs_teacher_percent"] = np.nan
    for idx, row in df.iterrows():
        ref = row["reference_teacher"]
        if pd.isna(ref) or ref not in teacher_params:
            continue
        ratio = row["parameter_count"] / teacher_params[ref] * 100.0
        df.loc[idx, "params_vs_teacher_percent"] = ratio
        df.loc[idx, "parameter_reduction_vs_teacher_percent"] = 100.0 - ratio

    df = df.sort_values(["feature_set", "parameter_count", "method"]).reset_index(drop=True)
    df.to_csv(OUTPUT_DIR / "mcu_oriented_lightweight_validation_25C.csv", index=False)
    markdown_table(df, OUTPUT_DIR / "mcu_oriented_lightweight_validation_25C.md")
    print(df.to_string(index=False))
    print(f"\nSaved MCU-oriented validation outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
