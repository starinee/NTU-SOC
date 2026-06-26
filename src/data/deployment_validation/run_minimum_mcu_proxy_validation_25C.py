from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_ROOT = PROJECT_ROOT / "dataset/processed"
OUT = PROCESSED_ROOT / "deployment_validation/minimum_acceptable_mcu_proxy_25C"
OUT.mkdir(parents=True, exist_ok=True)

SEQ = 60
INPUT_SIZE = 9
CONV_CHANNELS = 16
LSTM_HIDDEN = 16
DENSE_HIDDEN = 8
CPU = torch.device("cpu")
CHECKPOINT = PROCESSED_ROOT / "cnn_lstm_distilled_student_filtered_features_25degC/cnn_lstm_distilled_student_filtered_features_25degC_model.pt"


class CNNLSTMRegressor(nn.Module):
    def __init__(self, input_size=9, conv_channels=16, lstm_hidden=16, dense_hidden=8, dropout=0.05):
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


def load_model():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing distilled model checkpoint: {CHECKPOINT}")
    payload = torch.load(CHECKPOINT, map_location=CPU, weights_only=False)
    state = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
    model = CNNLSTMRegressor().to(CPU).eval()
    model.load_state_dict(state)
    return model, state


def benchmark(model, runs=1000, warmup=100):
    x = torch.randn(1, SEQ, INPUT_SIZE, device=CPU)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        start = time.perf_counter()
        for _ in range(runs):
            model(x)
        elapsed = time.perf_counter() - start
    return elapsed / runs * 1000.0


def quantize_state_dict_to_int8_archive(state):
    archive = {}
    meta = []
    for name, tensor in state.items():
        arr = tensor.detach().cpu().numpy().astype(np.float32)
        max_abs = float(np.max(np.abs(arr))) if arr.size else 0.0
        scale = max_abs / 127.0 if max_abs > 0 else 1.0
        q = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
        key = name.replace(".", "__")
        archive[key] = q
        archive[key + "__scale"] = np.array([scale], dtype=np.float32)
        meta.append({"tensor": name, "shape": list(arr.shape), "scale": scale, "int8_bytes": int(q.nbytes)})
    np.savez_compressed(OUT / "filtered_distilled_tiny_cnn_lstm_int8_weight_archive.npz", **archive)
    pd.DataFrame(meta).to_csv(OUT / "int8_weight_archive_tensors.csv", index=False)


def estimate_macs():
    conv1 = SEQ * CONV_CHANNELS * INPUT_SIZE * 5
    conv2 = SEQ * CONV_CHANNELS * CONV_CHANNELS * 3
    lstm = SEQ * 4 * (CONV_CHANNELS * LSTM_HIDDEN + LSTM_HIDDEN * LSTM_HIDDEN + LSTM_HIDDEN)
    head = LSTM_HIDDEN * DENSE_HIDDEN + DENSE_HIDDEN + DENSE_HIDDEN * 1 + 1
    return int(conv1 + conv2 + lstm + head)


def estimate_ram_bytes(params):
    input_bytes = SEQ * INPUT_SIZE * 4
    conv_activation = SEQ * CONV_CHANNELS * 4
    lstm_state = 2 * LSTM_HIDDEN * 4
    dense_activation = DENSE_HIDDEN * 4
    activation_peak = input_bytes + conv_activation + lstm_state + dense_activation
    int8_weight_bytes = params
    return int(activation_peak + int8_weight_bytes), int(activation_peak)


def markdown_table(df, path):
    lines = [
        "| " + " | ".join(df.columns) + " |",
        "| " + " | ".join(["---"] * len(df.columns)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in df.columns) + " |")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    model, state = load_model()
    params = int(sum(p.numel() for p in model.parameters()))
    latency_ms = benchmark(model)
    macs = estimate_macs()
    flops = macs * 2
    est_total_ram, est_activation_ram = estimate_ram_bytes(params)

    example = torch.randn(1, SEQ, INPUT_SIZE, device=CPU)
    traced = torch.jit.trace(model, example)
    traced.save(str(OUT / "filtered_distilled_tiny_cnn_lstm_torchscript.pt"))
    quantize_state_dict_to_int8_archive(state)

    int8_archive_size = (OUT / "filtered_distilled_tiny_cnn_lstm_int8_weight_archive.npz").stat().st_size
    torchscript_size = (OUT / "filtered_distilled_tiny_cnn_lstm_torchscript.pt").stat().st_size
    checkpoint_size = CHECKPOINT.stat().st_size

    rows = [{
        "model": "Filtered Distilled Tiny CNN-LSTM",
        "export_artifact": "TorchScript + per-tensor INT8 weight archive",
        "physical_mcu_tested": "No",
        "parameter_count": params,
        "checkpoint_size_KB": round(checkpoint_size / 1024, 2),
        "torchscript_size_KB": round(torchscript_size / 1024, 2),
        "compressed_int8_archive_KB": round(int8_archive_size / 1024, 2),
        "raw_int8_weight_KB_est": round(params / 1024, 2),
        "estimated_activation_ram_KB": round(est_activation_ram / 1024, 2),
        "estimated_total_ram_KB": round(est_total_ram / 1024, 2),
        "estimated_MACs_per_inference": macs,
        "estimated_FLOPs_per_inference": flops,
        "cpu_batch1_latency_ms": round(latency_ms, 4),
        "statement": "Deployment-oriented proxy validation only; not measured on ESP32/STM32/Arduino hardware.",
    }]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "minimum_acceptable_mcu_proxy_25C.csv", index=False)
    markdown_table(df, OUT / "minimum_acceptable_mcu_proxy_25C.md")
    (OUT / "README.md").write_text(
        "Minimum acceptable deployment proxy for the filtered distilled tiny CNN-LSTM. "
        "This folder contains a TorchScript export and a per-tensor INT8 weight archive. "
        "It is a TFLite-like embedded proxy artifact, not physical MCU validation. "
        "The report includes model size, estimated RAM, MACs/FLOPs, and CPU batch-1 latency.\n",
        encoding="utf-8",
    )
    print(df.to_string(index=False))
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
