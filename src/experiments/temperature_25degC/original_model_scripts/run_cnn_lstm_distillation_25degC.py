from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]

import pandas as pd
import torch

from cnn_lstm_utils_25degC import (
    build_training_windows,
    count_parameters,
    evaluate_model_on_splits,
    fit_feature_scaler,
    get_device,
    load_split_data,
    make_model,
    save_checkpoint,
    train_distilled,
)


TEACHER_OUTPUT_DIR = PROJECT_ROOT / "dataset/processed/cnn_lstm_teacher_25degC"
TEACHER_CHECKPOINT = TEACHER_OUTPUT_DIR / "cnn_lstm_teacher_25degC_model.pt"
OUTPUT_DIR = PROJECT_ROOT / "dataset/processed/cnn_lstm_distilled_student_25degC"

STUDENT_CONFIG = {
    "model_name": "cnn_lstm_distilled_student_25degC",
    "conv_channels": 16,
    "lstm_hidden": 16,
    "dense_hidden": 8,
    "dropout": 0.05,
}

DISTILL_ALPHA = 0.6


def load_teacher(device):
    if not TEACHER_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Teacher checkpoint not found: {TEACHER_CHECKPOINT}\n"
            "Run run_cnn_lstm_teacher_25degC.py first."
        )
    payload = torch.load(TEACHER_CHECKPOINT, map_location="cpu", weights_only=False)
    teacher = make_model(payload["config"])
    teacher.load_state_dict(payload["model_state_dict"])
    teacher.to(device)
    teacher.eval()
    return teacher, payload


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")
    print(f"Teacher checkpoint: {TEACHER_CHECKPOINT}")
    print(f"Output directory: {OUTPUT_DIR}")

    teacher_model, teacher_payload = load_teacher(device)
    _, train_frames, test_frames = load_split_data()
    train_df, feature_mean, feature_std = fit_feature_scaler(train_frames)
    x_train, y_train = build_training_windows(train_frames, feature_mean, feature_std)

    print(f"\nTraining rows after downsampling: {len(train_df)}")
    print(f"Training windows: {x_train.shape}")
    print(f"Feature mean: {feature_mean}")
    print(f"Feature std: {feature_std}")

    model = make_model(STUDENT_CONFIG)
    param_count = count_parameters(model)
    teacher_param_count = teacher_payload.get("parameter_count", count_parameters(teacher_model))
    print(f"Teacher parameters: {teacher_param_count:,}")
    print(f"Distilled tiny CNN-LSTM student parameters: {param_count:,}")
    print(f"Distillation alpha: {DISTILL_ALPHA}")

    model, _ = train_distilled(
        student_model=model,
        teacher_model=teacher_model,
        x_train=x_train,
        y_train=y_train,
        output_dir=OUTPUT_DIR,
        device=device,
        alpha=DISTILL_ALPHA,
        max_epochs=80,
        patience=12,
        batch_size=256,
        learning_rate=1e-3,
        weight_decay=1e-5,
    )

    results_df = evaluate_model_on_splits(
        model=model,
        train_frames=train_frames,
        test_frames=test_frames,
        x_train=x_train,
        y_train=y_train,
        feature_mean=feature_mean,
        feature_std=feature_std,
        output_dir=OUTPUT_DIR,
        device=device,
        prediction_col="soc_cnn_lstm_distilled_student",
        method_name="Distilled Tiny CNN-LSTM",
    )
    results_path = OUTPUT_DIR / "cnn_lstm_distilled_student_25degC_summary.csv"
    results_df.to_csv(results_path, index=False)

    checkpoint_path = OUTPUT_DIR / "cnn_lstm_distilled_student_25degC_model.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        STUDENT_CONFIG,
        feature_mean,
        feature_std,
        extra={
            "parameter_count": param_count,
            "teacher_checkpoint": str(TEACHER_CHECKPOINT),
            "teacher_parameter_count": teacher_param_count,
            "distillation_alpha": DISTILL_ALPHA,
        },
    )

    print("\nSaved:")
    print(results_path)
    print(checkpoint_path)
    print("\nSummary:")
    print(pd.read_csv(results_path).to_string(index=False))


if __name__ == "__main__":
    main()
