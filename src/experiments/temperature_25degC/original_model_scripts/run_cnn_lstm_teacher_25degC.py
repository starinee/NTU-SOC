from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]

import pandas as pd

from cnn_lstm_utils_25degC import (
    build_training_windows,
    count_parameters,
    evaluate_model_on_splits,
    fit_feature_scaler,
    get_device,
    load_split_data,
    make_model,
    save_checkpoint,
    train_supervised,
)


OUTPUT_DIR = PROJECT_ROOT / "dataset/processed/cnn_lstm_teacher_25degC"

TEACHER_CONFIG = {
    "model_name": "cnn_lstm_teacher_25degC",
    "conv_channels": 64,
    "lstm_hidden": 64,
    "dense_hidden": 32,
    "dropout": 0.05,
}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")
    print(f"Output directory: {OUTPUT_DIR}")

    _, train_frames, test_frames = load_split_data()
    train_df, feature_mean, feature_std = fit_feature_scaler(train_frames)
    x_train, y_train = build_training_windows(train_frames, feature_mean, feature_std)

    print(f"\nTraining rows after downsampling: {len(train_df)}")
    print(f"Training windows: {x_train.shape}")
    print(f"Feature mean: {feature_mean}")
    print(f"Feature std: {feature_std}")

    model = make_model(TEACHER_CONFIG)
    param_count = count_parameters(model)
    print(f"CNN-LSTM teacher parameters: {param_count:,}")

    model, _ = train_supervised(
        model=model,
        x_train=x_train,
        y_train=y_train,
        output_dir=OUTPUT_DIR,
        device=device,
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
        prediction_col="soc_cnn_lstm_teacher",
        method_name="CNN-LSTM Teacher",
    )
    results_path = OUTPUT_DIR / "cnn_lstm_teacher_25degC_summary.csv"
    results_df.to_csv(results_path, index=False)

    checkpoint_path = OUTPUT_DIR / "cnn_lstm_teacher_25degC_model.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        TEACHER_CONFIG,
        feature_mean,
        feature_std,
        extra={"parameter_count": param_count},
    )

    print("\nSaved:")
    print(results_path)
    print(checkpoint_path)
    print("\nSummary:")
    print(pd.read_csv(results_path).to_string(index=False))


if __name__ == "__main__":
    main()
