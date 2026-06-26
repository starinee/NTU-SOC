from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_ROOT = PROJECT_ROOT / "dataset/dataset_trad/Panasonic 18650PF Data"
CONVERTED_CSV_DIR = PROJECT_ROOT / "dataset/processed/panasonic_raw_csv"
MANIFEST_PATH = CONVERTED_CSV_DIR / "manifest.csv"

TEMPERATURES_C = [25, 10, 0]
TRANSFER_TRAIN_TEMPERATURE_C = 25
TRANSFER_TEST_TEMPERATURES_C = [25, 10, 0]

TRAIN_PROFILES = ["Cycle_1", "Cycle_2", "Cycle_3", "Cycle_4", "US06"]
TEST_PROFILES = ["UDDS", "LA92", "NN"]
EXCLUDED_PROFILES = ["HWFET", "LA92_NN", "US06_HWFET", "HWFET_UDDS"]

CAPACITY_AH = 2.9
INITIAL_SOC_PERCENT = 100.0
DOWNSAMPLE_STEP = 10
SEQUENCE_LENGTH = 60
TRAIN_STRIDE = 5
TEST_STRIDE = 1
RANDOM_SEED = 42

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

MODEL_SET = [
    "Instantaneous MLP",
    "Filtered-feature MLP",
    "LSTM",
    "CNN-LSTM Teacher",
    "Tiny CNN-LSTM Student",
    "Distilled Tiny CNN-LSTM",
    "Filtered CNN-LSTM Teacher",
    "Filtered Tiny CNN-LSTM Student",
    "Filtered Distilled Tiny CNN-LSTM",
]
