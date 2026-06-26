import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
MANIFEST_PATH = PROJECT_ROOT / "dataset/processed/panasonic_raw_csv/manifest.csv"
OUTPUT_DIR = PROJECT_ROOT / "dataset/processed/ocv_lookup_inspection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

manifest = pd.read_csv(MANIFEST_PATH)

# 选 25degC + C20 OCV 文件
c20_files = manifest[
    manifest["output_csv"].astype(str).str.contains("25degC", case=False, na=False)
    &
    manifest["output_csv"].astype(str).str.contains("C20 OCV", case=False, na=False)
].copy()

print("25degC C20 OCV related files:")
print(c20_files[["file_name", "test_type", "output_csv"]].to_string())

if len(c20_files) == 0:
    raise ValueError("No 25degC C20 OCV files found.")

for idx, row in c20_files.iterrows():
    file_name = row["file_name"]
    file_path = row["output_csv"]

    print("\nInspecting:", file_name)

    df = pd.read_csv(file_path)

    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    required_cols = ["time_s", "voltage_V", "current_A", "ah"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print("Missing columns:", missing)
        continue

    print(df[required_cols].describe())

    safe_name = Path(file_name).stem.replace(" ", "_").replace(".", "p")

    plt.figure(figsize=(10, 4))
    plt.plot(df["time_s"], df["voltage_V"])
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.title(f"Voltage: {file_name}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"voltage_{safe_name}.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(df["time_s"], df["current_A"])
    plt.xlabel("Time (s)")
    plt.ylabel("Current (A)")
    plt.title(f"Current: {file_name}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"current_{safe_name}.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(df["time_s"], df["ah"])
    plt.xlabel("Time (s)")
    plt.ylabel("Ah")
    plt.title(f"Ah: {file_name}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"ah_{safe_name}.png", dpi=200)
    plt.close()

print("\nInspection plots saved to:")
print(OUTPUT_DIR)
