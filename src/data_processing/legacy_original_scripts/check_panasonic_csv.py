from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
manifest_path = PROJECT_ROOT / "dataset/processed/panasonic_raw_csv/manifest.csv"

manifest = pd.read_csv(manifest_path)

print("Manifest shape:", manifest.shape)
print("\nColumns:")
print(manifest.columns.tolist())

print("\nFirst 5 rows:")
print(manifest.head())

print("\nFile count by type:")
print(manifest.groupby(["ambient_temp_C", "temperature_profile", "test_group", "test_type"]).size())


import matplotlib.pyplot as plt

# 选第一个 drive cycle 文件
drive_files = manifest[manifest["test_type"] == "drive_cycle"]["output_csv"].tolist()

print("\nNumber of drive cycle files:", len(drive_files))
print("First drive cycle file:")
print(drive_files[0])

df = pd.read_csv(drive_files[0])

print("\nDrive cycle CSV shape:", df.shape)

print("\nDrive cycle columns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nData description:")
print(df[["time_s", "voltage_V", "current_A", "ah"]].describe())

plt.figure()
plt.plot(df["time_s"], df["voltage_V"])
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.title("Voltage vs Time")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(df["time_s"], df["current_A"])
plt.xlabel("Time (s)")
plt.ylabel("Current (A)")
plt.title("Current vs Time")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(df["time_s"], df["ah"])
plt.xlabel("Time (s)")
plt.ylabel("Ah")
plt.title("Ah vs Time")
plt.grid(True)
plt.show()
