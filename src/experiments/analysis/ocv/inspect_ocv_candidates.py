from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
MANIFEST_PATH = PROJECT_ROOT / "dataset/processed/panasonic_raw_csv/manifest.csv"

manifest = pd.read_csv(MANIFEST_PATH)

print("Manifest shape:", manifest.shape)
print("\nColumns:")
print(manifest.columns.tolist())

print("\nAll file count by temperature / test group / test type:")
print(
    manifest
    .groupby(["ambient_temp_C", "temperature_profile", "test_group", "test_type"])
    .size()
)

print("\nFiles under 25degC:")
m25 = manifest[
    (manifest["ambient_temp_C"] == 25) |
    (manifest["file_name"].str.contains("25degC", case=False, na=False))
]

print(m25[["ambient_temp_C", "temperature_profile", "test_group", "test_type", "file_name", "output_csv"]])

print("\nPotential OCV-related files: pause / charge / pre_charge / Charges and Pauses")
ocv_candidates = manifest[
    manifest["test_type"].isin(["pause", "charge", "pre_charge"])
    |
    manifest["test_group"].astype(str).str.contains("Charges", case=False, na=False)
    |
    manifest["file_name"].astype(str).str.contains("pause|charge|prechg|dis", case=False, na=False)
]

print(ocv_candidates[[
    "ambient_temp_C",
    "temperature_profile",
    "test_group",
    "test_type",
    "file_name",
    "output_csv"
]].to_string())

print("\nOCV candidate count by temperature:")
print(
    ocv_candidates
    .groupby(["ambient_temp_C", "temperature_profile", "test_type"])
    .size()
)
