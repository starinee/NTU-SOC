from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.io import loadmat


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = PROJECT_ROOT / "dataset/dataset_trad/Panasonic 18650PF Data"
OUTPUT_DIR = PROJECT_ROOT / "dataset/processed/panasonic_raw_csv"

FIELD_MAP = {
    "Time": "time_s",
    "Voltage": "voltage_V",
    "Current": "current_A",
    "Ah": "ah",
    "Wh": "wh",
    "Power": "power_W",
    "Battery_Temp_degC": "battery_temp_C",
    "Chamber_Temp_degC": "chamber_temp_C",
}


def _get_struct_field(struct_obj, field_name):
    if hasattr(struct_obj, field_name):
        return getattr(struct_obj, field_name)
    return None


def _to_1d_array(value):
    if value is None:
        return None
    arr = np.asarray(value).squeeze()
    if arr.ndim == 0:
        return np.array([arr.item()])
    return arr.reshape(-1)


def read_meas_mat(file_path):
    mat = loadmat(file_path, squeeze_me=True, struct_as_record=False)
    if "meas" not in mat:
        useful_keys = [key for key in mat.keys() if not key.startswith("__")]
        raise KeyError(f"No 'meas' variable found in {file_path}. Available keys: {useful_keys}")

    extracted = {}
    for matlab_field, clean_col in FIELD_MAP.items():
        arr = _to_1d_array(_get_struct_field(mat["meas"], matlab_field))
        if arr is not None:
            extracted[clean_col] = arr
    if not extracted:
        raise ValueError(f"No usable fields extracted from {file_path}")

    main_len = max(len(v) for v in extracted.values())
    clean_data = {}
    for col, arr in extracted.items():
        if len(arr) == main_len:
            clean_data[col] = arr
        elif len(arr) == 1:
            clean_data[col] = np.repeat(arr[0], main_len)
        else:
            print(f"Warning: skipping {col} in {file_path.name}, length={len(arr)}, expected={main_len}")

    df = pd.DataFrame(clean_data)
    if "time_s" in df.columns:
        df["time_s"] = df["time_s"] - df["time_s"].iloc[0]
    return df


def parse_panasonic_metadata(file_path, raw_root):
    rel_parts = file_path.relative_to(raw_root).parts
    top_folder = rel_parts[0]
    file_name = file_path.name
    file_lower = file_name.lower()

    temp_match = re.search(r"(-?\d+)degc", top_folder.lower())
    ambient_temp_C = int(temp_match.group(1)) if temp_match else np.nan

    if "trise with pause" in top_folder.lower():
        temperature_profile = "trise_with_pause"
    elif "trise" in top_folder.lower():
        temperature_profile = "trise"
    else:
        temperature_profile = "constant"

    if len(rel_parts) >= 2:
        possible_group = rel_parts[1]
        test_group = possible_group if possible_group.lower() in ["5 pulse", "5 pulse test", "5 pulse disch", "charges and pauses", "drive cycles", "eis"] else temperature_profile
    else:
        test_group = temperature_profile

    if "eis" in file_lower or test_group.lower() == "eis":
        test_type = "eis"
    elif "5pulse" in file_lower or "dispulse" in file_lower or "pulse" in test_group.lower():
        test_type = "hppc_pulse"
    elif any(x in file_lower for x in ["udds", "us06", "la92", "hwfet", "nn", "mix"]) or "drive" in test_group.lower():
        test_type = "drive_cycle"
    elif "prechg" in file_lower:
        test_type = "pre_charge"
    elif "charge" in file_lower:
        test_type = "charge"
    elif "pause" in file_lower:
        test_type = "pause"
    elif "cycle" in file_lower:
        test_type = "cycle"
    else:
        test_type = "unknown"

    cycle_candidates = ["UDDS", "US06", "LA92", "HWFET", "NN", "MIX"]
    found_cycles = [cycle for cycle in cycle_candidates if cycle.lower() in file_lower]
    cycle_name = "_".join(found_cycles) if found_cycles else ""

    cell_match = re.search(r"\b(\d{4})\b", file_name)
    cell_id = cell_match.group(1) if cell_match else ""

    return {
        "ambient_temp_C": ambient_temp_C,
        "temperature_profile": temperature_profile,
        "test_group": test_group,
        "test_type": test_type,
        "cycle_name": cycle_name,
        "cell_id": cell_id,
        "file_name": file_name,
        "source_path": str(file_path),
    }


def convert_one_file(file_path, raw_root, output_dir):
    df = read_meas_mat(file_path)
    metadata = parse_panasonic_metadata(file_path, raw_root)
    for key, value in metadata.items():
        df[key] = value

    rel = file_path.relative_to(raw_root)
    safe_name = "__".join(rel.with_suffix("").parts)
    output_path = output_dir / f"{safe_name}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def convert_all(raw_root=RAW_ROOT, output_dir=OUTPUT_DIR, skip_eis=True):
    raw_root = Path(raw_root)
    output_dir = Path(output_dir)
    if not raw_root.exists():
        raise FileNotFoundError(
            f"Missing raw Panasonic MAT dataset folder: {raw_root}. "
            "Place the dataset there or pass raw_root to convert_all()."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    mat_files = sorted(raw_root.rglob("*.mat"))
    if skip_eis:
        mat_files = [path for path in mat_files if "eis" not in str(path).lower()]
    if not mat_files:
        raise FileNotFoundError(
            f"No .mat files found under {raw_root}. The reproducible pipeline requires the original Panasonic MAT files."
        )

    records = []
    print(f"Found {len(mat_files)} .mat files under {raw_root}")
    for i, file_path in enumerate(mat_files, start=1):
        try:
            output_path = convert_one_file(file_path, raw_root, output_dir)
            meta = parse_panasonic_metadata(file_path, raw_root)
            meta["output_csv"] = str(output_path)
            records.append(meta)
            print(f"[{i}/{len(mat_files)}] OK: {file_path.name}")
        except Exception as exc:
            print(f"[{i}/{len(mat_files)}] FAILED: {file_path.name}")
            print(f"    {exc}")

    manifest = pd.DataFrame(records)
    manifest_path = output_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"Manifest saved to: {manifest_path}")
    return manifest


if __name__ == "__main__":
    convert_all()
