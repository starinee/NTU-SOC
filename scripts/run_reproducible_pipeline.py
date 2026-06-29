#!/usr/bin/env python3
from pathlib import Path
import argparse
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "dataset/processed/panasonic_raw_csv/manifest.csv"


STRICT_SCRIPT = "src/data/temperature_experiments/run_strict_matched_temperature_pipeline_25C_10C_0C.py"
TABLE_SCRIPT = "src/experiments/analysis/paper_outputs/make_final_paper_tables_25degC.py"
DEPLOYMENT_SCRIPTS = [
    "src/data/deployment_validation/run_mcu_oriented_lightweight_validation_25C.py",
    "src/data/deployment_validation/run_minimum_mcu_proxy_validation_25C.py",
]


def run(script):
    path = PROJECT_ROOT / script
    if not path.exists():
        raise FileNotFoundError(f"Missing pipeline step: {script}")
    print(f"\n=== Running {script} ===", flush=True)
    subprocess.run([sys.executable, str(path)], cwd=PROJECT_ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run the strict SOC-estimation reproduction pipeline.")
    parser.add_argument(
        "--temperature",
        choices=["all", "strict", "tables", "deployment"],
        default="all",
        help=(
            "Use all to refresh paper tables and deployment proxy outputs from the committed strict results. "
            "Use strict to rerun model training, which requires the processed Panasonic manifest."
        ),
    )
    args = parser.parse_args()

    if args.temperature == "strict":
        if not MANIFEST_PATH.exists():
            raise FileNotFoundError(
                f"Missing processed dataset manifest: {MANIFEST_PATH}\n"
                "Place the Panasonic raw MAT files under dataset/dataset_trad/Panasonic 18650PF Data/ "
                "and run src/data_processing/convert_panasonic_mat_to_csv.py first."
            )
        run(STRICT_SCRIPT)
        print("\nStrict matched pipeline completed.")
        return

    if args.temperature == "tables":
        run(TABLE_SCRIPT)
        print("\nFinal paper tables regenerated.")
        return

    if args.temperature == "deployment":
        for script in DEPLOYMENT_SCRIPTS:
            run(script)
        print("\nDeployment-oriented proxy outputs regenerated.")
        return

    if MANIFEST_PATH.exists():
        run(STRICT_SCRIPT)
    else:
        print(
            f"\nProcessed dataset manifest not found at {MANIFEST_PATH}.\n"
            "Skipping model retraining and using the committed strict matched outputs. "
            "Run with --temperature strict after preparing the Panasonic dataset to fully retrain.",
            flush=True,
        )
    run(TABLE_SCRIPT)
    print(
        "\nCommitted deployment-oriented proxy outputs were not overwritten by --temperature all. "
        "CPU latency is machine-dependent; run --temperature deployment only when you intentionally "
        "want to refresh those proxy measurements.",
        flush=True,
    )
    print("\nReproducible pipeline completed.")


if __name__ == "__main__":
    main()
