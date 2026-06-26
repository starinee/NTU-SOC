#!/usr/bin/env python3
from pathlib import Path
import argparse
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


STEPS = {
    "all": [
        "src/experiments/run_reproducible_temperature_study.py",
        "src/data/temperature_experiments/generate_profile_error_analysis_10C_0C.py",
        "src/data/deployment_validation/run_mcu_oriented_lightweight_validation_25C.py",
        "src/data/deployment_validation/run_minimum_mcu_proxy_validation_25C.py",
        "src/data/traditional_baselines/run_cc_sensitivity_current_noise_25C.py",
    ],
    "25": ["src/experiments/temperature_25degC/run_25degC_within_temperature.py"],
    "10": ["src/experiments/temperature_10degC/run_10degC_within_temperature.py"],
    "0": ["src/experiments/temperature_0degC/run_0degC_within_temperature.py"],
    "transfer": ["src/experiments/temperature_transfer/run_25degC_to_10degC_0degC_transfer.py"],
}


def run(script):
    path = PROJECT_ROOT / script
    if not path.exists():
        raise FileNotFoundError(f"Missing pipeline step: {script}")
    print(f"\n=== Running {script} ===", flush=True)
    subprocess.run([sys.executable, str(path)], cwd=PROJECT_ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run the reproducible SOC-estimation pipeline.")
    parser.add_argument(
        "--temperature",
        choices=sorted(STEPS),
        default="all",
        help="Use all for the complete matched pipeline, or run one temperature/transfer subset.",
    )
    args = parser.parse_args()
    for script in STEPS[args.temperature]:
        run(script)
    print("\nReproducible pipeline completed.")


if __name__ == "__main__":
    main()
