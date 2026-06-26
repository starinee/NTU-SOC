#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"

echo "[1/4] Rebuilding CSV manifest and running strict matched temperature study"
"$PYTHON_BIN" src/experiments/run_reproducible_temperature_study.py

echo "[2/4] Generating profile-wise and error-analysis figures"
"$PYTHON_BIN" src/data/temperature_experiments/generate_profile_error_analysis_10C_0C.py

echo "[3/4] Running MCU-oriented lightweight validation"
"$PYTHON_BIN" src/data/deployment_validation/run_mcu_oriented_lightweight_validation_25C.py

echo "[4/4] Running CC current-noise sensitivity"
"$PYTHON_BIN" src/data/traditional_baselines/run_cc_sensitivity_current_noise_25C.py

echo "Reproducible pipeline completed."
