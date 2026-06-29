#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -f dataset/processed/panasonic_raw_csv/manifest.csv ]]; then
  echo "[1/4] Running strict matched temperature study"
  "$PYTHON_BIN" src/data/temperature_experiments/run_strict_matched_temperature_pipeline_25C_10C_0C.py
else
  echo "[1/4] Processed manifest not found; using committed strict matched outputs"
  echo "      Run scripts/run_reproducible_pipeline.py --temperature strict after preparing the Panasonic dataset to retrain."
fi

echo "[2/4] Regenerating final paper tables"
"$PYTHON_BIN" src/experiments/analysis/paper_outputs/make_final_paper_tables_25degC.py

echo "[3/4] Keeping committed deployment-oriented proxy outputs"
echo "      CPU latency is machine-dependent; run the deployment scripts explicitly to refresh them."

echo "Reproducible pipeline completed."
