# GitHub Upload Guide

This directory is the clean public release tree. It is intended to be the
contents of the GitHub repository, not a local research working directory.

## Recommended Public Repository Contents

Upload these:

- `README.md`
- `DATASET.md`
- `RESULTS.md`
- `CODE_STRUCTURE.md`
- `requirements.txt`
- `.gitignore`
- `scripts/`
- `src/`
- `docs/`
- selected small result tables and figures under `dataset/processed/`

Do not upload these:

- `venv/`
- `.idea/`
- `.DS_Store`
- `dataset/dataset_trad/`
- `dataset/dataset_datadriven/`
- `dataset/processed/panasonic_raw_csv/`
- per-sample prediction CSV dumps
- per-sample traditional-baseline CSV dumps such as `ocv_corrected_cc_result_*`,
  `cc_result_*`, and `cc_sensitivity_result_*`
- model checkpoints: `*.pt`, `*.pth`, `*.onnx`
- thesis drafts: `*.docx`
- local reference PDFs: `*.pdf`
- `attachments_for_senior/`

## Create the Repository

```bash
git init
git add .
git status
git commit -m "Initial public release"
```

Review `git status` before committing. The supplied `.gitignore` prevents common local artifacts from being added later.

## Reproducible Pipeline

After cloning the public repository and placing the Panasonic raw data in the
layout described in `DATASET.md`, run:

```bash
python -m pip install -r requirements.txt
bash scripts/run_reproducible_pipeline.sh
```

The core pipeline rebuilds the processed CSV manifest and reruns the strict
matched 25degC, 10degC, 0degC, and 25degC-to-low-temperature transfer study.

## Current Limitation to State Publicly

The lightweight deployment experiment is a deployment-oriented proxy. It reports
parameter count, model size, estimated FP32/INT8 parameter memory, and MacBook
CPU latency. It is not a physical MCU-board validation.
