# Open-Source Readiness Checklist

## Ready

- Clear top-level `README.md`
- Dependency file: `requirements.txt`
- Dataset documentation: `DATASET.md`
- Code map: `CODE_STRUCTURE.md`
- Result summary and literature comparison: `RESULTS.md`
- Reproducible runner: `scripts/run_reproducible_pipeline.sh`
- Clean public release tree with no raw data, checkpoints, or local document artifacts.
- `.gitignore` excludes raw data, local Word files, checkpoints, caches, and large prediction dumps.
- Temperature experiments are grouped under `src/data/temperature_experiments/`.
- Deployment validation is grouped under `src/data/deployment_validation/`.
- Traditional baseline extensions are grouped under `src/data/traditional_baselines/`.

## Not Yet Fully Ready

- No physical MCU-board validation yet.
- Raw dataset is local-only; `DATASET.md` records the source DOI and instructs users to obtain it independently.
- The repository contains thesis PDFs and Word drafts locally. They are ignored by `.gitignore`, but should be excluded from any public GitHub upload.

## Recommended Before Public Upload

1. Create a fresh Git repository or clean branch.
2. Add only source code, summary tables, small figures, and documentation.
3. Do not add raw MAT/CSV files or model checkpoints unless explicitly allowed.
4. Run:

```bash
python -m py_compile src/data/*.py src/data/*/*.py
bash scripts/run_reproducible_pipeline.sh
```

5. Check `git status` before committing.

## Current Verdict

The release tree is ready for a GitHub-style repository after the raw dataset is obtained separately. The repository does not include raw data, model checkpoints, or thesis drafts.
