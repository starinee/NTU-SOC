# Deployment Validation

This folder contains deployment-oriented validation scripts for the lightweight SOC models.

## Scripts

- `run_mcu_oriented_lightweight_validation_25C.py`
  - Benchmarks trained 25degC sequence models on CPU.
  - Reports parameter count, checkpoint size, estimated FP32/INT8 weight memory, batch-1 latency, batch-256 latency, and compression ratio relative to the teacher model.
  - Output: `dataset/processed/deployment_validation/mcu_oriented_lightweight_validation_25C/`.

## Interpretation

The script is a reproducible MCU-oriented proxy validation, not a physical MCU deployment. It provides the minimum engineering evidence for lightweight deployment feasibility before actual embedded deployment.
