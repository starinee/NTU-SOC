# Traditional Baselines

This folder contains traditional-method sensitivity experiments.

## Scripts

- `run_cc_sensitivity_current_noise_25C.py`
  - Extends Coulomb Counting sensitivity beyond capacity and initial SOC errors.
  - Adds current sensor scale bias, offset bias, Gaussian current noise, current drift, and combined nonideal cases.
  - Output: `dataset/processed/traditional_baselines/cc_current_noise_sensitivity_25C/`.

## Purpose

This addresses the concern that the reference SOC and ideal Coulomb Counting share nearly the same Ah-integration logic. The additional cases simulate practical CC nonidealities more explicitly.
