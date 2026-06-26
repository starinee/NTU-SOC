# Voltage Lookup vs Filtered Dynamic Analysis

This folder compares traditional voltage-based baselines against filtered-feature
learning models on the same 25degC test profiles.

- `Terminal-voltage lookup`: directly maps loaded terminal voltage to SOC using the
  OCV-SOC curve.
- `OCV-corrected CC`: propagates SOC by Coulomb Counting and applies OCV-based
  corrections during low-current/rest-like periods.
- `Filtered-feature MLP`: lightweight tabular model with moving-average voltage,
  current, power, and current-change features.
- `Filtered CNN-LSTM Teacher`: sequence model using the filtered feature set.

The current-magnitude and dynamic-intensity bins are quantile bins over the
sample-level comparison table. Dynamic intensity is measured as a rolling mean of
absolute current slew rate.
