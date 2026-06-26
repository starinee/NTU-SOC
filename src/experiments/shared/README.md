# Shared temperature-pipeline contract

This folder defines the controlled experimental contract used by the 25degC,
10degC, 0degC, and temperature-transfer scripts.

Only temperature is allowed to change across the three within-temperature
lines. The split, downsampling, SOC reference, feature definitions, model list,
and model hyperparameters must stay matched to the original 25degC pipeline.

Controlled constants are in `temperature_pipeline_contract.py`.
