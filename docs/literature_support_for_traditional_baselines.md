# Literature Support for Traditional Baselines and Temperature Transfer

This note maps the implemented baseline methods to published work so the thesis
can justify that each comparison is grounded in existing SOC-estimation practice.
The four traditional baselines below should be cited carefully: CC and CC
sensitivity are supported by Coulomb-counting error analysis; terminal-voltage
lookup is supported as a deliberately simplified voltage/OCV lookup baseline;
OCV-corrected CC is supported by papers that explicitly combine Coulomb Counting
with SOC-OCV recalibration.

## Coulomb Counting and CC Sensitivity

**Implemented code**

- `src/experiments/analysis/traditional_baselines/run_coulomb_counting_batch.py`
- `src/experiments/analysis/traditional_baselines/run_coulomb_counting_sensitivity.py`
- `src/data/traditional_baselines/run_cc_sensitivity_current_noise_25C.py`

**Verified literature support**

Movassagh et al. treat Coulomb Counting as a traditional SOC-estimation method
that is reliable only when initial SOC and capacity are known accurately. They
then formally derive error terms from current measurement, numerical integration,
battery-capacity uncertainty, and timing oscillator drift. This directly supports
both the ideal CC baseline and the non-ideal sensitivity cases in this project.

**Use in thesis**

- Ideal CC: a same-origin reference check, not proof that CC is always best in a
  real BMS.
- CC sensitivity: initial SOC error, capacity mismatch, current scale/offset,
  current noise, and drift simulate practical non-idealities.

**Citation**

- Movassagh, K., Raihan, S. A., Balasingam, B., & Pattipati, K. (2021). *A
  Critical Look at Coulomb Counting Towards Improving the Kalman Filter Based
  State of Charge Tracking Algorithms in Rechargeable Batteries*. arXiv:2101.05435.
  https://doi.org/10.48550/arXiv.2101.05435

## Terminal-Voltage Lookup and the OCV-SOC Curve

**Implemented code**

- `src/experiments/analysis/ocv/build_ocv_soc_lookup_25degC.py`
- `src/experiments/analysis/traditional_baselines/run_terminal_voltage_lookup_25degC.py`
- `src/experiments/analysis/traditional_baselines/compare_voltage_lookup_vs_filtered_dynamic_25degC.py`

**Key output**

- `dataset/processed/ocv_lookup/ocv_soc_curve_25degC.png`
- `dataset/processed/ocv_lookup/ocv_soc_curve_25degC_reversed_x.png`

**Verified literature support**

Pillai, Nguyen, and Balasingam describe the OCV-SOC characteristic as a central
battery-management object and analyse how OCV model uncertainty affects SOC and
capacity estimation. The terminal-voltage lookup baseline in this project should
therefore be described precisely: it is not claiming loaded terminal voltage is
true OCV. It deliberately passes loaded terminal voltage through the 25 °C
OCV-SOC lookup to test how badly a simple voltage-only method degrades under
current load and dynamic drive profiles.

**Citation**

- Pillai, P., Nguyen, J., & Balasingam, B. (2023). *Performance Analysis of
  Empirical Open-Circuit Voltage Modeling in Lithium Ion Batteries, Part-1:
  Performance Measures*. arXiv:2306.16542.
  https://doi.org/10.48550/arXiv.2306.16542

## OCV-Corrected Coulomb Counting

**Implemented code**

- `src/experiments/analysis/traditional_baselines/run_ocv_corrected_cc_25degC.py`

**Verified literature support**

Baccouche et al. directly support the hybrid idea used here: an improved
Coulomb-counting method uses a piecewise SOC-OCV relationship for periodic
recalibration, and the solution is validated on a PIC18F MCU-family hardware
platform. This is stronger support than merely citing separate CC and OCV papers.
The thesis should state that our implementation follows the same engineering
principle, not that it reproduces their embedded implementation exactly.

**Citation**

- Baccouche, I., Jemmali, S., Mlayah, A., Manai, B., & Ben Amara, N. E. (2018).
  *Implementation of an Improved Coulomb-Counting Algorithm Based on a Piecewise
  SOC-OCV Relationship for SOC Estimation of Li-Ion Battery*. arXiv:1803.10654.
  https://doi.org/10.48550/arXiv.1803.10654

## Temperature Transfer

**Implemented code**

- `src/data/temperature_experiments/run_strict_matched_temperature_pipeline_25C_10C_0C.py`
- `src/experiments/temperature_transfer/run_25degC_to_10degC_0degC_transfer.py`

**Why it is literature-supported**

Temperature transfer asks whether a model trained at one ambient temperature can
generalize to another without retraining. This is a known weakness of fixed-
temperature SOC models because voltage response, internal resistance, and usable
capacity are temperature-dependent. In this project, the strict matched study
uses the same profiles, downsampling, feature construction, sequence length, and
model list for 25 °C, 10 °C, and 0 °C; only the temperature line changes.

**Suggested citation**

- Qin, Y., Adams, S., & Yuen, C. (2021). *A Transfer Learning-based State of
  Charge Estimation for Lithium-Ion Battery at Varying Ambient Temperatures*.
  https://arxiv.org/abs/2101.03704

## Panasonic 18650PF Dataset and Profile-Wise Reporting

**Implemented code**

- `src/data_processing/convert_panasonic_mat_to_csv.py`
- `src/data/temperature_experiments/generate_profile_error_analysis_10C_0C.py`

**Why it is literature-supported**

The Panasonic 18650PF drive-cycle dataset and profile-wise tests such as UDDS,
LA92, US06, and NN are widely used in data-driven SOC-estimation studies. The
profile-wise RMSE output in this project avoids hiding difficult drive cycles
inside a single averaged test metric.

**Suggested citations**

- Lima, M., et al. (2020). *Deep Neural Network for Lithium-ion Battery State of
  Charge Estimation*. https://arxiv.org/abs/2009.09543
- Herle, S., et al. (2020). *NARX based State of Charge estimation for Li-ion
  Batteries using Dynamic Drive Cycle Data*. https://arxiv.org/abs/2012.10725
