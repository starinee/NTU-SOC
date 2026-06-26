# Results and External Comparison

## 25degC Data-Driven Results

From `dataset/processed/final_paper_tables_25degC/final_data_driven_performance_table_paper.md`:

| Method | Feature set | RMSE (%SOC) | MAE (%SOC) | Max error (%SOC) |
| --- | --- | ---: | ---: | ---: |
| Instantaneous MLP | Raw V/I/T | 1.986 | 1.520 | 18.749 |
| Filtered-feature MLP | Raw + EMA V/I | 0.583 | 0.468 | 3.667 |
| LSTM | Raw V/I/T | 0.621 | 0.492 | 3.379 |
| Filtered-feature CNN-LSTM Teacher | Raw + EMA V/I | 0.694 | 0.501 | 3.127 |
| Filtered-feature Tiny CNN-LSTM Student | Raw + EMA V/I | 0.798 | 0.612 | 3.507 |
| Filtered-feature Distilled Tiny CNN-LSTM | Raw + EMA V/I | 0.780 | 0.542 | 4.248 |

The strongest 25degC result is the filtered-feature MLP with RMSE 0.583% SOC. The LSTM is close at 0.621% SOC. The filtered tiny CNN-LSTM student preserves sub-1% RMSE while using only 7.58% of the filtered teacher parameters.

## Lightweight Result

From `dataset/processed/final_paper_tables_25degC/model_complexity_and_compression_table_paper.md` and `dataset/processed/deployment_validation/mcu_oriented_lightweight_validation_25C/mcu_oriented_lightweight_validation_25C.md`:

| Model | Parameters | RMSE (%SOC) | Params vs teacher |
| --- | ---: | ---: | ---: |
| Filtered-feature CNN-LSTM Teacher | 50,689 | 0.694 | 100.00% |
| Filtered-feature Tiny CNN-LSTM Student | 3,841 | 0.798 | 7.58% |
| Filtered-feature Distilled Tiny CNN-LSTM | 3,841 | 0.780 | 7.58% |

The lightweight result is good as a model-compression result. It is not yet a physical MCU deployment result; it is an MCU-oriented proxy using model size, estimated memory, and CPU latency.

## Temperature Transfer

From `dataset/processed/temperature_experiments/strict_matched_temperature_pipeline_25C_10C_0C/strict_matched_test_average.md`:

| Method | 25->25 RMSE | 25->10 RMSE | 25->0 RMSE |
| --- | ---: | ---: | ---: |
| Filtered-feature MLP | 0.597 | 29.027 | 42.158 |
| LSTM | 0.630 | 28.533 | 39.002 |
| Distilled Tiny CNN-LSTM | 0.913 | 11.851 | 14.702 |
| Filtered CNN-LSTM Teacher | 0.670 | 6.876 | 15.655 |

Direct transfer from 25degC to lower temperatures degrades heavily. The filtered CNN-LSTM teacher is the strongest 25->10 model among the selected rows, while the distilled tiny CNN-LSTM is the strongest 25->0 model among the selected rows. Both still degrade substantially relative to 25degC testing.

Within-temperature retraining gives much lower errors:

| Method | 10->10 RMSE | 0->0 RMSE |
| --- | ---: | ---: |
| Filtered-feature MLP | 0.807 | 1.195 |
| LSTM | 0.933 | 1.170 |
| Filtered CNN-LSTM Teacher | 0.939 | 1.451 |

This supports the conclusion that temperature shift is a real domain shift and should not be hidden by only reporting same-temperature accuracy.

## Traditional Baselines

The ideal Coulomb Counting result is very low because the reference SOC is also Ah-integration based. This is not evidence that CC is always best in engineering. The project therefore adds nonideal CC sensitivity:

- capacity error,
- initial SOC error,
- current sensor scale bias,
- current offset,
- Gaussian current noise,
- current drift.

Main table:

- `dataset/processed/traditional_baselines/cc_current_noise_sensitivity_25C/cc_current_noise_sensitivity_25C_average_by_case.md`

## Comparison with Published Results

Several published Panasonic 18650PF SOC-estimation papers report very low errors, but many use different data splits, K-fold cross-validation, or all-drive-cycle training/testing. Therefore, the comparison is useful for scale, not a strict leaderboard.

| Source | Dataset / method | Reported result | Comparison |
| --- | --- | --- | --- |
| Lima et al., "State-of-charge Estimation of a Li-ion Battery using Deep Learning and Stochastic Optimization" | Panasonic 18650PF, ten drive cycles, deep learning | Error below 1.0% in all drive cycles | Our best same-temperature 25degC models are comparable: 0.583-0.798% RMSE. |
| Lima et al., "State-of-Charge Estimation of a Li-Ion Battery using Deep Forward Neural Networks" | Panasonic 18650PF, deep forward networks, K-fold workflow | Paper focuses on overfitting-resistant workflow for SOC estimation | Our filtered-feature MLP/LSTM results are in the same sub-1% range, but our split is profile-based rather than K-fold. |
| Herle et al., "Analysis of NARXNN for State of Charge Estimation for Li-ion Batteries on various Drive Cycles" | LA92, US06, UDDS, HWFET; NARXNN | MSE in the 1e-5 range | This is a strong reported result, but the setup and metric differ. Our sub-1% same-temperature results are credible, while our temperature-transfer results are intentionally harder. |

## Overall Judgment

The 25degC same-temperature results are good and publishable for a master's thesis-level comparison: sub-1% RMSE is consistent with reported Panasonic 18650PF literature. The lightweight direction is also meaningful because the tiny student models reduce parameters by over 92% while retaining sub-1% same-temperature RMSE.

The weakest part is direct temperature transfer. That is not necessarily a failure; it is a useful result showing that 25degC-trained voltage/SOC mappings do not directly generalize to low temperature. The thesis should present this as a limitation and motivation for temperature-aware training, domain adaptation, or temperature-conditioned models.

## GitHub Readiness

This release tree is ready to publish. It contains source code, documentation,
compact summary tables, and selected figures, while excluding raw datasets,
per-sample prediction dumps, model checkpoints, thesis drafts, and reference
PDFs.

## References

- A. B. de Lima, M. B. C. Salles, and J. R. Cardoso, "State-of-charge Estimation of a Li-ion Battery using Deep Learning and Stochastic Optimization", arXiv:2011.09673. https://arxiv.org/abs/2011.09673
- A. B. de Lima, M. B. C. Salles, and J. R. Cardoso, "State-of-Charge Estimation of a Li-Ion Battery using Deep Forward Neural Networks", arXiv:2009.09543. https://arxiv.org/abs/2009.09543
- A. Herle, J. Channegowda, and K. Naraharisetti, "Analysis of NARXNN for State of Charge Estimation for Li-ion Batteries on various Drive Cycles", arXiv:2012.10725. https://arxiv.org/abs/2012.10725
- K. Movassagh, S. A. Raihan, B. Balasingam, and K. Pattipati, "A Critical Look at Coulomb Counting Towards Improving the Kalman Filter Based State of Charge Tracking Algorithms in Rechargeable Batteries", arXiv:2101.05435. https://arxiv.org/abs/2101.05435
- P. Pillai, J. Nguyen, and B. Balasingam, "Performance Analysis of Empirical Open-Circuit Voltage Modeling in Lithium Ion Batteries, Part-1", arXiv:2306.16542. https://arxiv.org/abs/2306.16542
