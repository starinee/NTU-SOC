# Results and External Comparison

## 25degC Data-Driven Results

From `dataset/processed/final_paper_tables_25degC/final_data_driven_performance_table_paper.md`:

| Method | Feature set | RMSE (%SOC) | MAE (%SOC) | Mean profile-wise maximum absolute error (%SOC) |
| --- | --- | ---: | ---: | ---: |
| Instantaneous MLP | Raw V/I/T | 1.986 | 1.520 | 18.749 |
| Filtered-feature MLP | Raw + EMA V/I | 0.597 | 0.474 | 3.862 |
| LSTM | Raw V/I/T | 0.612 | 0.482 | 3.407 |
| Filtered-feature CNN-LSTM Teacher | Raw + EMA V/I | 0.817 | 0.589 | 3.317 |
| Filtered-feature Tiny CNN-LSTM Student | Raw + EMA V/I | 0.963 | 0.739 | 4.046 |
| Filtered-feature Distilled Tiny CNN-LSTM | Raw + EMA V/I | 1.034 | 0.774 | 4.365 |

Reported metrics are the unweighted mean of profile-wise test metrics over UDDS,
LA92, and NN from the strict matched pipeline. Under the defined 25degC
profile-held-out evaluation protocol, the filtered-feature MLP achieved sub-1%
profile-averaged RMSE. The filtered tiny CNN-LSTM student also remains below
1% RMSE while using only 7.58% of the filtered teacher parameters.

## Lightweight Result

From `dataset/processed/final_paper_tables_25degC/model_complexity_and_compression_table_paper.md` and `dataset/processed/deployment_validation/mcu_oriented_lightweight_validation_25C/mcu_oriented_lightweight_validation_25C.md`:

| Model | Parameters | RMSE (%SOC) | Params vs teacher |
| --- | ---: | ---: | ---: |
| Filtered-feature CNN-LSTM Teacher | 50,689 | 0.817 | 100.00% |
| Filtered-feature Tiny CNN-LSTM Student | 3,841 | 0.963 | 7.58% |
| Filtered-feature Distilled Tiny CNN-LSTM | 3,841 | 1.034 | 7.58% |

The lightweight result is useful as a model-compression result. It is not a
physical MCU deployment result; it is a deployment-oriented proxy using model
size, estimated memory, and CPU latency. The filtered tiny models use 7.58% of
the filtered teacher parameters, corresponding to a 92.422% parameter reduction.
The CPU latency proxy values reported in the thesis are 0.892 ms for the
filtered teacher, 1.056 ms for the filtered tiny student, and 0.775 ms for the
filtered distilled tiny model.

## Temperature Transfer

From `dataset/processed/temperature_experiments/strict_matched_temperature_pipeline_25C_10C_0C/strict_matched_test_average.md`:

| Method | 25->25 RMSE | 25->10 RMSE | 25->0 RMSE |
| --- | ---: | ---: | ---: |
| Filtered-feature MLP | 0.597 | 29.027 | 42.158 |
| Filtered CNN-LSTM Teacher | 0.817 | 13.221 | 23.457 |
| Filtered Tiny CNN-LSTM Student | 0.963 | 14.924 | 22.976 |
| Filtered Distilled Tiny CNN-LSTM | 1.034 | 15.208 | 23.011 |

Direct transfer from 25degC to lower temperatures degrades heavily. The large
25->10degC and 25->0degC errors are treated as evidence that the
terminal-voltage/SOC relation and dynamics exhibit a substantial
temperature-domain shift.

Within-temperature retraining gives much lower errors:

| Method | 10->10 RMSE | 0->0 RMSE |
| --- | ---: | ---: |
| Filtered-feature MLP | 0.807 | 1.195 |
| LSTM | 1.015 | 1.097 |
| Filtered CNN-LSTM Teacher | 0.886 | 1.306 |

This supports the conclusion that temperature shift is a real domain shift and should not be hidden by only reporting same-temperature accuracy.

## Comparison with Published Results

Several published Panasonic 18650PF SOC-estimation papers report very low errors, but many use different data splits, K-fold cross-validation, or all-drive-cycle training/testing. Therefore, the comparison is useful for scale, not a strict leaderboard.

| Source | Dataset / method | Reported result | Comparison |
| --- | --- | --- | --- |
| Lima et al., "State-of-charge Estimation of a Li-ion Battery using Deep Learning and Stochastic Optimization" | Panasonic 18650PF, ten drive cycles, deep learning | Error below 1.0% in all drive cycles | Our best same-temperature 25degC models are comparable under the defined profile-held-out protocol: 0.597% RMSE for the filtered-feature MLP and sub-1% RMSE for the filtered tiny CNN-LSTM student. |
| Lima et al., "State-of-Charge Estimation of a Li-Ion Battery using Deep Forward Neural Networks" | Panasonic 18650PF, deep forward networks, K-fold workflow | Paper focuses on overfitting-resistant workflow for SOC estimation | Our filtered-feature MLP/LSTM results are in the same sub-1% range, but our split is profile-based rather than K-fold. |
| Herle et al., "Analysis of NARXNN for State of Charge Estimation for Li-ion Batteries on various Drive Cycles" | LA92, US06, UDDS, HWFET; NARXNN | MSE in the 1e-5 range | This is a strong reported result, but the setup and metric differ. Our sub-1% same-temperature results are credible, while our temperature-transfer results are intentionally harder. |

## Overall Judgment

The 25degC same-temperature results are good and publishable for a master's
thesis-level comparison: under the defined 25degC profile-held-out evaluation
protocol, the filtered-feature MLP achieved sub-1% profile-averaged RMSE. The
lightweight direction is also meaningful because the tiny student models reduce
parameters by 92.422%, with the filtered tiny student retaining sub-1%
same-temperature RMSE.

The weakest part is direct temperature transfer. That is not necessarily a failure; it is a useful result showing that 25degC-trained voltage/SOC mappings do not directly generalize to low temperature. The thesis should present this as a limitation and motivation for temperature-aware training, domain adaptation, or temperature-conditioned models.

## GitHub Readiness

The source code and documentation are ready for a GitHub-style release if the
clean release workflow is used:

```bash
bash scripts/create_github_release_tree.sh
```

Do not upload the whole local working folder directly. The local folder contains
raw datasets, regenerated CSV dumps, thesis drafts, reference PDFs, editor
metadata, and a virtual environment.

## References

- A. B. de Lima, M. B. C. Salles, and J. R. Cardoso, "State-of-charge Estimation of a Li-ion Battery using Deep Learning and Stochastic Optimization", arXiv:2011.09673. https://arxiv.org/abs/2011.09673
- A. B. de Lima, M. B. C. Salles, and J. R. Cardoso, "State-of-Charge Estimation of a Li-Ion Battery using Deep Forward Neural Networks", arXiv:2009.09543. https://arxiv.org/abs/2009.09543
- A. Herle, J. Channegowda, and K. Naraharisetti, "Analysis of NARXNN for State of Charge Estimation for Li-ion Batteries on various Drive Cycles", arXiv:2012.10725. https://arxiv.org/abs/2012.10725
- K. Movassagh, S. A. Raihan, B. Balasingam, and K. Pattipati, "A Critical Look at Coulomb Counting Towards Improving the Kalman Filter Based State of Charge Tracking Algorithms in Rechargeable Batteries", arXiv:2101.05435. https://arxiv.org/abs/2101.05435
- P. Pillai, J. Nguyen, and B. Balasingam, "Performance Analysis of Empirical Open-Circuit Voltage Modeling in Lithium Ion Batteries, Part-1", arXiv:2306.16542. https://arxiv.org/abs/2306.16542
