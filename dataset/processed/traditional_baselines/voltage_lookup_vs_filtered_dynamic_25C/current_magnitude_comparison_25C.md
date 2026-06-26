# 25degC Error vs Current Magnitude

| method | bin | sample_count | rmse_percent | mae_percent | p95_abs_error_percent | mean_abs_current_A | mean_dynamic_intensity_A_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Terminal-voltage lookup | Low | 17358 | 2.326 | 1.5598 | 4.7037 | 0.0849 | 0.4568 |
| Terminal-voltage lookup | Medium | 17231 | 5.3598 | 4.5279 | 9.8475 | 0.7708 | 0.5438 |
| Terminal-voltage lookup | High | 18424 | 14.9038 | 12.7854 | 27.8723 | 3.0856 | 0.8891 |
| OCV-corrected CC | Low | 17358 | 0.0814 | 0.0577 | 0.1501 | 0.0849 | 0.4568 |
| OCV-corrected CC | Medium | 17229 | 0.07 | 0.0539 | 0.136 | 0.7708 | 0.5438 |
| OCV-corrected CC | High | 18426 | 0.0561 | 0.0399 | 0.1293 | 3.0854 | 0.8891 |
| Filtered-feature MLP | Low | 16415 | 0.6008 | 0.4796 | 1.22 | 0.0863 | 0.4249 |
| Filtered-feature MLP | Medium | 16489 | 0.6218 | 0.5068 | 1.1739 | 0.7692 | 0.487 |
| Filtered-feature MLP | High | 15302 | 0.6689 | 0.524 | 1.3141 | 2.8554 | 0.7031 |
| Filtered CNN-LSTM Teacher | Low | 16290 | 0.7325 | 0.5362 | 1.4894 | 0.0862 | 0.4259 |
| Filtered CNN-LSTM Teacher | Medium | 16471 | 0.6675 | 0.4798 | 1.4541 | 0.7687 | 0.4869 |
| Filtered CNN-LSTM Teacher | High | 15268 | 0.6942 | 0.5034 | 1.5228 | 2.8542 | 0.7032 |
