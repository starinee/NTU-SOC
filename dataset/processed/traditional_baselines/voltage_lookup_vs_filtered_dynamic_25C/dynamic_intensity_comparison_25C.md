# 25degC Error vs Dynamic Intensity

| method | bin | sample_count | rmse_percent | mae_percent | p95_abs_error_percent | mean_abs_current_A | mean_dynamic_intensity_A_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Terminal-voltage lookup | Low | 17020 | 4.9506 | 3.4949 | 10.253 | 0.699 | 0.1995 |
| Terminal-voltage lookup | Medium | 16892 | 6.5019 | 4.7904 | 13.1028 | 0.9862 | 0.4241 |
| Terminal-voltage lookup | High | 19101 | 13.633 | 10.4839 | 27.3592 | 2.2538 | 1.2106 |
| OCV-corrected CC | Low | 17020 | 0.0849 | 0.062 | 0.1503 | 0.699 | 0.1995 |
| OCV-corrected CC | Medium | 16892 | 0.0782 | 0.0624 | 0.1385 | 0.9862 | 0.4241 |
| OCV-corrected CC | High | 19101 | 0.0406 | 0.0292 | 0.0775 | 2.2538 | 1.2106 |
| Filtered-feature MLP | Low | 16733 | 0.6223 | 0.5115 | 1.1948 | 0.7109 | 0.2028 |
| Filtered-feature MLP | Medium | 16841 | 0.6665 | 0.5403 | 1.2746 | 0.9803 | 0.4238 |
| Filtered-feature MLP | High | 14632 | 0.5954 | 0.4504 | 1.2276 | 2.0086 | 1.0411 |
| Filtered CNN-LSTM Teacher | Low | 16648 | 0.7588 | 0.5547 | 1.6319 | 0.7135 | 0.2034 |
| Filtered CNN-LSTM Teacher | Medium | 16795 | 0.6782 | 0.4871 | 1.4593 | 0.9807 | 0.4239 |
| Filtered CNN-LSTM Teacher | High | 14586 | 0.6481 | 0.4737 | 1.3301 | 2.0085 | 1.0413 |
