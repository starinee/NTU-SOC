| method | feature_set | source | parameter_count | checkpoint_size_MB | fp32_weight_size_MB_est | int8_weight_storage_estimate_MB | cpu_batch1_latency_ms | cpu_batch256_latency_ms | cpu_batch256_per_sample_ms | reference_teacher | params_vs_teacher_percent | parameter_reduction_vs_teacher_percent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Filtered Distilled Tiny CNN-LSTM | Raw + EMA V/I | filtered_distilled_tiny_cnn-lstm_25degC_model.pt | 3841 | 0.0195 | 0.0147 | 0.0037 | 0.7747 | 13.516 | 0.0528 | Filtered CNN-LSTM Teacher | 7.5776 | 92.4224 |
| Filtered Tiny CNN-LSTM Student | Raw + EMA V/I | filtered_tiny_cnn-lstm_student_25degC_model.pt | 3841 | 0.0195 | 0.0147 | 0.0037 | 1.0558 | 16.2074 | 0.0633 | Filtered CNN-LSTM Teacher | 7.5776 | 92.4224 |
| Filtered-feature MLP | Raw + EMA V/I | architecture_only_no_checkpoint_saved | 11649 |  | 0.0444 | 0.0111 |  |  |  |  |  |  |
| Filtered CNN-LSTM Teacher | Raw + EMA V/I | filtered_cnn-lstm_teacher_25degC_model.pt | 50689 | 0.1982 | 0.1934 | 0.0483 | 0.8917 | 86.147 | 0.3365 |  |  |  |
| Instantaneous MLP | Raw V/I/T | architecture_only_no_checkpoint_saved | 2369 |  | 0.009 | 0.0023 |  |  |  |  |  |  |
| Distilled Tiny CNN-LSTM | Raw V/I/T | distilled_tiny_cnn-lstm_25degC_model.pt | 3361 | 0.0175 | 0.0128 | 0.0032 | 0.7283 | 14.6413 | 0.0572 | CNN-LSTM Teacher | 6.8917 | 93.1083 |
| Tiny CNN-LSTM Student | Raw V/I/T | tiny_cnn-lstm_student_25degC_model.pt | 3361 | 0.0175 | 0.0128 | 0.0032 | 0.7701 | 11.7385 | 0.0459 | CNN-LSTM Teacher | 6.8917 | 93.1083 |
| LSTM | Raw V/I/T | lstm_25degC_model.pt | 5825 | 0.0258 | 0.0222 | 0.0056 | 0.663 | 8.8536 | 0.0346 |  |  |  |
| CNN-LSTM Teacher | Raw V/I/T | cnn-lstm_teacher_25degC_model.pt | 48769 | 0.1907 | 0.186 | 0.0465 | 1.0883 | 60.422 | 0.236 |  |  |  |
