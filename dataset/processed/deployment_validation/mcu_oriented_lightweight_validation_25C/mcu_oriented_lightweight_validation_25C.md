| method | feature_set | source | parameter_count | checkpoint_size_MB | fp32_weight_size_MB_est | int8_weight_size_MB_est | cpu_batch1_latency_ms | cpu_batch256_latency_ms | cpu_batch256_per_sample_ms | reference_teacher | params_vs_teacher_percent | parameter_reduction_vs_teacher_percent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Filtered Distilled Tiny CNN-LSTM | Raw + EMA V/I | cnn_lstm_distilled_student_filtered_features_25degC_model.pt | 3841 | 0.0197 | 0.0147 | 0.0037 | 0.5185 | 10.1371 | 0.0396 | Filtered CNN-LSTM Teacher | 7.5776 | 92.4224 |
| Filtered Tiny CNN-LSTM Student | Raw + EMA V/I | cnn_lstm_student_filtered_features_25degC_model.pt | 3841 | 0.0194 | 0.0147 | 0.0037 | 0.5253 | 10.1232 | 0.0395 | Filtered CNN-LSTM Teacher | 7.5776 | 92.4224 |
| Filtered-feature MLP | Raw + EMA V/I | architecture_only_no_checkpoint_saved | 11649 |  | 0.0444 | 0.0111 |  |  |  |  |  |  |
| Filtered CNN-LSTM Teacher | Raw + EMA V/I | cnn_lstm_teacher_filtered_features_25degC_model.pt | 50689 | 0.1982 | 0.1934 | 0.0483 | 0.608 | 42.6509 | 0.1666 |  |  |  |
| Instantaneous MLP | Raw V/I/T | architecture_only_no_checkpoint_saved | 2369 |  | 0.009 | 0.0023 |  |  |  |  |  |  |
| Distilled Tiny CNN-LSTM | Raw V/I/T | cnn_lstm_distilled_student_25degC_model.pt | 3361 | 0.0176 | 0.0128 | 0.0032 | 0.518 | 9.6316 | 0.0376 | CNN-LSTM Teacher | 6.8917 | 93.1083 |
| Tiny CNN-LSTM Student | Raw V/I/T | cnn_lstm_student_25degC_model.pt | 3361 | 0.0174 | 0.0128 | 0.0032 | 0.5173 | 9.596 | 0.0375 | CNN-LSTM Teacher | 6.8917 | 93.1083 |
| LSTM | Raw V/I/T | lstm_baseline_25degC_model.pt | 5825 | 0.0256 | 0.0222 | 0.0056 | 0.4753 | 8.1595 | 0.0319 |  |  |  |
| CNN-LSTM Teacher | Raw V/I/T | cnn_lstm_teacher_25degC_model.pt | 48769 | 0.1907 | 0.186 | 0.0465 | 0.6013 | 40.3681 | 0.1577 |  |  |  |
