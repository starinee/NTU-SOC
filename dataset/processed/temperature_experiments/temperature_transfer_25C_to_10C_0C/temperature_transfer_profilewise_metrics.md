| train_temperature_C | test_temperature_C | split | method | file_name | cycle_name | sample_count | parameter_count | MAE_percent | RMSE_percent | MAX_ERROR_percent | P95_ABS_ERROR_percent | P99_ABS_ERROR_percent | FINAL_ERROR_percent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| none | 0 | test | Coulomb Counting | 06-01-17_10.36 0degC_LA92_Pan18650PF.mat | LA92 | 8267 |  | 0.0977 | 0.1157 | 0.2444 | 0.2011 | 0.2169 | -0.1994 |
| none | 0 | test | Coulomb Counting | 06-01-17_10.36 0degC_NN_Pan18650PF.mat | NN | 6328 |  | 0.0975 | 0.1105 | 0.2537 | 0.1717 | 0.2048 | 0.0675 |
| none | 0 | test | Coulomb Counting | 06-02-17_17.14 0degC_UDDS_Pan18650PF.mat | UDDS | 12853 |  | 0.0292 | 0.0345 | 0.0973 | 0.059 | 0.0819 | -0.0819 |
| none | 10 | test | Coulomb Counting | 03-27-17_09.06 10degC_LA92_Pan18650PF.mat | LA92 | 12596 |  | 0.1189 | 0.1264 | 0.2196 | 0.1736 | 0.1895 | -0.0987 |
| none | 10 | test | Coulomb Counting | 03-27-17_09.06 10degC_NN_Pan18650PF.mat | NN | 10511 |  | 0.125 | 0.136 | 0.2802 | 0.2086 | 0.2291 | -0.2291 |
| none | 10 | test | Coulomb Counting | 03-27-17_09.06 10degC_UDDS_Pan18650PF.mat | UDDS | 21047 |  | 0.072 | 0.0771 | 0.1517 | 0.1107 | 0.1204 | -0.0968 |
| none | 25 | test | Coulomb Counting | 03-21-17_00.29 25degC_UDDS_Pan18650PF.mat | UDDS | 22419 |  | 0.1112 | 0.1226 | 0.2393 | 0.2076 | 0.214 | 0.1218 |
| none | 25 | test | Coulomb Counting | 03-21-17_09.38 25degC_LA92_Pan18650PF.mat | LA92 | 14088 |  | 0.1177 | 0.1285 | 0.2339 | 0.1853 | 0.2 | -0.1663 |
| none | 25 | test | Coulomb Counting | 03-21-17_16.27 25degC_NN_Pan18650PF.mat | NN | 11699 |  | 0.0414 | 0.049 | 0.1464 | 0.0895 | 0.1261 | -0.1261 |
| 25 | 25 | test | Instantaneous MLP | 03-21-17_00.29 25degC_UDDS_Pan18650PF.mat | UDDS | 22419 |  | 1.3107 | 1.6465 | 13.8947 | 3.0842 | 4.0785 | 4.0785 |
| 25 | 25 | test | Instantaneous MLP | 03-21-17_09.38 25degC_LA92_Pan18650PF.mat | LA92 | 14088 |  | 1.4027 | 1.8858 | 16.7613 | 3.6943 | 5.9012 | 1.4972 |
| 25 | 25 | test | Instantaneous MLP | 03-21-17_16.27 25degC_NN_Pan18650PF.mat | NN | 11699 |  | 1.8466 | 2.4267 | 25.5912 | 4.5601 | 7.6376 | 0.2793 |
| 25 | 25 | test | Filtered-feature MLP | 03-21-17_00.29 25degC_UDDS_Pan18650PF.mat | UDDS | 22419 |  | 0.6263 | 0.7389 | 4.1486 | 1.3156 | 1.6289 | 1.6287 |
| 25 | 25 | test | Filtered-feature MLP | 03-21-17_09.38 25degC_LA92_Pan18650PF.mat | LA92 | 14088 |  | 0.359 | 0.4719 | 3.1469 | 0.9528 | 1.379 | -0.6577 |
| 25 | 25 | test | Filtered-feature MLP | 03-21-17_16.27 25degC_NN_Pan18650PF.mat | NN | 11699 |  | 0.4361 | 0.5794 | 4.2901 | 1.1536 | 1.7701 | -0.919 |
| 25 | 25 | test | LSTM | 03-21-17_00.29 25degC_UDDS_Pan18650PF.mat | UDDS | 22360 | 5825 | 0.8423 | 0.9916 | 3.0327 | 1.8205 | 2.353 | 2.5556 |
| 25 | 25 | test | LSTM | 03-21-17_09.38 25degC_LA92_Pan18650PF.mat | LA92 | 14029 | 5825 | 0.5565 | 0.7335 | 3.4392 | 1.5155 | 2.2073 | 0.8882 |
| 25 | 25 | test | LSTM | 03-21-17_16.27 25degC_NN_Pan18650PF.mat | NN | 11640 | 5825 | 0.7007 | 0.8857 | 3.863 | 1.7791 | 2.3433 | 0.238 |
| 25 | 25 | test | Filtered CNN-LSTM Teacher | 03-21-17_00.29 25degC_UDDS_Pan18650PF.mat | UDDS | 22360 | 50689 | 0.9088 | 1.0051 | 2.5364 | 1.7329 | 2.0253 | 1.8379 |
| 25 | 25 | test | Filtered CNN-LSTM Teacher | 03-21-17_09.38 25degC_LA92_Pan18650PF.mat | LA92 | 14029 | 50689 | 0.4333 | 0.552 | 2.5837 | 1.0534 | 1.6854 | -0.1417 |
| 25 | 25 | test | Filtered CNN-LSTM Teacher | 03-21-17_16.27 25degC_NN_Pan18650PF.mat | NN | 11640 | 50689 | 0.4627 | 0.5821 | 2.2971 | 1.1292 | 1.5459 | -0.227 |
| 25 | 25 | test | Filtered Tiny CNN-LSTM Student | 03-21-17_00.29 25degC_UDDS_Pan18650PF.mat | UDDS | 22360 | 3841 | 0.7641 | 1.1618 | 5.0825 | 2.7772 | 4.2712 | 4.3563 |
| 25 | 25 | test | Filtered Tiny CNN-LSTM Student | 03-21-17_09.38 25degC_LA92_Pan18650PF.mat | LA92 | 14029 | 3841 | 0.862 | 1.2106 | 6.1305 | 2.4356 | 4.3703 | 1.5295 |
| 25 | 25 | test | Filtered Tiny CNN-LSTM Student | 03-21-17_16.27 25degC_NN_Pan18650PF.mat | NN | 11640 | 3841 | 1.1031 | 1.4842 | 7.9063 | 2.9929 | 4.3695 | -0.2199 |
| 25 | 25 | test | Filtered Distilled Tiny CNN-LSTM | 03-21-17_00.29 25degC_UDDS_Pan18650PF.mat | UDDS | 22360 | 3841 | 0.9479 | 1.2335 | 5.018 | 2.5256 | 3.68 | 1.3244 |
| 25 | 25 | test | Filtered Distilled Tiny CNN-LSTM | 03-21-17_09.38 25degC_LA92_Pan18650PF.mat | LA92 | 14029 | 3841 | 0.7397 | 1.0758 | 5.181 | 2.0901 | 4.1693 | -0.6259 |
| 25 | 25 | test | Filtered Distilled Tiny CNN-LSTM | 03-21-17_16.27 25degC_NN_Pan18650PF.mat | NN | 11640 | 3841 | 0.8201 | 1.1029 | 4.9636 | 2.2519 | 3.6045 | -0.059 |
| 25 | 10 | test | Instantaneous MLP | 03-27-17_09.06 10degC_LA92_Pan18650PF.mat | LA92 | 12596 |  | 16.5178 | 18.7165 | 46.1348 | 31.3301 | 34.3139 | 34.9217 |
| 25 | 10 | test | Instantaneous MLP | 03-27-17_09.06 10degC_NN_Pan18650PF.mat | NN | 10511 |  | 14.4388 | 16.4137 | 40.1504 | 28.3752 | 32.1073 | 33.567 |
| 25 | 10 | test | Instantaneous MLP | 03-27-17_09.06 10degC_UDDS_Pan18650PF.mat | UDDS | 21047 |  | 21.1365 | 23.4737 | 46.4683 | 37.148 | 40.9776 | 41.8399 |
| 25 | 10 | test | Filtered-feature MLP | 03-27-17_09.06 10degC_LA92_Pan18650PF.mat | LA92 | 12596 |  | 26.5275 | 28.5655 | 46.2347 | 38.8312 | 40.7192 | 41.2177 |
| 25 | 10 | test | Filtered-feature MLP | 03-27-17_09.06 10degC_NN_Pan18650PF.mat | NN | 10511 |  | 23.7708 | 25.4264 | 42.5154 | 35.8972 | 38.6474 | 39.7175 |
| 25 | 10 | test | Filtered-feature MLP | 03-27-17_09.06 10degC_UDDS_Pan18650PF.mat | UDDS | 21047 |  | 30.5178 | 33.0897 | 49.1998 | 45.0345 | 46.7071 | 46.4509 |
| 25 | 10 | test | LSTM | 03-27-17_09.06 10degC_LA92_Pan18650PF.mat | LA92 | 12537 | 5825 | 20.9296 | 24.9874 | 53.8083 | 43.1901 | 49.7848 | 9.5338 |
| 25 | 10 | test | LSTM | 03-27-17_09.06 10degC_NN_Pan18650PF.mat | NN | 10452 | 5825 | 19.8741 | 23.7033 | 56.2098 | 41.6636 | 47.9674 | 9.2994 |
| 25 | 10 | test | LSTM | 03-27-17_09.06 10degC_UDDS_Pan18650PF.mat | UDDS | 20988 | 5825 | 20.717 | 24.9413 | 54.0442 | 44.8039 | 48.3794 | 15.1127 |
| 25 | 10 | test | Filtered CNN-LSTM Teacher | 03-27-17_09.06 10degC_LA92_Pan18650PF.mat | LA92 | 12537 | 50689 | 3.9931 | 4.5937 | 10.5267 | 8.3865 | 9.7076 | 10.3597 |
| 25 | 10 | test | Filtered CNN-LSTM Teacher | 03-27-17_09.06 10degC_NN_Pan18650PF.mat | NN | 10452 | 50689 | 3.888 | 4.5569 | 12.1786 | 8.4066 | 9.8058 | 10.3582 |
| 25 | 10 | test | Filtered CNN-LSTM Teacher | 03-27-17_09.06 10degC_UDDS_Pan18650PF.mat | UDDS | 20988 | 50689 | 3.6753 | 4.8725 | 12.9439 | 10.3054 | 11.9504 | 12.5457 |
| 25 | 10 | test | Filtered Tiny CNN-LSTM Student | 03-27-17_09.06 10degC_LA92_Pan18650PF.mat | LA92 | 12537 | 3841 | 8.9782 | 11.4587 | 24.6166 | 21.4883 | 23.6913 | 24.6052 |
| 25 | 10 | test | Filtered Tiny CNN-LSTM Student | 03-27-17_09.06 10degC_NN_Pan18650PF.mat | NN | 10452 | 3841 | 7.9803 | 10.2568 | 23.9348 | 19.8184 | 22.2869 | 23.9348 |
| 25 | 10 | test | Filtered Tiny CNN-LSTM Student | 03-27-17_09.06 10degC_UDDS_Pan18650PF.mat | UDDS | 20988 | 3841 | 10.6593 | 13.2339 | 27.5819 | 25.303 | 26.7285 | 25.2135 |
| 25 | 10 | test | Filtered Distilled Tiny CNN-LSTM | 03-27-17_09.06 10degC_LA92_Pan18650PF.mat | LA92 | 12537 | 3841 | 9.5133 | 11.7354 | 25.5426 | 21.997 | 23.9144 | 25.5204 |
| 25 | 10 | test | Filtered Distilled Tiny CNN-LSTM | 03-27-17_09.06 10degC_NN_Pan18650PF.mat | NN | 10452 | 3841 | 9.1244 | 11.2405 | 25.3544 | 21.0473 | 23.7342 | 25.3544 |
| 25 | 10 | test | Filtered Distilled Tiny CNN-LSTM | 03-27-17_09.06 10degC_UDDS_Pan18650PF.mat | UDDS | 20988 | 3841 | 11.2236 | 13.5731 | 27.6072 | 25.3137 | 26.8909 | 15.5484 |
| 25 | 0 | test | Instantaneous MLP | 06-01-17_10.36 0degC_LA92_Pan18650PF.mat | LA92 | 8267 |  | 32.0057 | 35.7754 | 63.198 | 54.5342 | 61.7933 | 63.198 |
| 25 | 0 | test | Instantaneous MLP | 06-01-17_10.36 0degC_NN_Pan18650PF.mat | NN | 6328 |  | 28.3161 | 31.2661 | 59.3563 | 49.3681 | 57.3839 | 59.3229 |
| 25 | 0 | test | Instantaneous MLP | 06-02-17_17.14 0degC_UDDS_Pan18650PF.mat | UDDS | 12853 |  | 35.1294 | 39.4636 | 64.2067 | 58.9986 | 62.4987 | 64.1734 |
| 25 | 0 | test | Filtered-feature MLP | 06-01-17_10.36 0degC_LA92_Pan18650PF.mat | LA92 | 8267 |  | 37.0904 | 42.5135 | 79.9061 | 69.1078 | 77.763 | 79.9061 |
| 25 | 0 | test | Filtered-feature MLP | 06-01-17_10.36 0degC_NN_Pan18650PF.mat | NN | 6328 |  | 34.4298 | 38.7118 | 74.5446 | 61.7526 | 71.608 | 74.5031 |
| 25 | 0 | test | Filtered-feature MLP | 06-02-17_17.14 0degC_UDDS_Pan18650PF.mat | UDDS | 12853 |  | 39.2434 | 45.2494 | 80.0034 | 73.4365 | 78.42 | 80.0034 |
| 25 | 0 | test | LSTM | 06-01-17_10.36 0degC_LA92_Pan18650PF.mat | LA92 | 8208 | 5825 | 26.7619 | 32.2367 | 62.1319 | 57.1969 | 59.6112 | 7.2785 |
| 25 | 0 | test | LSTM | 06-01-17_10.36 0degC_NN_Pan18650PF.mat | NN | 6269 | 5825 | 26.1317 | 31.357 | 63.5942 | 54.033 | 59.095 | 7.2735 |
| 25 | 0 | test | LSTM | 06-02-17_17.14 0degC_UDDS_Pan18650PF.mat | UDDS | 12794 | 5825 | 26.1223 | 31.6876 | 61.8929 | 55.144 | 59.2151 | 7.4462 |
| 25 | 0 | test | Filtered CNN-LSTM Teacher | 06-01-17_10.36 0degC_LA92_Pan18650PF.mat | LA92 | 8208 | 50689 | 6.536 | 7.6104 | 19.4025 | 13.7055 | 16.6401 | 13.4313 |
| 25 | 0 | test | Filtered CNN-LSTM Teacher | 06-01-17_10.36 0degC_NN_Pan18650PF.mat | NN | 6269 | 50689 | 7.7412 | 8.995 | 25.7251 | 16.4748 | 19.7983 | 12.4755 |
| 25 | 0 | test | Filtered CNN-LSTM Teacher | 06-02-17_17.14 0degC_UDDS_Pan18650PF.mat | UDDS | 12794 | 50689 | 4.645 | 5.5274 | 16.5402 | 10.2833 | 13.4553 | 13.8966 |
| 25 | 0 | test | Filtered Tiny CNN-LSTM Student | 06-01-17_10.36 0degC_LA92_Pan18650PF.mat | LA92 | 8208 | 3841 | 16.0063 | 18.577 | 36.1827 | 32.2899 | 35.7813 | 31.6126 |
| 25 | 0 | test | Filtered Tiny CNN-LSTM Student | 06-01-17_10.36 0degC_NN_Pan18650PF.mat | NN | 6269 | 3841 | 14.3586 | 16.8815 | 35.7433 | 30.0298 | 33.6882 | 31.243 |
| 25 | 0 | test | Filtered Tiny CNN-LSTM Student | 06-02-17_17.14 0degC_UDDS_Pan18650PF.mat | UDDS | 12794 | 3841 | 16.6509 | 19.1214 | 35.9252 | 31.0828 | 35.0541 | 31.8119 |
| 25 | 0 | test | Filtered Distilled Tiny CNN-LSTM | 06-01-17_10.36 0degC_LA92_Pan18650PF.mat | LA92 | 8208 | 3841 | 12.9729 | 15.9491 | 36.2878 | 29.6019 | 35.4476 | 36.2878 |
| 25 | 0 | test | Filtered Distilled Tiny CNN-LSTM | 06-01-17_10.36 0degC_NN_Pan18650PF.mat | NN | 6269 | 3841 | 11.6664 | 14.1258 | 34.3703 | 25.8523 | 33.3755 | 34.3703 |
| 25 | 0 | test | Filtered Distilled Tiny CNN-LSTM | 06-02-17_17.14 0degC_UDDS_Pan18650PF.mat | UDDS | 12794 | 3841 | 14.2601 | 17.8414 | 36.8472 | 32.6208 | 35.7294 | 36.8472 |
| 10 | 10 | test | Coulomb Counting | 03-27-17_09.06 10degC_LA92_Pan18650PF.mat | LA92 | 12596 |  | 0.1189 | 0.1264 | 0.2196 | 0.1736 | 0.1895 | -0.0987 |
| 10 | 10 | test | Coulomb Counting | 03-27-17_09.06 10degC_NN_Pan18650PF.mat | NN | 10511 |  | 0.125 | 0.136 | 0.2802 | 0.2086 | 0.2291 | -0.2291 |
| 10 | 10 | test | Coulomb Counting | 03-27-17_09.06 10degC_UDDS_Pan18650PF.mat | UDDS | 21047 |  | 0.072 | 0.0771 | 0.1517 | 0.1107 | 0.1204 | -0.0968 |
| 10 | 10 | test | Instantaneous MLP | 03-27-17_09.06 10degC_LA92_Pan18650PF.mat | LA92 | 12596 |  | 1.9387 | 2.7808 | 34.8631 | 5.6507 | 9.8271 | 3.0279 |
| 10 | 10 | test | Instantaneous MLP | 03-27-17_09.06 10degC_NN_Pan18650PF.mat | NN | 10511 |  | 2.3173 | 3.0308 | 33.6941 | 5.7838 | 8.839 | 1.8412 |
| 10 | 10 | test | Instantaneous MLP | 03-27-17_09.06 10degC_UDDS_Pan18650PF.mat | UDDS | 21047 |  | 1.9241 | 2.5606 | 16.2153 | 5.2007 | 7.4918 | 7.0434 |
| 10 | 10 | test | Filtered-feature MLP | 03-27-17_09.06 10degC_LA92_Pan18650PF.mat | LA92 | 12596 |  | 0.5055 | 0.6923 | 4.9023 | 1.4907 | 2.2201 | 0.2245 |
| 10 | 10 | test | Filtered-feature MLP | 03-27-17_09.06 10degC_NN_Pan18650PF.mat | NN | 10511 |  | 0.4723 | 0.6316 | 3.5514 | 1.3355 | 1.9394 | 0.465 |
| 10 | 10 | test | Filtered-feature MLP | 03-27-17_09.06 10degC_UDDS_Pan18650PF.mat | UDDS | 21047 |  | 0.92 | 1.0984 | 4.3011 | 1.9354 | 2.4962 | 3.18 |
| 10 | 10 | test | LSTM | 03-27-17_09.06 10degC_LA92_Pan18650PF.mat | LA92 | 12537 | 5825 | 0.7282 | 0.956 | 5.4748 | 1.9702 | 2.7136 | -0.413 |
| 10 | 10 | test | LSTM | 03-27-17_09.06 10degC_NN_Pan18650PF.mat | NN | 10452 | 5825 | 0.7207 | 0.9426 | 3.9852 | 1.959 | 2.7493 | 0.0924 |
| 10 | 10 | test | LSTM | 03-27-17_09.06 10degC_UDDS_Pan18650PF.mat | UDDS | 20988 | 5825 | 0.8738 | 1.0472 | 4.1267 | 1.8786 | 2.4276 | 1.0484 |
| 10 | 10 | test | Filtered CNN-LSTM Teacher | 03-27-17_09.06 10degC_LA92_Pan18650PF.mat | LA92 | 12537 | 50689 | 0.6911 | 0.8153 | 2.862 | 1.5075 | 2.043 | 1.2236 |
| 10 | 10 | test | Filtered CNN-LSTM Teacher | 03-27-17_09.06 10degC_NN_Pan18650PF.mat | NN | 10452 | 50689 | 0.7276 | 0.9501 | 3.361 | 1.8709 | 2.5707 | 1.1494 |
| 10 | 10 | test | Filtered CNN-LSTM Teacher | 03-27-17_09.06 10degC_UDDS_Pan18650PF.mat | UDDS | 20988 | 50689 | 1.2044 | 1.5993 | 6.9716 | 3.2345 | 4.1843 | 6.9716 |
| 10 | 10 | test | Filtered Tiny CNN-LSTM Student | 03-27-17_09.06 10degC_LA92_Pan18650PF.mat | LA92 | 12537 | 3841 | 0.8406 | 1.135 | 5.0073 | 2.219 | 4.1995 | -0.9288 |
| 10 | 10 | test | Filtered Tiny CNN-LSTM Student | 03-27-17_09.06 10degC_NN_Pan18650PF.mat | NN | 10452 | 3841 | 0.6135 | 0.8513 | 4.5649 | 1.7785 | 2.8014 | -0.5327 |
| 10 | 10 | test | Filtered Tiny CNN-LSTM Student | 03-27-17_09.06 10degC_UDDS_Pan18650PF.mat | UDDS | 20988 | 3841 | 0.9458 | 1.32 | 4.8989 | 2.8322 | 4.6413 | 4.6475 |
| 10 | 10 | test | Filtered Distilled Tiny CNN-LSTM | 03-27-17_09.06 10degC_LA92_Pan18650PF.mat | LA92 | 12537 | 3841 | 1.5963 | 2.1061 | 9.1116 | 3.7365 | 7.5297 | 4.0204 |
| 10 | 10 | test | Filtered Distilled Tiny CNN-LSTM | 03-27-17_09.06 10degC_NN_Pan18650PF.mat | NN | 10452 | 3841 | 1.5482 | 2.0423 | 8.1255 | 4.2193 | 6.2069 | 2.4186 |
| 10 | 10 | test | Filtered Distilled Tiny CNN-LSTM | 03-27-17_09.06 10degC_UDDS_Pan18650PF.mat | UDDS | 20988 | 3841 | 2.1309 | 2.9522 | 12.608 | 6.2389 | 8.3519 | 12.608 |
| 0 | 0 | test | Coulomb Counting | 06-01-17_10.36 0degC_LA92_Pan18650PF.mat | LA92 | 8267 |  | 0.0977 | 0.1157 | 0.2444 | 0.2011 | 0.2169 | -0.1994 |
| 0 | 0 | test | Coulomb Counting | 06-01-17_10.36 0degC_NN_Pan18650PF.mat | NN | 6328 |  | 0.0975 | 0.1105 | 0.2537 | 0.1717 | 0.2048 | 0.0675 |
| 0 | 0 | test | Coulomb Counting | 06-02-17_17.14 0degC_UDDS_Pan18650PF.mat | UDDS | 12853 |  | 0.0292 | 0.0345 | 0.0973 | 0.059 | 0.0819 | -0.0819 |
| 0 | 0 | test | Instantaneous MLP | 06-01-17_10.36 0degC_LA92_Pan18650PF.mat | LA92 | 8267 |  | 2.0224 | 2.6709 | 12.2458 | 5.3454 | 7.2697 | 4.3107 |
| 0 | 0 | test | Instantaneous MLP | 06-01-17_10.36 0degC_NN_Pan18650PF.mat | NN | 6328 |  | 2.9943 | 3.9385 | 39.1529 | 7.8957 | 12.2103 | 5.4474 |
| 0 | 0 | test | Instantaneous MLP | 06-02-17_17.14 0degC_UDDS_Pan18650PF.mat | UDDS | 12853 |  | 1.5668 | 2.1588 | 22.7045 | 4.4849 | 5.9929 | 4.5686 |
| 0 | 0 | test | Filtered-feature MLP | 06-01-17_10.36 0degC_LA92_Pan18650PF.mat | LA92 | 8267 |  | 0.856 | 1.0231 | 6.2075 | 1.9057 | 2.8638 | 3.0289 |
| 0 | 0 | test | Filtered-feature MLP | 06-01-17_10.36 0degC_NN_Pan18650PF.mat | NN | 6328 |  | 1.3587 | 1.5908 | 6.0948 | 2.8796 | 3.5411 | 1.1543 |
| 0 | 0 | test | Filtered-feature MLP | 06-02-17_17.14 0degC_UDDS_Pan18650PF.mat | UDDS | 12853 |  | 0.7388 | 0.9699 | 5.3252 | 1.8906 | 3 | 3.5398 |
| 0 | 0 | test | LSTM | 06-01-17_10.36 0degC_LA92_Pan18650PF.mat | LA92 | 8208 | 5825 | 1.1053 | 1.3088 | 3.3804 | 2.3748 | 2.8694 | 2.272 |
| 0 | 0 | test | LSTM | 06-01-17_10.36 0degC_NN_Pan18650PF.mat | NN | 6269 | 5825 | 0.9748 | 1.2296 | 5.9794 | 2.3711 | 3.1367 | 1.5422 |
| 0 | 0 | test | LSTM | 06-02-17_17.14 0degC_UDDS_Pan18650PF.mat | UDDS | 12794 | 5825 | 0.6066 | 0.7967 | 3.0554 | 1.6332 | 2.4049 | 3.0247 |
| 0 | 0 | test | Filtered CNN-LSTM Teacher | 06-01-17_10.36 0degC_LA92_Pan18650PF.mat | LA92 | 8208 | 50689 | 0.8529 | 1.0365 | 4.1017 | 2.0266 | 2.7137 | -0.5405 |
| 0 | 0 | test | Filtered CNN-LSTM Teacher | 06-01-17_10.36 0degC_NN_Pan18650PF.mat | NN | 6269 | 50689 | 1.806 | 2.6988 | 13.001 | 6.3108 | 9.6798 | -2.0502 |
| 0 | 0 | test | Filtered CNN-LSTM Teacher | 06-02-17_17.14 0degC_UDDS_Pan18650PF.mat | UDDS | 12794 | 50689 | 0.9932 | 1.4697 | 5.5375 | 3.4851 | 4.5864 | 0.4788 |
| 0 | 0 | test | Filtered Tiny CNN-LSTM Student | 06-01-17_10.36 0degC_LA92_Pan18650PF.mat | LA92 | 8208 | 3841 | 1.9188 | 3.3315 | 13.0701 | 8.7916 | 12.5118 | 2.6927 |
| 0 | 0 | test | Filtered Tiny CNN-LSTM Student | 06-01-17_10.36 0degC_NN_Pan18650PF.mat | NN | 6269 | 3841 | 1.8259 | 2.8455 | 12.5833 | 7.1203 | 10.6948 | 1.6789 |
| 0 | 0 | test | Filtered Tiny CNN-LSTM Student | 06-02-17_17.14 0degC_UDDS_Pan18650PF.mat | UDDS | 12794 | 3841 | 1.627 | 3.0012 | 12.884 | 8.3691 | 12.268 | 3.2876 |
| 0 | 0 | test | Filtered Distilled Tiny CNN-LSTM | 06-01-17_10.36 0degC_LA92_Pan18650PF.mat | LA92 | 8208 | 3841 | 2.7209 | 3.5264 | 11.3385 | 7.0936 | 10.7841 | 1.0408 |
| 0 | 0 | test | Filtered Distilled Tiny CNN-LSTM | 06-01-17_10.36 0degC_NN_Pan18650PF.mat | NN | 6269 | 3841 | 2.8429 | 3.315 | 10.8533 | 5.7644 | 8.9616 | 0.1525 |
| 0 | 0 | test | Filtered Distilled Tiny CNN-LSTM | 06-02-17_17.14 0degC_UDDS_Pan18650PF.mat | UDDS | 12794 | 3841 | 2.1746 | 3.0328 | 11.1532 | 6.6653 | 10.5345 | 1.6178 |
