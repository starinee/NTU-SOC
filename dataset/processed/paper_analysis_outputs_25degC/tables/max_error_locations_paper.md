| cycle | method                           | max_abs_error_percent | signed_error_percent | time_s    | reference_soc_percent | predicted_soc_percent |
| ----- | -------------------------------- | --------------------- | -------------------- | --------- | --------------------- | --------------------- |
| UDDS  | Coulomb Counting                 | 0.154                 | 0.154                | 22067.664 | 7.247                 | 7.401                 |
| UDDS  | Instantaneous MLP                | 13.895                | -13.895              | 13835.588 | 44.319                | 30.425                |
| UDDS  | Filtered-feature MLP             | 3.92                  | -3.92                | 13835.588 | 44.319                | 40.399                |
| UDDS  | LSTM                             | 2.229                 | 2.229                | 19707.614 | 17.529                | 19.758                |
| UDDS  | Filtered CNN-LSTM Teacher        | 2.805                 | -2.805               | 173.999   | 99.299                | 96.494                |
| UDDS  | Filtered Distilled Tiny CNN-LSTM | 4.361                 | -4.361               | 59.005    | 99.761                | 95.4                  |
| LA92  | Coulomb Counting                 | 0.087                 | -0.087               | 13278.796 | 15.334                | 15.248                |
| LA92  | Instantaneous MLP                | 16.761                | -16.761              | 428.993   | 96.727                | 79.966                |
| LA92  | Filtered-feature MLP             | 2.893                 | 2.893                | 11710.86  | 25.983                | 28.876                |
| LA92  | LSTM                             | 2.601                 | -2.601               | 488.994   | 95.9                  | 93.299                |
| LA92  | Filtered CNN-LSTM Teacher        | 2.993                 | -2.993               | 130.998   | 99.431                | 96.439                |
| LA92  | Filtered Distilled Tiny CNN-LSTM | 4.501                 | -4.501               | 61.001    | 99.916                | 95.414                |
| NN    | Coulomb Counting                 | 0.04                  | 0.04                 | 9585.4    | 28.698                | 28.738                |
| NN    | Instantaneous MLP                | 25.591                | -25.591              | 5983.63   | 56.848                | 31.257                |
| NN    | Filtered-feature MLP             | 4.189                 | 4.189                | 6229.633  | 53.664                | 57.853                |
| NN    | LSTM                             | 5.307                 | -5.307               | 6239.635  | 53.186                | 47.879                |
| NN    | Filtered CNN-LSTM Teacher        | 3.584                 | -3.584               | 94.99     | 97.937                | 94.353                |
| NN    | Filtered Distilled Tiny CNN-LSTM | 3.88                  | -3.88                | 58.996    | 99.161                | 95.281                |
