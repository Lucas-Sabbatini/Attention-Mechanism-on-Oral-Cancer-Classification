#### XGBoost Classifier
##### window_length=14, poly_order=9, deriv=0
| Preprocessing Pipeline | Accuracy | Precision | Recall (Sensitivity) | Specificity | Mean(SE, SP) |
|------------------------|----------|-----------|----------------------|-------------|-------------|
| Raw (No Normalization) | 0.5905 ± 0.2059 | 0.6650 ± 0.1671 | 0.6750 ± 0.2512 | 0.4667 ± 0.2963 | 0.5708 ± 0.2213 |
| Rubberband (No SavGol) | 0.5881 ± 0.1710 | 0.6550 ± 0.1630 | 0.7167 ± 0.2363 | 0.4167 ± 0.2814 | 0.5667 ± 0.1658 |
| **AsLS (No SavGol)** | **0.7595 ± 0.2196** | **0.7900 ± 0.1960** | **0.8500 ± 0.1658** | **0.6167 ± 0.3804** | **0.7333 ± 0.2452** |
| Polynomial | 0.4952 ± 0.1432 | 0.5767 ± 0.1114 | 0.6750 ± 0.1601 | 0.2333 ± 0.2494 | 0.4542 ± 0.1629 |

#### SVM-RBF Classifier
##### window_length=11, poly_order=10, deriv=2
| Preprocessing Pipeline | Accuracy | Precision | Recall (Sensitivity) | Specificity | Mean(SE, SP) |
|------------------------|----------|-----------|----------------------|-------------|-------------|
| Raw (No Normalization) | 0.6024 ± 0.0564 | 0.6024 ± 0.0564 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.5000 ± 0.0000 |
| Rubberband (No SavGol) | 0.6024 ± 0.0564 | 0.6024 ± 0.0564 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.5000 ± 0.0000 |
| **AsLS (No SavGol)** | **0.6167 ± 0.0643** | **0.6119 ± 0.0584** | **1.0000 ± 0.0000** | **0.0333 ± 0.1000** | **0.5167 ± 0.0500** |
| Polynomial | 0.6167 ± 0.0643 | 0.6119 ± 0.0584 | 1.0000 ± 0.0000 | 0.0333 ± 0.1000 | 0.5167 ± 0.0500 |

#### LightGBM Classifier
#### window_length=17, poly_order=12, deriv=0
| Preprocessing Pipeline | Accuracy | Precision | Recall (Sensitivity) | Specificity | Mean(SE, SP) |
|------------------------|----------|-----------|----------------------|-------------|-------------|
| Raw (No Normalization) | 0.4905 ± 0.1955 | 0.5955 ± 0.1980 | 0.6167 ± 0.2273 | 0.3000 ± 0.3399 | 0.4583 ± 0.2024 |
| Rubberband (No SavGol) | 0.5452 ± 0.2147 | 0.6083 ± 0.1705 | 0.7000 ± 0.2449 | 0.3500 ± 0.2930 | 0.5250 ± 0.2175 |
| **AsLS (No SavGol)** | **0.7262 ± 0.1688** | **0.8017 ± 0.1776** | **0.7417 ± 0.1601** | **0.7167 ± 0.2587** | **0.7292 ± 0.1780** |
| Polynomial | 0.4667 ± 0.1704 | 0.5683 ± 0.2036 | 0.5167 ± 0.2000 | 0.3833 ± 0.3167 | 0.4500 ± 0.1880 |


In general, increase performance sacrifing the  Standard Deviation of each metric. (Not the case for LightGBM)