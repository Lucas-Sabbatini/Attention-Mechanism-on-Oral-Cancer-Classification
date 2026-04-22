#### XGBoost Classifier
##### window_length=6, poly_order=2, deriv=0
| Preprocessing Pipeline | Accuracy | Precision | Recall (Sensitivity) | Specificity | Mean(SE, SP) |
|------------------------|----------|-----------|----------------------|-------------|-------------|s
| **AsLS (No SavGol)** | **0.7452 ± 0.2175** | **0.7567 ± 0.1855** | **0.8750 ± 0.1677** | **0.5500 ± 0.3655** | **0.7125 ± 0.2411** |

#### SVM-RBF Classifier
##### window_length=3, poly_order=2, deriv=0
| Preprocessing Pipeline | Accuracy | Precsision | Recall (Sensitivity) | Specificity | Mean(SE, SP) |
|------------------------|----------|-----------|----------------------|-------------|-------------|
| **AsLS (No SavGol)** | **0.6214 ± 0.1623** | **0.6467 ± 0.1256** | **0.8250 ± 0.1601** | **0.3167 ± 0.2291** | **0.5708 ± 0.1625** |

#### LightGBM Classifier
##### window_length=24, poly_order=3, deriv=0
| Preprocessing Pipeline | Accuracy | Precision | Recall (Sensitivity) | Specificity | Mean(SE, SP) |
|------------------------|----------|-----------|----------------------|-------------|-------------|
| **AsLS (No SavGol)** | **0.7429 ± 0.2031** | **0.8500 ± 0.2049** | **0.7417 ± 0.1601** | **0.7500 ± 0.3436** | **0.7458 ± 0.2177** |

#### TabPFN Classifier
##### window_length=9, poly_order=4, deriv=2    
| Preprocessing Pipeline | Accuracy | Precision | Recall (Sensitivity) | Specificity | Mean(SE, SP) |
|------------------------|----------|-----------|----------------------|-------------|-------------|
| AsLS (No SavGol) | 0.6810 ± 0.1491 | 0.7200 ± 0.1282 | 0.7500 ± 0.2500 | 0.5667 ± 0.1856 | 0.6583 ± 0.1341 |

#### CatBoost Classifier
##### window_length=3, poly_order=2, deriv=2
| Preprocessing Pipeline | Accuracy | Precision | Recall (Sensitivity) | Specificity | Mean(SE, SP) |
|------------------------|----------|-----------|----------------------|-------------|-------------|
| **AsLS (No SavGol)** | **0.7571 ± 0.1170** | **0.7838 ± 0.1255** | **0.8750 ± 0.1677** | **0.6000 ± 0.2809** | **0.7375 ± 0.1278** |

#### TabM Classifier
##### window_length=3, poly_order=2, deriv=0
| Preprocessing Pipeline | Accuracy | Precision | Recall (Sensitivity) | Specificity | Mean(SE, SP) |
|------------------------|----------|-----------|----------------------|-------------|-------------|
| **AsLS (No SavGol)** | **0.6024 ± 0.1536** | **0.6733 ± 0.1581** | **0.7167 ± 0.1756** | **0.4167 ± 0.3270** | **0.5667 ± 0.1740** |

#### RealMLP Classifier
##### window_length=13, poly_order=4, deriv=0
| Preprocessing Pipeline | Accuracy | Precision | Recall (Sensitivity) | Specificity | Mean(SE, SP) |
|------------------------|----------|-----------|----------------------|-------------|-------------|
| **AsLS (No SavGol)** | **0.6643 ± 0.2459** | **0.7233 ± 0.2796** | **0.7250 ± 0.2839** | **0.5833 ± 0.3184** | **0.6542 ± 0.2472** |
