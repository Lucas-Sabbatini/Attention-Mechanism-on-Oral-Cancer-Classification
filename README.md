# Attention Mechanism on Oral Cancer Classification


This project aims to validate the hypothesis of a Transformer-based model for oral cancer classification using spectroscopic fingerprint data. Other machine learning algorithms (Support Vector Machines, Convolutional Neural Networks, Long Short-Term Memory and Extreme Gradient Boosting) are being tested and evaluated for comparison purposes.

## Dataset

The dataset is protected by the Federal University of Uberlândia, and therefore cannot be made public for ethical reasons.

<p align="center">
  <img src="./ploting/img/all_samples_overlapped.png" alt="Overview" width="49%" />
  <img src="./ploting/img/mean_std_plot.png" alt="Mean and Standard Deviation" width="49%" />
</p>


- **Input**: Spectroscopic data with wavenumber measurements
- **Output**: Binary classification (-1: non-cancerous, 1: cancerous)
- **Features**: Spectral intensities across different wavenumbers
- **Class distribution**: Cancerous (26 samples) and Non-cancerous (39 samples)

## Preprocessing Pipeline

### 1. **Baseline Correction**: 
Spectroscopy data can suffer several kinds of distorsion, such as radiation scattering, absorption by the supporting substrate, fluctuations in data acquisition conditions, and instrumental instabilities can compromise the accuracy of absorbance values. To mitigate these effects, baseline correction is applied resulting in a purer and more interpretable signal, enabling the precise determination of spectral parameters. 

![Different Baseline Correction Methos](./ploting/img/baseline_corrections_comparison_sample_0.png)
![Baseline Estimate Methods Overview](./ploting/img/baseline_estimate_comparison_sample_0.png)

In this project we are willing to evaluate three different baseline correction algorithims:
1. **Polynomial baseline correction**: A Polynomial function is fitted to the spectrum and subtracted to remove baseline drift.
2. **Rubberband**: A convex hull is constructed over the spectrum, and the baseline is estimated by connecting the lowest points of the convex hull.
3. **Asymmetric least squares (ASLS)**: An iterative method that minimizes a cost function combining fidelity to the data and smoothness of the baseline, with an asymmetry parameter to handle positive peaks.

### 2. **Wavenumber Truncation**: 
Focuses analysis on the biological relevant spectral region (850-3050 cm⁻¹) in order to avoid noises and outliers from less informative regions.

### 3. **Normalization**: 
Standardizes data for model training, there are several normalization techniques available, like Min-Max Scaling, Mean Normalization but the most importat in this project is **Amidae-I** normalization.

Wich, normalizes each spectrum by its highest intensity value within the Amidae-I region (1660-1630 cm⁻¹).

## Training and Evaluation

We applied Stratified k-fold validation with k=10 to ensure robust evaluation of model performance. Dealing with these imbalanced dataset besides the lack of samples.

<p align="center">
    <img src="./ploting/img/stratified_kfold_visualization.png" alt="Stratified K-Fold Visualization" width="60%" />
</p>

### Metrics:
- **Accuracy**: Overall correctness of the model.
- **Precision**: Proportion of positive identifications that were actually correct.
- **Sensitivity (Recall)**: Proportion of actual positives that were correctly identified.
- **Specificity**: Proportion of actual negatives that were correctly identified.
- **F1-Score**: Harmonic mean of precision and recall, providing a balance between the two.

## Models

### Preprocessing Pipeline Comparison

The following tables show the performance of XGBoost and SVM-RBF models across different preprocessing pipelines using 10-fold stratified cross-validation:

#### XGBoost Classifier

| Preprocessing Pipeline | Accuracy | Precision | Recall (Sensitivity) | Specificity | F1 Score |
|------------------------|----------|-----------|----------------------|-------------|----------|
| **Raw (No Normalization)** | **0.6429 ± 0.1483** | **0.7100 ± 0.1234** | 0.7250 ± 0.1750 | **0.5167 ± 0.2522** | **0.7074 ± 0.1185** |
| Rubberband (No SavGol) | 0.6024 ± 0.1300 | 0.6733 ± 0.1517 | **0.7750 ± 0.1750** | 0.3833 ± 0.3078 | 0.6960 ± 0.0951 |
| AsLS (No Normalization) | 0.5857 ± 0.2331 | 0.6238 ± 0.2747 | 0.6500 ± 0.3000 | 0.4833 ± 0.3686 | 0.6263 ± 0.2674 |
| Polynomial | 0.5595 ± 0.1252 | 0.6367 ± 0.1394 | 0.6667 ± 0.1581 | 0.3833 ± 0.3167 | 0.6400 ± 0.1118 |
| AsLS | 0.4667 ± 0.1393 | 0.5583 ± 0.1902 | 0.6083 ± 0.2175 | 0.2500 ± 0.3096 | 0.5636 ± 0.1561 |

#### Support Vector Machine (RBF Kernel)

| Preprocessing Pipeline | Accuracy | Precision | Recall (Sensitivity) | Specificity | F1 Score |
|------------------------|----------|-----------|----------------------|-------------|----------|
| **Raw (No Normalization)** | **0.6226 ± 0.1140** | **0.6562 ± 0.1100** | 0.8625 ± 0.1850 | 0.2583 ± 0.3139 | **0.7288 ± 0.0920** |
| Rubberband (No SavGol) | 0.6024 ± 0.1002 | 0.6379 ± 0.1198 | **0.8875 ± 0.1672** | 0.1917 ± 0.2900 | 0.7232 ± 0.0789 |
| AsLS (No Normalization) | 0.6083 ± 0.1733 | 0.6226 ± 0.1986 | 0.8250 ± 0.2750 | **0.2750 ± 0.3467** | 0.6956 ± 0.2040 |
| Polynomial | 0.5810 ± 0.0994 | 0.6195 ± 0.1077 | 0.8333 ± 0.2007 | 0.1917 ± 0.2947 | 0.6952 ± 0.1014 |
| AsLS | 0.5345 ± 0.1261 | 0.5804 ± 0.1420 | 0.8042 ± 0.2490 | 0.1250 ± 0.2521 | 0.6569 ± 0.1479 |

**Key Findings:**
- **Raw data without normalization** achieves the best performance across both models
- XGBoost shows better balance between sensitivity and specificity compared to SVM-RBF
- SVM-RBF achieves higher recall but lower specificity, indicating bias toward the positive class


## Installation

1. Clone the repository
```bash
git clone git@github.com:Lucas-Sabbatini/Attention-Mechanism-on-Oral-Cancer-Classification.git
```

2. Create an Virual Enviroment
```bash
python3 -m venv .venv
```

3. Activate it
```bash
source .venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```



## Usage

### Using Preprocessing Components

```python
from preProcess.savitzky_filter import SavitzkyFilter
from preProcess.fingerprint_trucate import WavenumberTruncator
from preProcess.normalization import Normalization

# Apply Savitzky-Golay filter
filtered_data = SavitzkyFilter().buildFilter(X_data)

# Truncate wavenumber range
truncator = WavenumberTruncator()
truncated_data = truncator.trucate_range(3050.0, 850.0, X_data)

# Normalize data
normalizer = Normalization()
normalized_data = normalizer.normalize_data(X_data)
```
