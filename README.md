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

![teste](./ploting/img/stratified_kfold_visualization.png)

### Metrics:
- **Accuracy**: Overall correctness of the model.
- **Precision**: Proportion of positive identifications that were actually correct.
- **Sensitivity (Recall)**: Proportion of actual positives that were correctly identified.
- **Specificity**: Proportion of actual negatives that were correctly identified.
- **F1-Score**: Harmonic mean of precision and recall, providing a balance between the two.

## Models

### XGBoost Classifier

**Results:**

| Metric              | Score   |
|---------------------|---------|
| Accuracy            | 0.7692  |
| Precision           | 0.8571  |
| Sensitivity (Recall)| 0.7500  |
| Specificity         | 0.8000  |
| F1-Score            | 0.8000  |

### Support Vector Machines - Linear Kernel

**Results:**

| Metric              | Score   |
|---------------------|---------|
| Accuracy            | 0.6154  |
| Precision           | 0.6364  |
| Sensitivity (Recall)| 0.8750  |
| Specificity         | 0.2000  |
| F1-Score            | 0.7368  |


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
