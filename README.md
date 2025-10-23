# Attention Mechanism on Oral Cancer Classification


This project aims to validate the hypothesis of a Transformer-based model for oral cancer classification using spectroscopic fingerprint data. Other machine learning algorithms (Support Vector Machines, Convolutional Neural Networks, Long Short-Term Memory and Extreme Gradient Boosting) are being tested and evaluated for comparison purposes.

## Dataset

The dataset is protected by the Federal University of Uberlândia, and therefore cannot be made public for ethical reasons.

- **Input**: Spectroscopic data with wavenumber measurements
- **Output**: Binary classification (-1: non-cancerous, 1: cancerous)
- **Features**: Spectral intensities across different wavenumbers

## Preprocessing Pipeline

1. **Savitzky-Golay Filter**: Smooths spectral data to reduce noise
2. **Wavenumber Truncation**: Focuses analysis on the relevant spectral region (850-3050 cm⁻¹)
3. **Normalization**: (Optional) Standardizes data for model training

## Models

### XGBoost Classifier

**Model Parameters:**
- `max_depth`: 2
- `eta` (learning rate): 1
- `objective`: binary:logistic
- `nthread`: 4
- `eval_metric`: AUC
- `num_round`: 10 boosting rounds

**Training Configuration:**
- Train/Test Split: 80/20
- Random State: 42
- Prediction Threshold: 0.5

**Results:**

| Metric              | Score   |
|---------------------|---------|
| Accuracy            | 0.7692  |
| Precision           | 0.8571  |
| Sensitivity (Recall)| 0.7500  |
| Specificity         | 0.8000  |
| F1-Score            | 0.8000  |


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
