import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from preProcess.baseline_correction import BaselineCorrection

project_root = Path(__file__).parent.parent
dataset_path = project_root / "dataset_cancboca.dat"
wavenumbers_path = project_root / "wavenumbers_cancboca.dat"

dataset = np.loadtxt(dataset_path)
X = dataset[:, :-1]
y = dataset[:, -1].astype(int)
y = np.where(y == -1, 0, 1)

wavenumbers = np.loadtxt(wavenumbers_path)

index = 0  
sample = X[index]
label = y[index]

label_text = 'Cancerous' if label == 1 else 'Healthy'

# Apply baseline correction methods
baseline_corrector = BaselineCorrection()

# AsLS baseline correction
baseline_asls = baseline_corrector.asls_baseline(sample)
sample_asls = sample - baseline_asls

# Rubberband baseline correction
baseline_rubberband = baseline_corrector.rubberband_baseline(sample)
sample_rubberband = sample - baseline_rubberband

# Polynomial baseline correction
baseline_polynomial = baseline_corrector.polynomial_baseline(sample)
sample_polynomial = sample - baseline_polynomial

# Create plot with all methods overlapped
plt.figure(figsize=(12, 6))
plt.plot(wavenumbers, sample, color='black', linewidth=2.5, label='Original', alpha=0.8)
plt.plot(wavenumbers, sample_asls, color='blue', linewidth=1.5, label='AsLS', alpha=0.9)
plt.plot(wavenumbers, sample_rubberband, color='green', linewidth=1.5, label='Rubberband', alpha=0.9)
plt.plot(wavenumbers, sample_polynomial, color='red', linewidth=1.5, label='Polynomial', alpha=0.9)


plt.title(f"Baseline Correction Methods — Sample {index} ({label_text})", fontsize=14)
plt.xlabel("Wavenumber (cm⁻¹)", fontsize=12)
plt.ylabel("Intensity", fontsize=12)
plt.xlim(4000, 500)
plt.legend(loc='best', fontsize=10)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

output_path = Path(__file__).parent / "img" / f"baseline_corrections_comparison_sample_{index}.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()