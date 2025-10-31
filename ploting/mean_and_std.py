import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

project_root = Path(__file__).parent.parent
dataset_path = project_root / "dataset_cancboca.dat"
wavenumbers_path = project_root / "wavenumbers_cancboca.dat"

dataset = np.loadtxt(dataset_path)
X = dataset[:, :-1]
y = dataset[:, -1].astype(int)
y = np.where(y == -1, 0, 1)
wavenumbers = np.loadtxt(wavenumbers_path)

X_healthy = X[y == 0]
X_cancerous = X[y == 1]

mean_healthy = X_healthy.mean(axis=0)
std_healthy = X_healthy.std(axis=0)
mean_cancerous = X_cancerous.mean(axis=0)
std_cancerous = X_cancerous.std(axis=0)

plt.figure(figsize=(10, 10))
plt.plot(wavenumbers, mean_healthy, color='green', linewidth=2, label='Healthy (mean)')
plt.fill_between(wavenumbers, mean_healthy - std_healthy, mean_healthy + std_healthy, 
                 color='green', alpha=0.2, label='Healthy (±1 std)')

plt.plot(wavenumbers, mean_cancerous, color='red', linewidth=2, label='Cancerous (mean)')
plt.fill_between(wavenumbers, mean_cancerous - std_cancerous, mean_cancerous + std_cancerous, 
                 color='red', alpha=0.2, label='Cancerous (±1 std)')

plt.title('Mean Spectra with Standard Deviation by Class', fontsize=14, fontweight='bold')
plt.xlabel('Wavenumber (cm⁻¹)', fontsize=12)
plt.ylabel('Intensity', fontsize=12)
plt.xlim(4000, 500)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

output_path = Path(__file__).parent / "img" / "mean_std_plot.png"
plt.savefig(output_path, dpi=300)
plt.close()