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

plt.figure(figsize=(10, 10))
for i in range(len(X)-1, -1, -1):
    sample = X[i]
    label = y[i]
    color = "red" if label == 1 else "green"
    alpha = 0.75
    
    #First line from each class to add its legend to the plot
    if i == len(X)-1 or (label == 0 and y[i+1:].sum() == len(y[i+1:])):
        label_text = 'Cancerous' if label == 1 else 'Healthy'
        plt.plot(wavenumbers, sample, color=color, linewidth=1, alpha=alpha, label=label_text)
    else:
        plt.plot(wavenumbers, sample, color=color, linewidth=1, alpha=alpha)

plt.title(f"All Samples Overlapped", fontsize=14)
plt.xlabel("Wavenumber (cm⁻¹)")
plt.ylabel("Intensity")
plt.xlim(4000, 500)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

output_path = Path(__file__).parent / "img" / "all_samples_overlapped.png"
plt.savefig(output_path, dpi=300)  
plt.close()