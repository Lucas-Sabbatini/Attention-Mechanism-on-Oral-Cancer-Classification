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

index = 0  
sample = X[index]
label = y[index]

color = "red" if label == 1 else "green"
label_text = 'Cancerous' if label == 1 else 'Healthy'

plt.figure(figsize=(10, 4))
plt.plot(wavenumbers, sample, color=color, linewidth=2)
plt.title(f"Sample {index} — {label_text}", fontsize=14)
plt.xlabel("Wavenumber (cm⁻¹)")
plt.ylabel("Intensity")
plt.xlim(4000, 500)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

output_path = Path(__file__).parent / "img" / f"sample_{index}_class_{label}.png"
plt.savefig(output_path, dpi=300)  
plt.close()