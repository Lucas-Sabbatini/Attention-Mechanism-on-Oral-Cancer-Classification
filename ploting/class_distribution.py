import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

project_root = Path(__file__).parent.parent
dataset_path = project_root / "dataset_cancboca.dat"

dataset = np.loadtxt(dataset_path)
X = dataset[:, :-1]
y = dataset[:, -1].astype(int)
y = np.where(y == -1, 0, 1)

unique, counts = np.unique(y, return_counts=True)
class_names = ['Healthy', 'Cancerous']

plt.figure(figsize=(8, 6))
bars = plt.bar(class_names, counts, color=['green', 'red'], alpha=0.7)

for bar, count in zip(bars, counts):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{count}',
             ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.title('Class Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Class', fontsize=12)
plt.ylabel('Number of Samples', fontsize=12)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()

output_path = Path(__file__).parent / "img" / "class_distribution.png"
plt.savefig(output_path, dpi=300)
plt.close()