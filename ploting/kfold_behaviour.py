import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from pathlib import Path

from sklearn.model_selection import StratifiedKFold

cmap_data = plt.cm.Paired
cmap_cv = plt.cm.coolwarm
n_splits = 10

def plot_cv_indices(cv, X, y, ax, n_splits, lw=10):
    """Create a sample plot for indices of a cross-validation object."""
    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(
            range(len(indices)),
            [ii + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=lw,
            cmap=cmap_cv,
            vmin=-0.2,
            vmax=1.2,
        )

    # Plot the data classes and groups at the end
    ax.scatter(
        range(len(X)), [ii + 1.5] * len(X), c=y, marker="_", lw=lw, cmap=cmap_data
    )


    # Formatting
    yticklabels = list(range(n_splits)) + ["class"]
    ax.set(
        yticks=np.arange(n_splits + 1) + 0.5,
        yticklabels=yticklabels,
        xlabel="Sample index",
        ylabel="CV iteration",
        ylim=[n_splits + 2.2, -0.2],
        xlim=[0, 64],
    )
    ax.set_title("{}".format(type(cv).__name__), fontsize=15)
    return ax

project_root = Path(__file__).parent.parent
dataset_path = project_root / "dataset_cancboca.dat"

dataset = np.loadtxt(dataset_path)
X = dataset[:, :-1]
y = dataset[:, -1].astype(int)
y = np.where(y == -1, 0, 1)

fig, ax = plt.subplots()
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
plot_cv_indices(cv, X, y, ax, n_splits)
ax.legend(
        [Patch(color=cmap_cv(0.8)), Patch(color=cmap_cv(0.02))],
        ["Testing set", "Training set"],
        loc=(1.02, 0.8),
    )

output_path = project_root / "ploting" / "img" / "stratified_kfold_visualization.png"
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {output_path}")
