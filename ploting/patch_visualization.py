import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

project_root = Path(__file__).parent.parent
dataset_path = project_root / "dataset_cancboca.dat"
wavenumbers_path = project_root / "wavenumbers_cancboca.dat"

dataset = np.loadtxt(dataset_path)
X = dataset[:, :-1]
y = dataset[:, -1].astype(int)
y = np.where(y == -1, 0, 1)

wavenumbers = np.loadtxt(wavenumbers_path)

# Patching parameters (must match SpectralTransformer defaults)
NUM_SPECTRAL_POINTS = X.shape[1]   # 1867
PATCH_SIZE = 16
PATCH_STRIDE = PATCH_SIZE // 2     # 8  (50% overlap)
NUM_PATCHES = (NUM_SPECTRAL_POINTS - PATCH_SIZE) // PATCH_STRIDE + 1  # 232

# Use a sample spectrum for the background
index = 0
sample = X[index]
label = y[index]
spectrum_color = "red" if label == 1 else "green"

# Compute patch start/end indices (in spectral-point space)
patch_starts = [i * PATCH_STRIDE for i in range(NUM_PATCHES)]
patch_ends   = [s + PATCH_SIZE for s in patch_starts]

# Convert to wavenumber space (wavenumbers are descending, so index 0 = highest wn)
wn_patch_left  = [wavenumbers[s] for s in patch_starts]   # higher wn (left edge on plot)
wn_patch_right = [wavenumbers[e - 1] for e in patch_ends] # lower wn  (right edge on plot)

# ── Figure layout: 2 rows ───────────────────────────────────────────────────
fig, (ax_full, ax_zoom) = plt.subplots(
    2, 1,
    figsize=(14, 9),
    gridspec_kw={"height_ratios": [2, 1.6]},
)
fig.suptitle(
    f"Spectral Patch Embedding — patch_size={PATCH_SIZE}, stride={PATCH_STRIDE} (50% overlap)\n"
    f"{NUM_SPECTRAL_POINTS} spectral points → {NUM_PATCHES} patches",
    fontsize=13, fontweight="bold",
)

even_color = "#4C72B0"
odd_color  = "#DD8452"

# ── Panel 1: Full spectrum with patch density stripe ────────────────────────
# Instead of trying to show every thin patch band (barely visible at this scale),
# draw a "coverage" stripe below the spectrum: a colored row per patch showing
# even/odd alternation, making the sliding window pattern visible.
ax_full.plot(wavenumbers, sample, color=spectrum_color, linewidth=1.2, zorder=3)

y_min_f, y_max_f = sample.min(), sample.max()
y_range_f = y_max_f - y_min_f
stripe_bottom = y_min_f - 0.30 * y_range_f
stripe_h      = 0.10 * y_range_f
row_h         = stripe_h / 2   # two interleaved rows (even / odd)

for i, (wl, wr) in enumerate(zip(wn_patch_left, wn_patch_right)):
    row = i % 2                 # 0 = top row (even), 1 = bottom row (odd)
    color = even_color if row == 0 else odd_color
    rect = mpatches.Rectangle(
        (min(wl, wr), stripe_bottom + (1 - row) * row_h),
        abs(wl - wr), row_h,
        linewidth=0, facecolor=color, alpha=0.75, zorder=2,
    )
    ax_full.add_patch(rect)

# Stripe labels
ax_full.text(wavenumbers[0] + 30, stripe_bottom + 1.5 * row_h, "even patches",
             va="center", ha="left", fontsize=7, color=even_color, fontweight="bold")
ax_full.text(wavenumbers[0] + 30, stripe_bottom + 0.5 * row_h, "odd patches",
             va="center", ha="left", fontsize=7, color=odd_color, fontweight="bold")

ax_full.set_xlim(wavenumbers[0], wavenumbers[-1])
ax_full.set_ylim(stripe_bottom - 0.02 * y_range_f, y_max_f + 0.05 * y_range_f)
ax_full.invert_xaxis()
ax_full.set_ylabel("Intensity", fontsize=11)
ax_full.set_title(
    f"Full spectrum — {NUM_PATCHES} patch windows shown as interleaved stripes (even/odd rows)",
    fontsize=11,
)
ax_full.grid(True, linestyle="--", alpha=0.4)

legend_handles = [
    mpatches.Patch(color=even_color, alpha=0.8, label="Even patches"),
    mpatches.Patch(color=odd_color,  alpha=0.8, label="Odd patches"),
    plt.Line2D([0], [0], color=spectrum_color, linewidth=1.5, label="Spectrum"),
]
ax_full.legend(handles=legend_handles, loc="upper right", fontsize=9)

# ── Panel 2: Zoomed view showing first N patches with explicit overlap ───────
N_SHOW = 12   # number of patches to show in zoom
zoom_idx_start = patch_starts[0]
zoom_idx_end   = patch_ends[N_SHOW - 1]
wn_zoom_left   = wavenumbers[zoom_idx_start]
wn_zoom_right  = wavenumbers[zoom_idx_end - 1]

ax_zoom.plot(wavenumbers, sample, color=spectrum_color, linewidth=1.4, zorder=3)

y_min, y_max = sample.min(), sample.max()
y_range = y_max - y_min
bar_y_bottom = y_min - 0.35 * y_range   # place patch bars below the spectrum
bar_height   = 0.12 * y_range

# Draw each patch as a horizontal bar with a border; overlap regions stand out
overlap_color  = "#9B59B6"   # purple for overlap zone
patch_colors   = [even_color, odd_color]

for i in range(N_SHOW):
    ps = patch_starts[i]
    pe = patch_ends[i]
    wl = wavenumbers[ps]
    wr = wavenumbers[pe - 1]
    fc = patch_colors[i % 2]

    # Shaded band on the spectrum
    ax_zoom.axvspan(wr, wl, alpha=0.22, color=fc, linewidth=0, zorder=1)

    # Patch bar below the spectrum
    bar_left  = min(wl, wr)
    bar_width = abs(wl - wr)
    rect = mpatches.FancyBboxPatch(
        (bar_left, bar_y_bottom + (i % 2) * bar_height * 0.5),
        bar_width, bar_height,
        boxstyle="round,pad=0",
        linewidth=0.8, edgecolor="white",
        facecolor=fc, alpha=0.85,
        zorder=4,
    )
    ax_zoom.add_patch(rect)

    # Patch index label
    center_wn = (wl + wr) / 2
    ax_zoom.text(
        center_wn,
        bar_y_bottom + (i % 2) * bar_height * 0.5 + bar_height / 2,
        f"P{i}",
        ha="center", va="center", fontsize=7, color="white", fontweight="bold", zorder=5,
    )

    # Vertical dashed boundary line on the spectrum
    ax_zoom.axvline(wl, color="gray", linewidth=0.6, linestyle="--", alpha=0.7, zorder=2)

# Last boundary
ax_zoom.axvline(wn_patch_right[N_SHOW - 1], color="gray", linewidth=0.6, linestyle="--", alpha=0.7, zorder=2)

# Annotate overlap arrow between P0 and P1
p0_right = wavenumbers[patch_ends[0] - 1]
p1_left  = wavenumbers[patch_starts[1]]
overlap_center = (p0_right + p1_left) / 2   # they overlap by PATCH_SIZE//2 points
overlap_wn_span = abs(p1_left - wavenumbers[patch_ends[0] - 1])

ax_zoom.annotate(
    f"overlap\n({PATCH_SIZE // 2} pts)",
    xy=(overlap_center, y_min + 0.08 * y_range),
    xytext=(overlap_center, y_min + 0.30 * y_range),
    ha="center", fontsize=8, color=overlap_color,
    arrowprops=dict(arrowstyle="-|>", color=overlap_color, lw=1.2),
    zorder=6,
)

ax_zoom.set_xlim(wn_zoom_left + 10, wn_zoom_right - 10)
ax_zoom.invert_xaxis()
ax_zoom.set_ylim(bar_y_bottom - 0.05 * y_range, y_max + 0.05 * y_range)
ax_zoom.set_xlabel("Wavenumber (cm⁻¹)", fontsize=11)
ax_zoom.set_ylabel("Intensity", fontsize=11)
ax_zoom.set_title(
    f"Zoom — first {N_SHOW} patches · each covers {PATCH_SIZE} points · stride {PATCH_STRIDE} pts (50% overlap)",
    fontsize=11,
)
ax_zoom.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()

output_path = Path(__file__).parent / "img" / "patch_visualization.png"
plt.savefig(output_path, dpi=300)
plt.close()
print(f"Saved to {output_path}")
