import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob

# === Ground truth (excluding 'others') ===
gt = {
    "Hepatocytes": 0.702,
    "CentralVeinEndothelialcells": 0.008,
    "LSECs": 0.011,
    "PortalVeinEndothelialcells": 0.074,
    "Mesothelialcells": 0.005,
    "Kupffercells": 0.033,
    "Bcells": 0.014,
    "Cholangiocytes": 0.013,
    "Tcells": 0.011
}
gt_names = list(gt.keys())
gt_vector = np.array([gt[k] for k in gt_names])
gt_vector /= gt_vector.sum()

# === Config ===
methods = ["nnls", "rctd", "cell2loc"]  # match image order
base_dir = "deconv_proportions"
jsd_results = {}

# === Compute JSD per method ===
for method in methods:
    part_files = sorted(glob(f"{base_dir}/{method}/*"))
    if not part_files:
        print(f"❌ No files found for {method}")
        continue

    all_parts = []
    for f in part_files:
        df = pd.read_csv(f, sep="\t", index_col=0)
        for name in gt_names:
            if name not in df.columns:
                df[name] = 0
        df = df[gt_names]
        df = df.div(df.sum(axis=1), axis=0).fillna(0)
        all_parts.append(df)

    combined_df = pd.concat(all_parts)
    avg_vector = combined_df.mean(axis=0).values
    avg_vector /= avg_vector.sum()

    jsd = jensenshannon(gt_vector, avg_vector, base=2) ** 2
    jsd_results[method] = jsd

# === Plotting ===
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.1)

# Keep method order fixed
methods_ordered = ["nnls", "rctd", "cell2loc"]
jsd_vals = [jsd_results[m] for m in methods_ordered]
labels = ["NNLS", "RCTD", "cell2location"]
colors = sns.color_palette("Reds", n_colors=4)[1:]  # skip lightest red

fig, ax = plt.subplots(figsize=(4, 3))
bars = ax.bar(labels, jsd_vals, color=colors, edgecolor="black", width=0.6)

# Annotate with values
for bar, val in zip(bars, jsd_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{val:.3f}", ha='center', va='bottom', weight='bold', fontsize=9)

# Labels and title
ax.set_ylabel("JSD", fontsize=11)
ax.set_xlabel("Method", fontsize=11)
ax.set_title("Jensen-Shannon Divergence (Ground Truth vs nuclei)", fontsize=11, pad=10)

plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.ylim(0, max(jsd_vals) + 0.05)
plt.tight_layout()

plt.savefig("jsd_vertical_barplot_vs_gt.png", dpi=600, bbox_inches="tight")
plt.show()

print("✅ Saved vertical bar plot: jsd_vertical_barplot_vs_gt.png")
