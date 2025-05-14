import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# === Style settings ===
sns.set_style("ticks")
plt.rcParams.update({
    "font.family": "serif",
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "axes.linewidth": 0.8,
})

# === Data ===
methods = ["NNLS", "RCTD", "cell2location"]
x = np.arange(len(methods))
width = 0.35

jsd_visium = [0.380, 0.070, 0.203]
jsd_hd = [0.619, 0.043, 0.072]

# === Final professional color scheme
visium_color = "#4e79a7"    # Navy blue (Visium)
hd_color = "#f28e2b"        # Amber orange (Visium HD)

# === Plot
fig, ax = plt.subplots(figsize=(4.2, 2.6))
bars1 = ax.bar(x - width/2, jsd_visium, width, label="Visium", color=visium_color)
bars2 = ax.bar(x + width/2, jsd_hd, width, label="Visium HD", color=hd_color)

# === Annotate bars
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height + 0.015),
                ha='center', va='bottom', fontsize=8)

for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height + 0.015),
                ha='center', va='bottom', fontsize=8)

# === Axes and labels
ax.set_ylabel("Jensen-Shannon Divergence", fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.set_ylim(0, max(jsd_hd) + 0.1)
ax.set_title("JSD to Ground Truth across Deconvolution Methods", fontsize=10, pad=6)

# === Legend and cleanup
ax.legend(frameon=False, loc="upper right")
sns.despine(ax=ax)
ax.tick_params(length=3, width=0.8)

plt.tight_layout(pad=0.5)
plt.savefig("jsd_combined_final.png", bbox_inches="tight", dpi=600)
plt.savefig("jsd_combined_final.pdf", bbox_inches="tight")
plt.close()
print("✅ Saved: jsd_combined_final.png & .pdf")


