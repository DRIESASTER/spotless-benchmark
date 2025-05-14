import scanpy as sc
import spatialdata as sd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as mpl

# === Plotting style ===
mpl.rcParams.update({
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300
})

# === Load SpatialData ===
print("📂 Loading SpatialData...")
sdata_reg = sd.SpatialData.read("spatial_objects/vis.zarr")
adata_reg = sdata_reg.tables["table"]
spatial_coords = adata_reg.obsm["spatial"]
x_real = spatial_coords[:, 0]
y_real = spatial_coords[:, 1]

# === Deconvolution methods to compare ===
methods = {
    "Cell2location": "visium_cell2loc",
    "RCTD": "visium_rctd",
    "NNLS": "visium_nnls"
}

# === Step 1: Load and normalize proportion matrices ===
proportions_df = {}
for method, obsm_key in methods.items():
    if obsm_key in adata_reg.obsm:
        df = pd.DataFrame(adata_reg.obsm[obsm_key])
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
        df = df.div(df.sum(axis=1), axis=0).fillna(0)  # Normalize rows
        proportions_df[method] = df
    else:
        print(f"⚠️ Warning: {obsm_key} not found in adata_reg.obsm")

# === Step 2: Identify most divergent cell type ===
print("📊 Calculating divergence...")
avg_props = {m: df.mean(axis=0) for m, df in proportions_df.items()}
avg_props_df = pd.DataFrame(avg_props)
divergence = avg_props_df.max(axis=1) - avg_props_df.min(axis=1)
divergent_celltype = divergence.idxmax()
print(f"🧬 Most divergent cell type: {divergent_celltype}")

# === Step 3: Plot side-by-side maps ===
print("🖼️ Generating comparison plot...")

# Determine shared color scale
vmin = min(df[divergent_celltype].min() for df in proportions_df.values() if divergent_celltype in df)
vmax = max(df[divergent_celltype].max() for df in proportions_df.values() if divergent_celltype in df)

fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True, constrained_layout=True)
scplots = []

for i, (method_name, obsm_key) in enumerate(methods.items()):
    ax = axs[i]
    if method_name in proportions_df:
        df = proportions_df[method_name]
        if divergent_celltype in df.columns:
            values = df[divergent_celltype]
            sc = ax.scatter(x_real, y_real, c=values, cmap="viridis", vmin=vmin, vmax=vmax, s=20, alpha=0.9)
            scplots.append(sc)
            ax.set_title(f"{method_name}", fontsize=14)
        else:
            ax.text(0.5, 0.5, "Cell type not found", ha='center', va='center', transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, "Data missing", ha='center', va='center', transform=ax.transAxes)

    ax.invert_yaxis()
    ax.set_xlabel("X coordinate (µm)")
    if i == 0:
        ax.set_ylabel("Y coordinate (µm)")

# Colorbar (shared)
cbar = fig.colorbar(scplots[0], ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
cbar.set_label("Normalized Proportion", fontsize=12)

# Super title
fig.suptitle(f"Spatial distribution of most divergent cell type: '{divergent_celltype}'", fontsize=16)

# Save plot
output_file = f"divergent_celltype_{divergent_celltype}_comparison_normalized.png"
plt.savefig(output_file, dpi=300)
print(f"✅ Saved plot to: {output_file}")

