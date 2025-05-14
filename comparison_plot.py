import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from glob import glob
import scanpy as sc
import anndata as ad

# === CONFIG ===
visium_hd_dir = "vis_hd_rctd/"
visium_file = "deconv_proportions/proportions_rctd_liver_mouseVisium_JB01"
output_dir = "comparison_figures"
celltype_palette = sns.color_palette("tab20", 20)

# === Load HD files and merge ===
hd_files = sorted(glob(f"{visium_hd_dir}/*"))
df_hd = pd.concat([pd.read_csv(f, sep="\t", index_col=0) for f in hd_files])
df_hd.index = df_hd.index.astype(str)

# === Load Visium proportions ===
df_vis = pd.read_csv(visium_file, sep="\t", index_col=0)
df_vis.index = df_vis.index.astype(str)

# === Step 1: Global proportions ===
mean_vis = df_vis.mean(axis=0)
mean_hd = df_hd.mean(axis=0)

# Normalize to sum = 1
mean_vis_norm = mean_vis / mean_vis.sum()
mean_hd_norm = mean_hd / mean_hd.sum()

# Combine for plotting
df_prop = pd.DataFrame({
    "Visium": mean_vis_norm,
    "Visium HD": mean_hd_norm
}).reset_index().rename(columns={"index": "Cell type"})

# Plot global proportions
plt.figure(figsize=(10, 5))
df_prop_melt = df_prop.melt(id_vars="Cell type", var_name="Platform", value_name="Proportion")
sns.barplot(data=df_prop_melt, x="Cell type", y="Proportion", hue="Platform", palette="Set2")
plt.xticks(rotation=45, ha="right")
plt.title("Global Cell Type Proportions")
plt.tight_layout()
plt.savefig(f"{output_dir}/global_proportions_comparison.pdf")
plt.savefig(f"{output_dir}/global_proportions_comparison.png", dpi=300)
plt.show()

# === Step 2: Highest cell type per spot ===
top_vis = df_vis.idxmax(axis=1)
top_hd = df_hd.idxmax(axis=1)

# Plot distribution of dominant cell types
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.countplot(y=top_vis, order=top_vis.value_counts().index, ax=axes[0], palette=celltype_palette)
axes[0].set_title("Most Abundant Cell Type per Spot (Visium)")
axes[0].set_xlabel("Count")

sns.countplot(y=top_hd, order=top_hd.value_counts().index, ax=axes[1], palette=celltype_palette)
axes[1].set_title("Most Abundant Cell Type per Spot (Visium HD)")
axes[1].set_xlabel("Count")

plt.tight_layout()
plt.savefig(f"{output_dir}/dominant_celltype_comparison.pdf")
plt.savefig(f"{output_dir}/dominant_celltype_comparison.png", dpi=300)
plt.show()

# === Step 3: Spatial visualization on UMAP ===
# Convert both to AnnData for UMAP
adata_vis = ad.AnnData(df_vis.values)
adata_vis.obs_names = df_vis.index
adata_vis.var_names = df_vis.columns
adata_hd = ad.AnnData(df_hd.values)
adata_hd.obs_names = df_hd.index
adata_hd.var_names = df_hd.columns

# Add top cell types
adata_vis.obs["top_cell"] = top_vis
adata_hd.obs["top_cell"] = top_hd

# UMAP
for adata in [adata_vis, adata_hd]:
    sc.pp.scale(adata)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)

# Plot top cell type on UMAP
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
sc.pl.umap(adata_vis, color="top_cell", ax=axes[0], title="Visium - Dominant Cell Type", show=False, legend_loc='right margin', palette=celltype_palette)
sc.pl.umap(adata_hd, color="top_cell", ax=axes[1], title="Visium HD - Dominant Cell Type", show=False, legend_loc='right margin', palette=celltype_palette)
plt.tight_layout()
plt.savefig(f"{output_dir}/umap_top_celltype_comparison.pdf")
plt.savefig(f"{output_dir}/umap_top_celltype_comparison.png", dpi=300)
plt.show()