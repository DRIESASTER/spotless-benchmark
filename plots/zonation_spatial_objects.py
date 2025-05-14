import scanpy as sc
import spatialdata as sd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib as mpl

# === Plotting style ===
mpl.rcParams.update({
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 300
})

# === Load SpatialData ===
sdata_reg = sd.SpatialData.read("spatial_objects/vis.zarr")
adata_reg = sdata_reg.tables["table"]
spatial_coords = adata_reg.obsm["spatial"]
x_real = spatial_coords[:, 0]
y_real = spatial_coords[:, 1]

# === Deconvolution methods to compare ===
methods = {
    "cell2location": "visium_cell2loc",
    "rCTD": "visium_rctd",
    "NNLS": "visium_nnls"
}

# === Step 1: Dominant cell type plots + compositions ===
for method_name, obsm_key in methods.items():
    if obsm_key in adata_reg.obsm:
        print(f"✅ Processing: {method_name}")
        deconv = adata_reg.obsm[obsm_key]
        deconv_clean = pd.DataFrame(deconv).fillna(0)
        dominant_celltype = pd.Series(deconv_clean.idxmax(axis=1), index=adata_reg.obs_names)
        adata_reg.obs[f"dominant_{method_name}"] = dominant_celltype

        # Spatial plot
        plt.figure(figsize=(10, 10))
        sns.scatterplot(
            x=x_real,
            y=y_real,
            hue=dominant_celltype,
            palette="tab20",
            s=20,
            linewidth=0,
            alpha=0.9,
            legend=False
        )
        plt.gca().invert_yaxis()
        plt.xlabel("X coordinate (µm)")
        plt.ylabel("Y coordinate (µm)")
        plt.title(f"Dominant Cell Type - {method_name} (Regular Visium)")
        plt.tight_layout()
        plt.savefig(f"visium_reg_dominant_{method_name}.png", dpi=300)
        plt.close()

        # Composition bar chart
        proportions = dominant_celltype.value_counts(normalize=True).sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x=proportions.index,
            y=proportions.values,
            hue=proportions.index,
            palette="tab20",
            dodge=False,
            legend=False
        )
        plt.xticks(rotation=90)
        plt.ylabel("Proportion of Spots")
        plt.xlabel("Cell Type")
        plt.title(f"Cell Type Composition - {method_name} (Regular Visium)")
        plt.tight_layout()
        plt.savefig(f"visium_reg_celltype_composition_{method_name}.png", dpi=300)
        plt.close()

# === Step 2: Deconvolution agreement heatmap ===
dominants = pd.DataFrame({
    method: adata_reg.obs[f"dominant_{method}"]
    for method in methods if f"dominant_{method}" in adata_reg.obs
})

agreement = dominants.apply(lambda row: len(set(row)), axis=1)
adata_reg.obs["deconv_agreement"] = agreement

norm = mpl.colors.Normalize(vmin=1, vmax=3)
cmap = mpl.cm.viridis
plt.figure(figsize=(10, 10))
scatter = plt.scatter(
    x_real,
    y_real,
    c=agreement,
    cmap=cmap,
    norm=norm,
    s=20,
    alpha=0.9
)
plt.gca().invert_yaxis()
plt.xlabel("X coordinate (µm)")
plt.ylabel("Y coordinate (µm)")
plt.title("Deconvolution Agreement Across Methods (Regular Visium)")
cbar = plt.colorbar(scatter)
cbar.set_label("Number of Distinct Assignments")
plt.tight_layout()
plt.savefig("visium_reg_deconvolution_agreement.png", dpi=300)
plt.close()

# === Step 3: Clustering on each method's cell type proportions ===
for method_name, obsm_key in methods.items():
    if obsm_key in adata_reg.obsm:
        print(f"🔍 Clustering on {method_name} proportions")
        features = pd.DataFrame(adata_reg.obsm[obsm_key]).fillna(0)

        adata_sub = sc.AnnData(X=features.values)
        adata_sub.obs["x"] = x_real
        adata_sub.obs["y"] = y_real

        sc.pp.pca(adata_sub)
        sc.pp.neighbors(adata_sub)
        sc.tl.leiden(adata_sub, resolution=0.3, key_added="leiden")

        # Spatial cluster plot
        plt.figure(figsize=(10, 10))
        sns.scatterplot(
            x=adata_sub.obs["x"],
            y=adata_sub.obs["y"],
            hue=adata_sub.obs["leiden"],
            palette="tab10",
            s=20,
            linewidth=0,
            alpha=0.9
        )
        plt.gca().invert_yaxis()
        plt.xlabel("X coordinate (µm)")
        plt.ylabel("Y coordinate (µm)")
        plt.title(f"Spatial Clusters (Leiden) from {method_name} Proportions (Regular Visium)")
        plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(f"visium_reg_clustering_{method_name}.png", dpi=300)
        plt.close()

print("✅ All regular Visium analysis complete and plots saved.")
