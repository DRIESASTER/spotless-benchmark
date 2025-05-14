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
    "legend.fontsize": 12,
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
        group = adata_reg.obsm[obsm_key]
        cell_types = [k for k in group.keys() if k not in ("_index", "dominant_celltype", "highest_proportion")]
        deconv_df = pd.DataFrame({ct: group[ct] for ct in cell_types}, index=adata_reg.obs_names)
        deconv_clean = deconv_df.fillna(0)

        dominant_celltype = pd.Series(deconv_clean.idxmax(axis=1), index=adata_reg.obs_names)
        adata_reg.obs[f"dominant_{method_name}"] = dominant_celltype

        # Spatial plot
        plt.figure(figsize=(10, 10))
        sns.scatterplot(
            x=x_real,
            y=y_real,
            hue=dominant_celltype,
            palette="tab20",
            s=80,
            linewidth=0,
            alpha=0.95,
            legend=False
        )
        plt.gca().invert_yaxis()
        plt.xlabel("X coordinate (micrometers)")
        plt.ylabel("Y coordinate (micrometers)")
        plt.title(f"Dominant Cell Type per Spot using {method_name}", fontsize=16)
        plt.tight_layout()
        plt.savefig(f"visium_reg_dominant_{method_name}.png", dpi=300)
        plt.close()

        # Composition bar chart
        proportions = dominant_celltype.value_counts(normalize=True).sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x=proportions.index,
            y=proportions.values,
            palette="tab20"
        )
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Proportion of Spots")
        plt.xlabel("Cell Type")
        plt.title(f"Distribution of Dominant Cell Types using {method_name}", fontsize=14)
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
    s=80,
    alpha=0.95
)
plt.gca().invert_yaxis()
plt.xlabel("X coordinate (micrometers)")
plt.ylabel("Y coordinate (micrometers)")
plt.title("Agreement of Dominant Cell Type Assignments Across Methods")
cbar = plt.colorbar(scatter)
cbar.set_label("Number of Unique Cell Types Assigned per Spot")
plt.tight_layout()
plt.savefig("visium_reg_deconvolution_agreement.png", dpi=300)
plt.close()

# === Step 3: Clustering on each method's cell type proportions ===
if "clusters" not in adata_reg.uns:
    adata_reg.uns["clusters"] = {}

for method_name, obsm_key in methods.items():
    if obsm_key in adata_reg.obsm:
        print(f"🔍 Clustering on {method_name} proportions")
        group = adata_reg.obsm[obsm_key]
        cell_types = [k for k in group.keys() if k not in ("_index", "dominant_celltype", "highest_proportion")]
        features = pd.DataFrame({ct: group[ct] for ct in cell_types}, index=adata_reg.obs_names).fillna(0)

        adata_sub = sc.AnnData(X=features.values)
        adata_sub.obs.index = adata_reg.obs_names  # Ensure same index
        adata_sub.obs["x"] = x_real
        adata_sub.obs["y"] = y_real

        sc.pp.pca(adata_sub)
        sc.pp.neighbors(adata_sub)
        sc.tl.leiden(adata_sub, resolution=0.3, key_added="leiden")

        # Save clustering labels back to main AnnData
        cluster_col = f"cluster_{method_name}"
        adata_reg.obs[cluster_col] = adata_sub.obs["leiden"]

        # Also store as the default clustering label for downstream use
        adata_reg.obs["cluster_label"] = adata_sub.obs["leiden"]

        # Save all clusterings in .uns
        adata_reg.uns["clusters"][method_name] = adata_sub.obs["leiden"].to_list()

        # Spatial cluster plot
        plt.figure(figsize=(10, 10))
        sns.scatterplot(
            x=adata_sub.obs["x"],
            y=adata_sub.obs["y"],
            hue=adata_sub.obs["leiden"],
            palette="tab10",
            s=80,
            linewidth=0,
            alpha=0.95
        )
        plt.gca().invert_yaxis()
        plt.xlabel("X coordinate (micrometers)")
        plt.ylabel("Y coordinate (micrometers)")
        plt.title(f"Leiden Clustering of Spatial Spots ({method_name})")
        plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(f"visium_reg_clustering_{method_name}.png", dpi=300)
        plt.close()

# === Replace modified AnnData table and save SpatialData object ===
sdata_reg.tables["table"] = adata_reg
sdata_reg.write("spatial_objects/vis_labeled.zarr", overwrite=True)
print("💾 SpatialData object saved as vis_labeled.zarr with updated clustering and annotations.")

print("✅ All regular Visium analysis complete and plots saved.")
