import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
from glob import glob
from sklearn.cluster import KMeans
import re

# === CONFIG ===
visium_hd_dir = "vis_hd_rctd/"
visium_file = "deconv_proportions/prop"
tissue_positions_file = "tissue_positions.csv"  # must be same as used in your splitting
n_clusters = 12
spot_prefix = "s_016um_"  # your synthetic ID format

# === STEP 1: Load Visium HD proportions and Visium proportions ===
hd_files = sorted(glob(f"{visium_hd_dir}/*"))
df_hd = pd.concat([pd.read_csv(f, sep="\t", index_col=0) for f in hd_files])
df_hd.index = df_hd.index.astype(str)

df_vis = pd.read_csv(visium_file, sep="\t", index_col=0)
df_vis.index = df_vis.index.astype(str)

# === STEP 2: Wrap in AnnData ===
adata_hd = ad.AnnData(df_hd)
adata_vis = ad.AnnData(df_vis)

# === STEP 3: Recover spatial coordinates from synthetic spot names ===
def extract_xy(spot_id):
    match = re.match(rf"{spot_prefix}(\d{{5}})_(\d{{5}})", spot_id)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

for adata in [adata_hd, adata_vis]:
    adata.obs["x"], adata.obs["y"] = zip(*adata.obs_names.map(extract_xy))
    adata.obsm["spatial"] = adata.obs[["x", "y"]].values

# === STEP 4: Preprocessing ===
def preprocess(adata):
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    return adata

adata_vis = preprocess(adata_vis)
adata_hd = preprocess(adata_hd)

# === STEP 5: Apply KMeans with fixed cluster count ===
def kmeans_cluster(adata, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(adata.obsm["X_pca"])
    adata.obs["kmeans"] = pd.Categorical([f"C{i}" for i in clusters])
    return adata

adata_vis = kmeans_cluster(adata_vis, n_clusters)
adata_hd = kmeans_cluster(adata_hd, n_clusters)

# === STEP 6: UMAP plots ===
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sc.pl.umap(adata_vis, color="kmeans", title="Visium – 12 Clusters", ax=axes[0], show=False)
sc.pl.umap(adata_hd, color="kmeans", title="Visium HD – 12 Clusters", ax=axes[1], show=False)
plt.suptitle("KMeans Clustering (UMAP View)", fontsize=14)
plt.tight_layout()
plt.show()

# === STEP 7: Spatial plots ===
sc.pl.spatial(adata_vis, color="kmeans", title="Visium – 12 Clusters (Spatial)", spot_size=30)
sc.pl.spatial(adata_hd, color="kmeans", title="Visium HD – 12 Clusters (Spatial)", spot_size=5)