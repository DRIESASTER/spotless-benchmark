import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import squidpy as sq

# === CONFIG ===
spatial_dir = "visium_spatial/"  # Folder with image, .parquet, .json, etc.
proportions_file = "deconv_proportions/cell2loc/proportions_cell2location_part_0"
n_clusters = 12

# === STEP 1: Load image-backed AnnData with correct spatial info (Squidpy handles .parquet etc.)
adata_img = sq.read.visium(path=spatial_dir, count_file=None)
adata_img.var_names_make_unique()

# === STEP 2: Load deconvolution proportions
df = pd.read_csv(proportions_file, sep="\t", index_col=0)
df.index = df.index.astype(str)

# === STEP 3: Align to barcodes present in both AnnData and deconvolution
adata_img = adata_img[df.index]  # subset and reorder
adata_img.X = df.values
adata_img.var_names = df.columns

# === STEP 4: Preprocess and cluster
sc.pp.normalize_total(adata_img)
sc.pp.log1p(adata_img)
sc.pp.scale(adata_img)
sc.pp.pca(adata_img)
sc.pp.neighbors(adata_img)
sc.tl.umap(adata_img)

kmeans = KMeans(n_clusters=n_clusters, random_state=42)
adata_img.obs["kmeans"] = pd.Categorical([f"C{i}" for i in kmeans.fit_predict(adata_img.obsm["X_pca"])])

# === STEP 5: Plot clusters overlaid on real tissue image
sq.pl.spatial_scatter(
    adata_img,
    color="kmeans",
    title="Visium – 12 Clusters Over Tissue",
    size=1.5,
    library_id=adata_img.uns["spatial"].keys()[0],  # auto-detect correct image set
    img=True
)