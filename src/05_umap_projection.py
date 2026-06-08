# src/05_umap_projection.py

import os
import numpy as np
import umap
import matplotlib.pyplot as plt

EMB_PATH = "outputs/embeddings/en_video_embeddings_videomae_norm.npy"

OUT_DIR = "outputs/clusters"
FIG_DIR = "outputs/figures"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

X = np.load(EMB_PATH)

# 2D for visualization
umap_2d = umap.UMAP(
    n_neighbors=30,
    n_components=2,
    min_dist=0.1,
    metric="cosine",
    random_state=42
)

X_2d = umap_2d.fit_transform(X)
np.save(os.path.join(OUT_DIR, "en_umap_2d.npy"), X_2d)

plt.figure(figsize=(10, 8))
plt.scatter(X_2d[:, 0], X_2d[:, 1], s=8)
plt.title("English TikTok Videos: Video-Only UMAP Projection")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "en_umap_2d_video_only.png"), dpi=300)

# 15D for clustering
umap_15d = umap.UMAP(
    n_neighbors=30,
    n_components=15,
    min_dist=0.0,
    metric="cosine",
    random_state=42
)

X_15d = umap_15d.fit_transform(X)
np.save(os.path.join(OUT_DIR, "en_umap_15d_for_clustering.npy"), X_15d)

print("Saved UMAP projections.")