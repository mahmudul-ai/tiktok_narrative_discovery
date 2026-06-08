# src/06_hdbscan_clustering.py

import os
import numpy as np
import pandas as pd
import hdbscan
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score

X_CLUSTER_PATH = "outputs/clusters/en_umap_15d_for_clustering.npy"
X_2D_PATH = "outputs/clusters/en_umap_2d.npy"
META_PATH = "outputs/embeddings/en_embedding_metadata.csv"

OUT_DIR = "outputs/clusters"
FIG_DIR = "outputs/figures"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

X = np.load(X_CLUSTER_PATH)
X_2d = np.load(X_2D_PATH)
meta = pd.read_csv(META_PATH)

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=25,
    min_samples=10,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True
)

labels = clusterer.fit_predict(X)

meta["cluster_id"] = labels
meta["cluster_probability"] = clusterer.probabilities_

out_csv = os.path.join(OUT_DIR, "en_video_native_clusters.csv")
meta.to_csv(out_csv, index=False)

num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
noise_count = int((labels == -1).sum())
noise_ratio = noise_count / len(labels)

print("Number of videos:", len(labels))
print("Number of clusters:", num_clusters)
print("Noise count:", noise_count)
print("Noise ratio:", noise_ratio)

valid = labels != -1
if valid.sum() > 2 and len(set(labels[valid])) > 1:
    sil = silhouette_score(X[valid], labels[valid])
    db = davies_bouldin_score(X[valid], labels[valid])
    print("Silhouette:", sil)
    print("Davies-Bouldin:", db)

plt.figure(figsize=(11, 9))
plt.scatter(
    X_2d[:, 0],
    X_2d[:, 1],
    c=labels,
    s=10,
    cmap="tab20"
)
plt.title("Video-Native Narrative Clusters: English TikTok")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "en_video_native_hdbscan_clusters.png"), dpi=300)

# cluster size summary
summary = (
    meta.groupby("cluster_id")
    .size()
    .reset_index(name="num_videos")
    .sort_values("num_videos", ascending=False)
)

summary.to_csv(os.path.join(OUT_DIR, "en_cluster_size_summary.csv"), index=False)

print("Saved:", out_csv)