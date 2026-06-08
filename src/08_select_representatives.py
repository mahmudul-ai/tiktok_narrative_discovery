# src/08_select_representatives.py

import os
import shutil
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances

CLUSTERS_PATH = "outputs/clusters/en_video_native_clusters.csv"
EMB_PATH = "outputs/embeddings/en_video_embeddings_videomae_norm.npy"

OUT_DIR = "outputs/representatives"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CLUSTERS_PATH)
X = np.load(EMB_PATH)

representatives = []

for cluster_id in sorted(df["cluster_id"].unique()):
    if cluster_id == -1:
        continue

    cluster_indices = df.index[df["cluster_id"] == cluster_id].tolist()
    cluster_embeddings = X[cluster_indices]

    centroid = cluster_embeddings.mean(axis=0, keepdims=True)
    distances = cosine_distances(cluster_embeddings, centroid).flatten()

    ranked = sorted(
        zip(cluster_indices, distances),
        key=lambda x: x[1]
    )

    top_k = ranked[:10]

    cluster_dir = os.path.join(OUT_DIR, f"cluster_{cluster_id}")
    os.makedirs(cluster_dir, exist_ok=True)

    for rank, (idx, distance) in enumerate(top_k, start=1):
        row = df.iloc[idx].to_dict()
        src = row["video_path"]

        dst = os.path.join(
            cluster_dir,
            f"rank_{rank}_id_{row['id']}.mp4"
        )

        if os.path.exists(src):
            shutil.copy(src, dst)

        row["representative_rank"] = rank
        row["distance_to_centroid"] = distance
        row["representative_path"] = dst

        representatives.append(row)

rep_df = pd.DataFrame(representatives)
rep_df.to_csv(os.path.join(OUT_DIR, "en_cluster_representatives.csv"), index=False)

print("Saved representatives:", len(rep_df))