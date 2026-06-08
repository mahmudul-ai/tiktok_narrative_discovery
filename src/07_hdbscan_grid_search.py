# src/07_hdbscan_grid_search.py

import os
import numpy as np
import pandas as pd
import hdbscan
from sklearn.metrics import silhouette_score, davies_bouldin_score

X = np.load("outputs/clusters/en_umap_15d_for_clustering.npy")

results = []

for min_cluster_size in [10, 15, 20, 25, 30, 40, 50]:
    for min_samples in [5, 10, 15, 20]:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method="eom"
        )

        labels = clusterer.fit_predict(X)

        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_ratio = float((labels == -1).sum() / len(labels))

        valid = labels != -1

        sil = None
        db = None

        if valid.sum() > 2 and len(set(labels[valid])) > 1:
            sil = silhouette_score(X[valid], labels[valid])
            db = davies_bouldin_score(X[valid], labels[valid])

        results.append({
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "num_clusters": num_clusters,
            "noise_ratio": noise_ratio,
            "silhouette": sil,
            "davies_bouldin": db
        })

out = pd.DataFrame(results)
os.makedirs("outputs/clusters", exist_ok=True)
out.to_csv("outputs/clusters/en_hdbscan_grid_search.csv", index=False)

print(out.sort_values(["num_clusters", "noise_ratio"], ascending=[False, True]))