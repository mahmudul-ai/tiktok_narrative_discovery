# src/09_generate_review_sheet.py

import pandas as pd

clusters = pd.read_csv("outputs/clusters/en_video_native_clusters.csv")
reps = pd.read_csv("outputs/representatives/en_cluster_representatives.csv")

summary = []

for cluster_id in sorted(clusters["cluster_id"].unique()):
    if cluster_id == -1:
        continue

    cluster_df = clusters[clusters["cluster_id"] == cluster_id]
    rep_df = reps[reps["cluster_id"] == cluster_id]

    row = {
        "cluster_id": cluster_id,
        "num_videos": len(cluster_df),
        "avg_cluster_probability": cluster_df["cluster_probability"].mean(),
        "manual_narrative_label": "",
        "manual_notes": ""
    }

    for i, (_, rep) in enumerate(rep_df.head(5).iterrows(), start=1):
        row[f"rep_{i}_id"] = rep["id"]
        row[f"rep_{i}_path"] = rep["representative_path"]
        row[f"rep_{i}_title"] = rep.get("title", "")
        row[f"rep_{i}_url"] = rep.get("webpage_url", "")

    summary.append(row)

out = pd.DataFrame(summary)
out.to_csv("outputs/clusters/en_cluster_review_sheet.csv", index=False)

print("Saved review sheet.")