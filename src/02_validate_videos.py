# src/02_validate_videos.py

import cv2
import pandas as pd
from tqdm import tqdm

IN_PATH = "data/processed/en_metadata_merged.csv"
OUT_PATH = "data/processed/en_metadata_validated.csv"

df = pd.read_csv(IN_PATH)

records = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    path = row["video_path"]
    record = row.to_dict()

    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        record["readable"] = False
        record["fps"] = None
        record["num_frames"] = None
        record["duration_sec"] = None
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frames / fps if fps and fps > 0 else None

        record["readable"] = True
        record["fps"] = fps
        record["num_frames"] = frames
        record["duration_sec"] = duration

    cap.release()
    records.append(record)

out = pd.DataFrame(records)

print("Total:", len(out))
print("Readable:", out["readable"].sum())
print("Unreadable:", (~out["readable"]).sum())

out = out[out["readable"]].copy()
out.to_csv(OUT_PATH, index=False)

print("Saved:", OUT_PATH)