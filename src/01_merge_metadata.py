# src/01_merge_metadata.py

import os
import glob
import pandas as pd

RAW_METADATA_DIR = "data/raw_metadata"
VIDEO_DIR = "data/videos_en"
OUT_PATH = "data/processed/en_metadata_merged.csv"
BAD_FILES_LOG = "data/processed/bad_metadata_files.txt"

os.makedirs("data/processed", exist_ok=True)

csv_files = glob.glob(os.path.join(RAW_METADATA_DIR, "*.csv"))

print(f"Found {len(csv_files)} CSV files.")

dfs = []
bad_files = []

for path in csv_files:
    print(f"Reading: {path}")

    try:
        # First try normal fast read
        df = pd.read_csv(path)
    except Exception as e1:
        print(f"Standard read failed for {path}")
        print(f"Trying robust parser... Error was: {e1}")

        try:
            # More tolerant parser for messy TikTok metadata
            df = pd.read_csv(
                path,
                engine="python",
                on_bad_lines="skip"
            )
        except Exception as e2:
            print(f"Failed completely: {path}")
            print(e2)
            bad_files.append((path, str(e2)))
            continue

    df["source_csv"] = os.path.basename(path)
    dfs.append(df)

if not dfs:
    raise RuntimeError("No CSV files could be read successfully.")

metadata = pd.concat(dfs, ignore_index=True)

print("Columns found:")
print(metadata.columns.tolist())

# Make sure id exists
if "id" not in metadata.columns:
    raise ValueError("Missing required column: id")

# Remove duplicates by TikTok id
metadata = metadata.drop_duplicates(subset=["id"]).reset_index(drop=True)

# Create expected video path
metadata["video_path"] = metadata["id"].astype(str).apply(
    lambda x: os.path.join(VIDEO_DIR, f"{x}.mp4")
)

# Check whether video exists
metadata["video_exists"] = metadata["video_path"].apply(os.path.exists)

print("Total metadata rows:", len(metadata))
print("Videos found:", metadata["video_exists"].sum())
print("Missing videos:", (~metadata["video_exists"]).sum())

metadata_valid = metadata[metadata["video_exists"]].copy()
metadata_valid.to_csv(OUT_PATH, index=False)

print("Saved:", OUT_PATH)

if bad_files:
    with open(BAD_FILES_LOG, "w") as f:
        for path, error in bad_files:
            f.write(f"{path}\n{error}\n\n")

    print(f"Some files could not be read. See: {BAD_FILES_LOG}")