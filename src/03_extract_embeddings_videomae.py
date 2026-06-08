# src/03_extract_embeddings_videomae.py

import os
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import VideoMAEImageProcessor, VideoMAEModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

METADATA_PATH = "data/processed/en_metadata_validated.csv"
OUT_DIR = "outputs/embeddings"
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_NAME = "MCG-NJU/videomae-base"

NUM_FRAMES = 16
IMAGE_SIZE = 224


def sample_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise ValueError("No frames found")

    indices = np.linspace(0, total_frames - 1, num_frames).astype(int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()

        if not success:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise ValueError("Could not sample frames")

    # pad if fewer frames were read
    while len(frames) < num_frames:
        frames.append(frames[-1])

    return frames[:num_frames]


def main():
    df = pd.read_csv(METADATA_PATH)

    processor = VideoMAEImageProcessor.from_pretrained(MODEL_NAME)
    model = VideoMAEModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    embeddings = []
    kept_rows = []
    failed = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        video_path = row["video_path"]

        try:
            frames = sample_frames(video_path, NUM_FRAMES)

            inputs = processor(frames, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            # Mean-pool token embeddings into one video vector
            emb = outputs.last_hidden_state.mean(dim=1).squeeze(0)
            emb = emb.detach().cpu().numpy()

            embeddings.append(emb)
            kept_rows.append(row.to_dict())

        except Exception as e:
            failed.append({
                "id": row["id"],
                "video_path": video_path,
                "error": str(e)
            })

    embeddings = np.vstack(embeddings)
    kept_df = pd.DataFrame(kept_rows)
    failed_df = pd.DataFrame(failed)

    np.save(os.path.join(OUT_DIR, "en_video_embeddings_videomae.npy"), embeddings)
    kept_df.to_csv(os.path.join(OUT_DIR, "en_embedding_metadata.csv"), index=False)
    failed_df.to_csv(os.path.join(OUT_DIR, "en_embedding_failed.csv"), index=False)

    print("Embedding shape:", embeddings.shape)
    print("Saved embeddings to:", OUT_DIR)
    print("Failed videos:", len(failed_df))


if __name__ == "__main__":
    main()