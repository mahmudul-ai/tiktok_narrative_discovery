# src/04_normalize_embeddings.py

import os
import numpy as np
from sklearn.preprocessing import normalize

IN_PATH = "outputs/embeddings/en_video_embeddings_videomae.npy"
OUT_PATH = "outputs/embeddings/en_video_embeddings_videomae_norm.npy"

X = np.load(IN_PATH)
X_norm = normalize(X, norm="l2")

np.save(OUT_PATH, X_norm)

print("Original:", X.shape)
print("Saved normalized embeddings:", OUT_PATH)