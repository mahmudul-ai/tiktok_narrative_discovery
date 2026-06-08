# TikTok Video-Native Narrative Discovery

This repository implements a video-native narrative discovery pipeline for short-form TikTok videos. The initial version focuses on English MP4 videos only and avoids using textual metadata during clustering. The purpose is to test whether recurring strategic narratives can be discovered directly from the visual-temporal structure of videos.

## Project Goal

The core research question is:

> Can TikTok narratives be discovered from video representations without first converting videos into transcripts, captions, or text summaries?

Most conventional approaches follow a text-first pipeline:

```text
video → transcript/caption → text embedding/topic model → topics
```

This project instead follows a video-native pipeline:

```text
MP4 video → video encoder → video embedding → UMAP → HDBSCAN → representative videos → narrative interpretation
```

The first proof-of-concept uses only English MP4 files. Metadata is used for tracking, validation, and post-hoc interpretation, but not for clustering.

---

## Current Pipeline

```text
English TikTok MP4 files
        ↓
Metadata merging and validation
        ↓
Video readability validation
        ↓
VideoMAE embedding extraction
        ↓
Embedding normalization
        ↓
UMAP dimensionality reduction
        ↓
HDBSCAN clustering
        ↓
Representative video selection
        ↓
Manual / VLM-assisted narrative interpretation
```

---

## Repository Structure

```text
tiktok_narrative_discovery/
├── data/
│   ├── raw_metadata/
│   │   ├── *.csv
│   ├── videos_en/
│   │   ├── *.mp4
│   └── processed/
│       ├── en_metadata_merged.csv
│       └── en_metadata_validated.csv
│
├── outputs/
│   ├── embeddings/
│   ├── clusters/
│   ├── representatives/
│   ├── figures/
│   └── labels/
│
├── src/
│   ├── 01_merge_metadata.py
│   ├── 02_validate_videos.py
│   ├── 03_extract_embeddings_videomae.py
│   ├── 04_normalize_embeddings.py
│   ├── 05_umap_projection.py
│   ├── 06_hdbscan_clustering.py
│   ├── 07_hdbscan_grid_search.py
│   ├── 08_select_representatives.py
│   └── 09_generate_review_sheet.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Dataset Assumptions

The current implementation assumes:

1. Videos are stored locally as `.mp4` files.
2. English videos are stored in:

```text
data/videos_en/
```

3. Metadata is stored as multiple CSV files in:

```text
data/raw_metadata/
```

4. Each metadata row contains the following fields:

```text
id
title
uploader
artist
uploader_id
upload_date
view_count
like_count
repost_count
comment_count
description
tags
webpage_url
scraped_at
```

5. Video filenames are assumed to follow:

```text
{id}.mp4
```

Example:

```text
data/videos_en/1234567890.mp4
```

If your video filenames follow a different pattern, update the path-matching logic in `src/01_merge_metadata.py`.

---

## Environment Setup

Create a new Conda environment:

```bash
conda create -n tiktok python=3.12 -y
conda activate tiktok
```

Install dependencies:

```bash
pip install pandas numpy tqdm opencv-python matplotlib scikit-learn umap-learn hdbscan
pip install torch torchvision torchaudio
pip install transformers accelerate decord einops
pip install charset-normalizer chardet
```

Check GPU availability:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Expected output should show:

```text
True
<GPU name>
```

If CUDA is unavailable, embedding extraction will run on CPU and may be very slow.

---

## Step-by-Step Implementation Guide

### Step 1: Merge Metadata

This script merges multiple metadata CSV files, removes duplicate TikTok IDs, creates expected video paths, and checks whether the corresponding MP4 files exist.

```bash
python src/01_merge_metadata.py
```

Input:

```text
data/raw_metadata/*.csv
data/videos_en/*.mp4
```

Output:

```text
data/processed/en_metadata_merged.csv
```

This file contains only rows where the corresponding video exists.

---

### Step 2: Validate Video Files

This script checks whether each MP4 file can be opened and extracts basic video properties such as FPS, frame count, and duration.

```bash
python src/02_validate_videos.py
```

Input:

```text
data/processed/en_metadata_merged.csv
```

Output:

```text
data/processed/en_metadata_validated.csv
```

This file contains only readable videos.

---

### Step 3: Extract Video Embeddings

This script uses VideoMAE to extract one video-level embedding per TikTok video.

```bash
python src/03_extract_embeddings_videomae.py
```

Input:

```text
data/processed/en_metadata_validated.csv
```

Output:

```text
outputs/embeddings/en_video_embeddings_videomae.npy
outputs/embeddings/en_embedding_metadata.csv
outputs/embeddings/en_embedding_failed.csv
```

The embedding file is a NumPy array of shape:

```text
(number_of_videos, embedding_dimension)
```

Each row corresponds to one TikTok video.

---

### Step 4: Normalize Embeddings

This script applies L2 normalization to the video embeddings.

```bash
python src/04_normalize_embeddings.py
```

Input:

```text
outputs/embeddings/en_video_embeddings_videomae.npy
```

Output:

```text
outputs/embeddings/en_video_embeddings_videomae_norm.npy
```

Normalized embeddings are used for UMAP and clustering.

---

### Step 5: Run UMAP Projection

This script creates two UMAP representations:

1. A 2D projection for visualization.
2. A higher-dimensional projection for clustering.

```bash
python src/05_umap_projection.py
```

Input:

```text
outputs/embeddings/en_video_embeddings_videomae_norm.npy
```

Output:

```text
outputs/clusters/en_umap_2d.npy
outputs/clusters/en_umap_15d_for_clustering.npy
outputs/figures/en_umap_2d_video_only.png
```

The 2D plot is useful for visually inspecting whether videos form meaningful groups.

---

### Step 6: Run HDBSCAN Clustering

This script clusters the video embeddings using HDBSCAN.

```bash
python src/06_hdbscan_clustering.py
```

Input:

```text
outputs/clusters/en_umap_15d_for_clustering.npy
outputs/clusters/en_umap_2d.npy
outputs/embeddings/en_embedding_metadata.csv
```

Output:

```text
outputs/clusters/en_video_native_clusters.csv
outputs/clusters/en_cluster_size_summary.csv
outputs/figures/en_video_native_hdbscan_clusters.png
```

The output CSV includes:

```text
cluster_id
cluster_probability
```

Videos assigned to `cluster_id = -1` are treated as noise or outliers.

---

### Step 7: Tune HDBSCAN Parameters

This script runs a simple grid search over HDBSCAN parameters.

```bash
python src/07_hdbscan_grid_search.py
```

Output:

```text
outputs/clusters/en_hdbscan_grid_search.csv
```

Use this file to compare:

```text
number of clusters
noise ratio
silhouette score
Davies-Bouldin score
```

A useful clustering solution should usually avoid both extremes:

```text
too few clusters
too many tiny clusters
very high noise ratio
one giant dominant cluster
```

---

### Step 8: Select Representative Videos

This script selects the top representative videos from each cluster based on cosine distance to the cluster centroid.

```bash
python src/08_select_representatives.py
```

Output:

```text
outputs/representatives/
├── cluster_0/
├── cluster_1/
├── cluster_2/
└── en_cluster_representatives.csv
```

Each cluster folder contains the most central videos for that cluster. These are used for manual or VLM-assisted narrative interpretation.

Small cosine distances are expected for representative videos because they are selected to be close to the cluster centroid.

---

### Step 9: Generate Review Sheet

This script creates a CSV file for manual review and narrative labeling.

```bash
python src/09_generate_review_sheet.py
```

Output:

```text
outputs/clusters/en_cluster_review_sheet.csv
```

This file can be used by the research team to assign:

```text
manual_narrative_label
manual_notes
cluster coherence
visual motifs
emotional framing
political interpretation
```

---

## Recommended Manual Review Protocol

For each discovered cluster:

1. Open the representative videos.
2. Watch the top 5 to 10 representatives.
3. Assign a short narrative label.
4. Record recurring visual motifs.
5. Record emotional framing.
6. Note whether the cluster appears coherent or mostly stylistic.
7. Mark confidence as high, medium, or low.

Suggested fields:

```text
cluster_id
narrative_label
narrative_summary
visual_motifs
emotional_framing
strategic_frame
confidence
notes
```

Example:

```text
Cluster 4
Narrative label: China as a technological leader
Visual motifs: high-speed rail, cities, factories, robotics
Emotional framing: admiration, pride, futurism
Confidence: high
```

---

## Design Choices

### Why video-only first?

The first version intentionally avoids using `title`, `description`, `tags`, or transcripts during clustering. This allows us to test whether the video itself carries enough signal to form narrative groups.

### Why English first?

The English subset is large enough for unsupervised clustering and easier for the team to interpret during the first proof of concept.

### Why VideoMAE?

VideoMAE is accessible, GPU-compatible, and suitable for extracting visual-temporal embeddings without training a model from scratch.

### Why UMAP?

UMAP helps reduce high-dimensional embeddings into a space suitable for visualization and clustering.

### Why HDBSCAN?

HDBSCAN does not require specifying the number of clusters in advance and can mark uncertain videos as noise. This is appropriate for messy social media data.

### Why representative videos?

Instead of interpreting thousands of videos individually, the system selects representative examples from each cluster. This makes expert interpretation scalable.

---

## Current Research Contribution

The current version is a pipeline/system contribution:

> A video-native unsupervised narrative discovery pipeline for TikTok videos that clusters short-form political content directly in video representation space rather than through transcript-first topic modeling.

This can support a computational social science paper if paired with expert validation and narrative interpretation.

---

## Potential Extensions

### 1. Text baseline

Use `title`, `description`, and `tags` to create a text-only clustering baseline.

```text
title/description/tags → sentence embeddings → UMAP → HDBSCAN
```

This allows comparison against the video-only pipeline.

### 2. Metadata-based post-hoc analysis

Use metadata after clustering to analyze:

```text
uploader distribution
engagement statistics
upload-date patterns
common tags
common descriptions
```

### 3. Cross-language extension

After English proof of concept:

```text
English → Chinese → Japanese
```

This can reveal whether similar narrative structures appear across languages.

### 4. Narrative-aware representation learning

Use high-confidence clusters as pseudo-labels to train a small projection head for better narrative separation.

```text
initial clusters → pseudo-labels → projection head → refined embeddings → reclustering
```

This would move the project toward algorithmic novelty.

### 5. Visual motif extraction

Use representative frames or VLM-assisted interpretation to identify recurring visual symbols within each cluster.

---

## Troubleshooting

### CSV parsing error

If you see:

```text
EOF inside string
```

one of the CSV files likely contains malformed text, unmatched quotation marks, or multiline fields. Use the robust CSV loading version of `01_merge_metadata.py`.

### CUDA not available

Check:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

If CUDA is unavailable, reinstall a PyTorch build compatible with your NVIDIA driver.

### Slow tensor creation warning

This warning is not fatal:

```text
Creating a tensor from a list of numpy.ndarrays is extremely slow
```

It can be ignored for the first experiment or fixed later by converting sampled frames to a NumPy array before passing them to the processor.

### Very small representative distances

Small cosine distances for representative videos are normal because they are selected to be close to the cluster centroid.

---

## Reproducibility Notes

For reproducibility, record:

```text
number of videos
number of successfully embedded videos
VideoMAE model name
UMAP parameters
HDBSCAN parameters
number of clusters
noise ratio
manual interpretation protocol
```

Recommended parameters to document:

```text
NUM_FRAMES
UMAP n_neighbors
UMAP n_components
UMAP min_dist
HDBSCAN min_cluster_size
HDBSCAN min_samples
```

---

## Ethical and Research Notes

This project analyzes public social media content. The repository should not include raw TikTok videos, personally identifying information beyond what is necessary for research, or any data that cannot be redistributed.

