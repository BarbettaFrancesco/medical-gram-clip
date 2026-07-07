#!/bin/bash

set -e

echo "[INFO] Running CLIP smoke test..."
python src/train.py \
  --csv_path dummy_dataset/metadata.csv \
  --image_root dummy_dataset/images \
  --output_dir runs/clip \
  --loss_type clip \
  --epochs 1 \
  --batch_size 2 \
  --max_samples 20 \
  --num_workers 0 \
  --no_save_checkpoint

echo "[INFO] Running GRAM smoke test..."
python src/train.py \
  --csv_path dummy_dataset/metadata.csv \
  --image_root dummy_dataset/images \
  --output_dir runs/gram \
  --loss_type gram \
  --gram_repo_path external/GRAM \
  --epochs 1 \
  --batch_size 2 \
  --max_samples 20 \
  --num_workers 0 \
  --no_save_checkpoint

echo "[INFO] Smoke test completed successfully."
