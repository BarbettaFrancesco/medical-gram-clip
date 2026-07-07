python src/train.py \
  --csv_path data/metadata.csv \
  --image_root data/images \
  --output_dir runs/gram \
  --loss_type gram \
  --gram_repo_path external/GRAM \
  --epochs 3 \
  --batch_size 8
