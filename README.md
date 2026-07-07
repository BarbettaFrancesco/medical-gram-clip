# Medical GRAM vs CLIP Project

Minimal PyTorch/HuggingFace pipeline for comparing a CLIP InfoNCE baseline against GRAM on a MIMIC-CXR-style Kaggle subset.

## Setup

```bash
git clone https://github.com/ispamm/GRAM external/GRAM
pip install -r requirements.txt
```

## Train CLIP baseline

```bash
python train.py \
  --csv_path /path/to/metadata.csv \
  --image_root /path/to/images \
  --output_dir runs/clip \
  --loss_type clip \
  --epochs 3 \
  --batch_size 8 \
  --grad_accum_steps 4
```

## Train GRAM

```bash
python train.py \
  --csv_path /path/to/metadata.csv \
  --image_root /path/to/images \
  --output_dir runs/gram \
  --loss_type gram \
  --epochs 3 \
  --batch_size 8 \
  --grad_accum_steps 4
```

## Plot comparison

```bash
python plot_results.py --clip_metrics runs/clip/metrics.json --gram_metrics runs/gram/metrics.json --output comparison.png
```

Important: the GRAM adapter may need a tiny import edit depending on the exact class/function names in the official repository.