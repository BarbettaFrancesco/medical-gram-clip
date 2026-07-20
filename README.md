# Medical GRAM vs CLIP Project

Minimal PyTorch/HuggingFace pipeline for comparing a CLIP InfoNCE baseline against GRAM on a MIMIC-CXR-style Kaggle subset.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure your settings in `train_config.json`:
   - Adjust `"loss_type"` (`"clip"`, `"gram"`, `"medclip"`, or `"gram_med"`) and other training hyperparameters.

## Running Training

To start training:
```bash
python src/train.py --config train_config.json
```
This will run the training loop and evaluate the model using standard and semantic Recall@K. Metrics will be saved in your specified output directory (e.g. `output/vit/gram_med/metrics.json`).

## Plotting Results

To visualize and compare the metrics between two different runs (e.g. CLIP and GRAM), run:
```bash
python plot_results.py --clip_metrics output/vit/clip/metrics.json --gram_metrics output/vit/gram/metrics.json
```
This will generate and save comparison line plots for standard Recall@K metrics.
