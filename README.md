# Medical GRAM vs CLIP Project

This project implements a PyTorch/HuggingFace pipeline for comparing different contrastive learning approaches on a medical vision-language task, specifically using a MIMIC-CXR-style dataset. It trains a multimodal model to align chest X-rays with their corresponding radiology reports.

## Architecture

The model architecture is a dual-encoder multimodal network designed to align image and text modalities in a shared embedding space. Here is a clear description of the architecture:

1. **Vision Pathway**: 
   - Takes a chest X-ray image as input.
   - Passes through a **Vision Encoder** (typically a ViT like `google/vit-base-patch16-224` or a CNN).
   - The CLS token (or pooled output) is passed through an **Image Projector** (a linear layer) and L2 normalized to produce `image_embeds`.

2. **Text Pathway (Active)**: 
   - Takes a radiology report as input.
   - Passes through a **Text Encoder** (typically `NeuML/pubmedbert-base-embeddings`) that is actively fine-tuned.
   - The CLS token is passed through a **Text Projector** (a linear layer) and L2 normalized to produce `text_embeds`.

3. **Text Pathway (Frozen Target)**:
   - Takes the same radiology report as input.
   - Passes through a **Frozen Target Text Encoder** (a copy of the Text Encoder with `requires_grad=False`).
   - The CLS token is L2 normalized to produce `raw_text_embeds`. This provides stable, ground-truth semantic representations of the reports.

4. **Loss Computation**: 
   - The `image_embeds` and `text_embeds` are compared using either standard dot-product similarity (InfoNCE/CLIP) or volume computation based on Gram matrices (GRAM).
   - To prevent "false negative" penalties for distinct but semantically identical reports, soft targets are computed using the pairwise similarities of the `raw_text_embeds` (MedCLIP approach).
   - Available loss variants: `clip` (standard InfoNCE), `gram` (GRAM volume + hard targets), `medclip` (dot-product + soft targets), `gram_med` (GRAM volume + soft targets).

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
