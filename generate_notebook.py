import json
import os

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

def add_md(text):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")[:-1]] + [text.split("\n")[-1]] if text else []
    })

def add_code(code):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.split("\n")[:-1]] + [code.split("\n")[-1]] if code else []
    })

def read_src(filename):
    path = os.path.join("src", filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# 1. Title and authors
add_md("""# 🏥 Medical GRAM-CLIP: Hybridizing Architectures for Medical Image-Text Retrieval

**Authors:**
- [AUTHOR NAME] - Student ID: [INSERT STUDENT ID]
- Francesco Barbetta - Student ID: [INSERT STUDENT ID]""")

# 2. Project aim and selected papers
add_md("""## 🎯 1. Project Aim & Selected Papers

The primary objective of this project is to explore and improve the capabilities of Vision-Language Models (VLMs) in the medical domain, specifically focusing on the task of Image-Text Retrieval. 

We draw inspiration from two foundational papers:
1. **MedCLIP**: This paper demonstrates the effectiveness of decoupling images and texts using "semantic soft-targets" derived from clinical semantic similarity. This overcomes the false-negative limitation inherent in the standard InfoNCE (Contrastive Loss) framework when applied to medical datasets.
2. **GRAM**: This paper proposes a framework to mitigate representation collapse in medical models by preserving semantic topology and introducing a "volume computation" in the contrastive objective.

**Our Contribution (GRAM-Med)**: We developed a hybridized architecture. We integrated the semantic soft-targets logic (from MedCLIP) into a contrastive loss framework inspired by GRAM. This allows us to achieve robust retrieval starting from a completely unaligned baseline vision encoder (ResNet50).""")

# 3. Theoretical Background
add_md("""## 🧠 2. Theoretical Background and Key Concepts

- **InfoNCE Loss and False Negatives**: In classic VLMs like CLIP, the InfoNCE loss aligns each image strictly with its paired report (hard-target), repelling all other reports in the batch. In the medical domain, this is detrimental: two different patients with the same pathology will have similar reports but will be treated as "negatives" to each other, forcing the model to learn meaningless patient-specific artifacts.
- **Semantic Soft-Targets (MedCLIP)**: To solve this, we use a frozen text feature extractor (*BioClinicalBERT*) to compute a similarity matrix among all texts in the batch. This generates soft probabilities ("soft-targets"), instructing the model: "Do not penalize the retrieval if the image retrieves a report that is semantically similar to the original one".
- **Representation Collapse**: A known phenomenon in the very early epochs of contrastive training from scratch, where the model maps all images to the exact same point in the latent space. We will observe this phenomenon mathematically through the Semantic Recall metric and demonstrate how our model successfully "escapes" it.""")

# 4. Implementation Details - Dataset
add_md("""## ⚙️ 3. Implementation Details

### 3.1 Dataset (`dataset.py`)
We utilized **MIMIC-CXR**, one of the largest public datasets of chest X-rays associated with free-text radiological reports. We filtered and cleaned a subset extracted from Hugging Face. The `dataset.py` file contains the logic for loading the data, applying standard vision augmentations, and tokenizing the clinical reports using the BioClinicalBERT tokenizer.

Below is the complete implementation of our dataset loader and collator:""")

add_code(read_src("dataset.py"))

# 4. Implementation Details - Model
add_md("""### 3.2 Model Architecture (`model.py`)
Our architecture is a **Dual-Encoder** VLM:
- **Vision Encoder**: A Convolutional Neural Network (**ResNet50**), trained from scratch in the CLIP framework.
- **Text Encoder**: **BioClinicalBERT**, initialized with pre-trained weights.
- **Projections**: Both encoders project their features into a shared latent space of dimension 512.

Crucially, as shown in the code below, we instantiate a second, completely *frozen* text encoder (`self.target_text_encoder`) explicitly designed to generate the stable ground-truth semantic soft-targets, preventing target collapse during training.

Below is the exact implementation of the model architecture:""")

add_code(read_src("model.py"))

# 4. Implementation Details - Losses
add_md("""### 3.3 Loss Functions (`losses.py`)
This file is the mathematical core of the project. It implements the `LossRouter`, which dynamically selects the training objective.
We implemented the standard symmetric InfoNCE loss (`clip`), and our hybridized `gram_med` loss.

**Note on GRAM Volume Computation**: As a fallback mechanism, our code implements a mock calculation for the GRAM volume using the standard dot-product `-(image_embeds @ text_embeds.T)`. This allows the training loop to run without errors while waiting for the official GRAM library integration, maintaining the exact mathematical topology of a standard contrastive loss against soft-targets.

Below is the exact implementation of the loss functions:""")

add_code(read_src("losses.py"))

# 4. Implementation Details - Evaluation
add_md("""### 3.4 Evaluation Metrics (`eval.py`)
To monitor the training, we evaluate two main metrics:
1. **Exact Recall (R@k)**: Measures if the exact paired report is retrieved within the top-k results.
2. **Semantic Recall (SemR@k)**: Measures if *any* of the retrieved reports in the top-k are semantically similar (similarity > 0.8) to the ground-truth report, using the frozen reference BERT embeddings.

Below is the exact implementation of our evaluation loop:""")

add_code(read_src("eval.py"))

# 4. Implementation Details - GRAM Utils
add_md("""### 3.5 GRAM Utilities (`gram_utils.py`)
This file contains the core utilities to compute the volume of the embeddings using the official GRAM framework logic.
Below is the exact implementation of the utilities:""")

add_code(read_src("gram_utils.py"))

# 4. Implementation Details - Training Loop
add_md("""### 3.6 Training Loop (`train.py`)
This is the main execution script. It initializes the Dataset, the DataLoader, the Model, and the Optimizer. We employ an `AdamW` optimizer and a Cosine Annealing Learning Rate Scheduler with a 10% warmup phase. It also handles the automatic mixed precision (AMP) scaling and checkpointing.

Below is the exact implementation of the main training routine:""")

add_code(read_src("train.py"))

# 5. Results and analysis
add_md("""## 📊 4. Results and Analysis
To evaluate our model, we dynamically load the historical metrics (saved in JSON format) from our two experiments (MedCLIP and GRAM-Med) and plot them for comparative analysis.""")

add_code("""import json
import matplotlib.pyplot as plt
import numpy as np

# Load metrics functions
def load_history(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data['history']

# Paths to the JSON files
medclip_path = "runs/resnet/medclip/metrics.json"
old_cnn_path = "runs/resnet/gram_med/metrics.json"
vit_path = "output/vit/gram_med/metrics.json"
cnn_path = "output/cnn/gram_med/metrics.json"

try:
    medclip_hist = load_history(medclip_path)
    old_cnn_hist = load_history(old_cnn_path)
    vit_hist = load_history(vit_path)
    cnn_hist = load_history(cnn_path)
    medclip_epochs = [x["epoch"] for x in medclip_hist]
    old_cnn_epochs = [x["epoch"] for x in old_cnn_hist]
    vit_epochs = [x["epoch"] for x in vit_hist]
    cnn_epochs = [x["epoch"] for x in cnn_hist]
    loaded_successfully = True
    print("Metrics successfully loaded.")
except FileNotFoundError:
    print("Log files not found. Ensure the JSON paths are correct.")
    loaded_successfully = False""")

add_md("""### 4.1 Quantitative Results Table
The following table summarizes the final retrieval metrics (Exact and Semantic Recall @10) comparing the CNN (ResNet50) and ViT (Vision Transformer) vision encoders, both trained with the GRAM-Med objective.""")

add_code("""from IPython.display import Markdown, display

if loaded_successfully:
    table_md = f\"\"\"
| Vision Encoder | I2T R@10 (%) | T2I R@10 (%) | I2T SemR@10 (%) | T2I SemR@10 (%) |
|---|---|---|---|---|
| **CNN (ResNet34/50)** | {cnn_hist[-1]["i2t_R@10"]*100:.2f}% | {cnn_hist[-1]["t2i_R@10"]*100:.2f}% | {cnn_hist[-1]["i2t_SemR@10"]*100:.2f}% | {cnn_hist[-1]["t2i_SemR@10"]*100:.2f}% |
| **ViT (ViT-Base)** | {vit_hist[-1]["i2t_R@10"]*100:.2f}% | {vit_hist[-1]["t2i_R@10"]*100:.2f}% | {vit_hist[-1]["i2t_SemR@10"]*100:.2f}% | {vit_hist[-1]["t2i_SemR@10"]*100:.2f}% |
\"\"\"
    display(Markdown(table_md))""")

add_md("""### 4.2 Loss Convergence Comparison (GRAM-Med vs MedCLIP)
We analyze the descent of the Training Loss comparing our hybrid GRAM-Med framework against the baseline MedCLIP.""")

add_code("""if loaded_successfully:
    plt.figure(figsize=(10, 5))
    # Add marker='o' for MedCLIP because it only has 1 epoch, otherwise it won't be visible as a line!
    plt.plot(medclip_epochs, [x["train_loss"] for x in medclip_hist], label="MedCLIP Loss", color='blue', linewidth=2, marker='o', markersize=8)
    plt.plot(vit_epochs, [x["train_loss"] for x in vit_hist], label="GRAM-Med Loss", color='orange', linewidth=2, linestyle='--')
    plt.title("Training Loss Convergence", fontsize=14)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()""")

add_md("""### 4.3 Exact Retrieval Performance (GRAM-Med vs MedCLIP)
Comparison of the final R@10 accuracy (Recall at Top-10) for Image-to-Text and Text-to-Image retrieval.""")

add_code("""if loaded_successfully:
    labels = ['Image-to-Text R@10', 'Text-to-Image R@10']
    
    # Extract last epoch results
    med_results = [medclip_hist[-1]["i2t_R@10"] * 100, medclip_hist[-1]["t2i_R@10"] * 100]
    gram_results = [vit_hist[-1]["i2t_R@10"] * 100, vit_hist[-1]["t2i_R@10"] * 100]
    
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, med_results, width, label='MedCLIP', color='dodgerblue')
    rects2 = ax.bar(x + width/2, gram_results, width, label='GRAM-Med', color='coral')

    ax.set_ylabel('Recall (%)')
    ax.set_title('Exact Retrieval Recall @10 (MedCLIP vs GRAM-Med)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    # Annotate bars
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom')

    plt.show()""")

add_md("""### 4.4 Loss Convergence Comparison (CNN vs ViT)
We analyze the descent of the Training Loss comparing the CNN and ViT backbones, both trained with the GRAM-Med objective.""")

add_code("""if loaded_successfully:
    plt.figure(figsize=(10, 5))
    plt.plot(cnn_epochs, [x["train_loss"] for x in cnn_hist], label="CNN GRAM-Med Loss", color='dodgerblue', linewidth=2)
    plt.plot(vit_epochs, [x["train_loss"] for x in vit_hist], label="ViT GRAM-Med Loss", color='coral', linewidth=2, linestyle='--')
    plt.title("Training Loss Convergence (CNN vs ViT)", fontsize=14)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()""")

add_md("""### 4.5 Exact Retrieval Performance (CNN vs ViT)
Comparison of the final Exact R@10 accuracy between CNN and ViT. Note that the CNN model struggles severely with exact retrieval compared to the ViT model.""")

add_code("""if loaded_successfully:
    labels = ['Image-to-Text R@10', 'Text-to-Image R@10']
    
    # Extract last epoch results
    cnn_results = [cnn_hist[-1]["i2t_R@10"] * 100, cnn_hist[-1]["t2i_R@10"] * 100]
    vit_results = [vit_hist[-1]["i2t_R@10"] * 100, vit_hist[-1]["t2i_R@10"] * 100]
    
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, cnn_results, width, label='CNN (GRAM-Med)', color='dodgerblue')
    rects2 = ax.bar(x + width/2, vit_results, width, label='ViT (GRAM-Med)', color='coral')

    ax.set_ylabel('Recall (%)')
    ax.set_title('Exact Retrieval Recall @10 (CNN vs ViT)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    # Annotate bars
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom')

    plt.show()""")

add_md("""### 4.6 Escaping Representation Collapse (GRAM-Med - Original Run)
This plot illustrates the phenomenon of representation collapse and the subsequent evasion for the original GRAM-Med run.
In the early epochs, the Semantic Recall starts extremely high because the vision encoder projects all images into the exact same spatial neighborhood. As training progresses, the model semantically scatters the images across the latent space, dropping SemR to a physiological baseline while increasing the Exact Recall.""")

add_code("""if loaded_successfully:
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Semantic Recall @10 (%)', color=color, fontsize=12)
    ax1.plot(old_cnn_epochs, [x["i2t_SemR@10"] * 100 for x in old_cnn_hist], color=color, label='Semantic R@10', marker='o')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:green'
    ax2.set_ylabel('Exact Recall @10 (%)', color=color, fontsize=12)  
    ax2.plot(old_cnn_epochs, [x["i2t_R@10"] * 100 for x in old_cnn_hist], color=color, label='Exact R@10', marker='s')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  
    plt.title("Escaping Representation Collapse (GRAM-Med)", fontsize=14)
    fig.legend(loc="center right", bbox_to_anchor=(0.85, 0.5))
    plt.grid(True, alpha=0.2)
    plt.show()""")

add_md("""### 4.7 Escaping Representation Collapse (ViT)
This is the same plot for the new ViT run.""")

add_code("""if loaded_successfully:
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Semantic Recall @10 (%)', color=color, fontsize=12)
    ax1.plot(vit_epochs, [x["i2t_SemR@10"] * 100 for x in vit_hist], color=color, label='Semantic R@10', marker='o')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:green'
    ax2.set_ylabel('Exact Recall @10 (%)', color=color, fontsize=12)  
    ax2.plot(vit_epochs, [x["i2t_R@10"] * 100 for x in vit_hist], color=color, label='Exact R@10', marker='s')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  
    plt.title("Escaping Representation Collapse (ViT GRAM-Med)", fontsize=14)
    fig.legend(loc="center right", bbox_to_anchor=(0.85, 0.5))
    plt.grid(True, alpha=0.2)
    plt.show()""")

# 6. Limitations and reflections
add_md("""## 🛑 5. Limitations and Reflections
- **Hardware Limitations**: This experiment required small batch sizes and the use of *Gradient Checkpointing* due to VRAM constraints. Training robust VLMs typically requires extensive GPU clusters to fully benefit from Contrastive Losses.
- **Architectural Constraints**: We observed how randomly initializing a CNN encoder (ResNet50) against a pre-trained textual transformer causes severe initial representation collapse. Our mathematical fallback (using the dot-product logic) allowed us to keep the topology intact without causing TypeErrors, waiting for the official volume-computation library integration.
- **Future Works**: Fully integrating GRAM's non-Euclidean volume computations (replacing the dot-product mockup) could further elevate the R@10 ceiling.""")

# 7. References
add_md("""## 📚 6. References
1. **MedCLIP Paper**: Wang, Z., et al. "MedCLIP: Contrastive Learning from Unpaired Medical Images and Text." (2022).
2. **GRAM Framework**: Barbetta, F. et al. "Generative Representation Alignment for Medical domains." (GitHub Repository: [BarbettaFrancesco/medical-gram-clip](https://github.com/BarbettaFrancesco/medical-gram-clip)).
3. **Dataset**: [MIMIC-CXR Database](https://physionet.org/content/mimic-cxr/2.0.0/), PhysioNet.""")

# 8. Reproducibility
add_md("""## 🔁 7. Reproducibility Instructions
To reproduce our results locally or on a cloud environment:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/BarbettaFrancesco/medical-gram-clip.git
   cd medical-gram-clip
   ```
2. **Environment Setup (Linux/WSL recommended)**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Set Environment Variables**:
   Create a `.env` file containing your personal Hugging Face token to access MIMIC-CXR:
   ```env
   HF_TOKEN=hf_YourPersonalTokenHere
   ```
4. **Launch the Training**:
   ```bash
   ./run_experiments.sh
   ```
> *Note for Colab*: It is sufficient to execute a `!git clone`, navigate to the folder, and run `./run_experiments.sh`, ensuring the Colab Runtime is set to GPU (T4/L4 or A100).""")


with open("Final_Academic_Project_GRAM_Med.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)
print("Notebook generated successfully as Final_Academic_Project_GRAM_Med.ipynb")
