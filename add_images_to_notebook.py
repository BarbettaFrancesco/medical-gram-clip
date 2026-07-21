import json

path = '/home/gheras/medical_gram_clip_project/ENGLISH_COMPREHENSIVE_NOTEBOOK_V2.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        if len(cell['source']) > 0 and 'Theoretical Background and Key Concepts' in cell['source'][0]:
            # Add the images at the end of this cell
            cell['source'][-1] = cell['source'][-1] + "\n"
            cell['source'].extend([
                "\n",
                "---\n",
                "### 🖼️ Architecture & Concept Visualized\n",
                "**1. Dual-Encoder Architecture with Frozen Target**\n",
                "\n",
                "![Model Architecture](arch_diagram.png)\n",
                "\n",
                "**2. Concept of Semantic Soft-Targets vs Hard-Targets**\n",
                "\n",
                "![Loss Matrices Comparison](loss_diagram.png)\n",
                "\n",
                "---"
            ])

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
