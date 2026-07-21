import json
import base64
import os

notebook_path = '/home/gheras/medical_gram_clip_project/ENGLISH_COMPREHENSIVE_NOTEBOOK_V2.ipynb'
arch_img_path = '/home/gheras/medical_gram_clip_project/Gemini_Generated_Image_m1isexm1isexm1is.png'
loss_img_path = '/home/gheras/medical_gram_clip_project/loss_diagram.png'

def get_base64(path):
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

arch_b64 = get_base64(arch_img_path)
loss_b64 = get_base64(loss_img_path)

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        if len(cell['source']) > 0 and '3.2 Model Architecture' in cell['source'][0]:
            # Add to source
            if not any("architecture_final.png" in line for line in cell['source']):
                cell['source'].append("\n")
                cell['source'].append("![Dual Encoder Architecture](attachment:architecture_final.png)\n")
            
            # Add to attachments
            if "attachments" not in cell:
                cell["attachments"] = {}
            cell["attachments"]["architecture_final.png"] = {
                "image/png": arch_b64
            }
            print("Attached architecture diagram.")
        
        elif len(cell['source']) > 0 and '3.3 Loss Functions' in cell['source'][0]:
            # Add to source
            if not any("loss_diagram.png" in line for line in cell['source']):
                cell['source'].append("\n")
                cell['source'].append("![Loss Matrices Comparison](attachment:loss_diagram.png)\n")
            
            # Add to attachments
            if "attachments" not in cell:
                cell["attachments"] = {}
            cell["attachments"]["loss_diagram.png"] = {
                "image/png": loss_b64
            }
            print("Attached loss diagram.")

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print("Images embedded successfully.")
