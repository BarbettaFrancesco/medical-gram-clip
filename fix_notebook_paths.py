import json

notebook_path = '/home/gheras/medical_gram_clip_project/ENGLISH_COMPREHENSIVE_NOTEBOOK_V2.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        for i, line in enumerate(cell['source']):
            if 'runs/resnet/medclip/metrics.json' in line:
                cell['source'][i] = line.replace('runs/resnet/medclip/metrics.json', 'output/vit/medclip/metrics.json')

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print("Notebook paths updated.")
