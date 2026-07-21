import json
import matplotlib.pyplot as plt

def load_history(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data['history']

medclip_path = "output/vit/medclip/metrics.json"
vit_path = "output/vit/gram_med/metrics.json"

medclip_hist = load_history(medclip_path)
vit_hist = load_history(vit_path)

medclip_epochs = [x["epoch"] for x in medclip_hist]
vit_epochs = [x["epoch"] for x in vit_hist]

plt.figure(figsize=(10, 5))
plt.plot(medclip_epochs, [x["train_loss"] for x in medclip_hist], label="MedCLIP Loss", color='blue', linewidth=2)
plt.plot(vit_epochs, [x["train_loss"] for x in vit_hist], label="GRAM-Med Loss", color='orange', linewidth=2, linestyle='--')
plt.title("Training Loss Convergence", fontsize=14)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('debug.png')
