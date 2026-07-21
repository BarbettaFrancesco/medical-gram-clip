import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def create_arch_diagram():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    # Image path
    ax.add_patch(patches.Rectangle((0.1, 0.7), 0.2, 0.15, fill=True, color='lightblue', ec='black'))
    ax.text(0.2, 0.775, 'Image\n(X-Ray)', ha='center', va='center', fontsize=12, fontweight='bold')
    
    ax.add_patch(patches.Rectangle((0.4, 0.7), 0.2, 0.15, fill=True, color='salmon', ec='black'))
    ax.text(0.5, 0.775, 'Vision Encoder\n(ViT / CNN)', ha='center', va='center', fontsize=12)
    
    ax.add_patch(patches.Rectangle((0.7, 0.7), 0.2, 0.15, fill=True, color='lightgreen', ec='black'))
    ax.text(0.8, 0.775, 'Image\nEmbeddings', ha='center', va='center', fontsize=12)
    
    # Arrows
    ax.annotate('', xy=(0.4, 0.775), xytext=(0.3, 0.775), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate('', xy=(0.7, 0.775), xytext=(0.6, 0.775), arrowprops=dict(arrowstyle="->", lw=2))
    
    # Text path
    ax.add_patch(patches.Rectangle((0.1, 0.4), 0.2, 0.15, fill=True, color='lightblue', ec='black'))
    ax.text(0.2, 0.475, 'Text\n(Report)', ha='center', va='center', fontsize=12, fontweight='bold')
    
    ax.add_patch(patches.Rectangle((0.4, 0.4), 0.2, 0.15, fill=True, color='salmon', ec='black'))
    ax.text(0.5, 0.475, 'Text Encoder\n(PubMedBERT)', ha='center', va='center', fontsize=12)
    
    ax.add_patch(patches.Rectangle((0.7, 0.4), 0.2, 0.15, fill=True, color='lightgreen', ec='black'))
    ax.text(0.8, 0.475, 'Text\nEmbeddings', ha='center', va='center', fontsize=12)
    
    # Arrows
    ax.annotate('', xy=(0.4, 0.475), xytext=(0.3, 0.475), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate('', xy=(0.7, 0.475), xytext=(0.6, 0.475), arrowprops=dict(arrowstyle="->", lw=2))
    
    # Frozen path
    ax.add_patch(patches.Rectangle((0.4, 0.1), 0.2, 0.15, fill=True, color='lightgray', ec='black', hatch='//'))
    ax.text(0.5, 0.175, 'FROZEN Target\nText Encoder', ha='center', va='center', fontsize=12, fontweight='bold')
    
    ax.add_patch(patches.Rectangle((0.7, 0.1), 0.2, 0.15, fill=True, color='gold', ec='black'))
    ax.text(0.8, 0.175, 'Semantic\nSoft-Targets', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Arrows
    ax.annotate('', xy=(0.4, 0.175), xytext=(0.2, 0.4), arrowprops=dict(arrowstyle="->", lw=2, connectionstyle="angle,angleA=0,angleB=90,rad=10"))
    ax.annotate('', xy=(0.7, 0.175), xytext=(0.6, 0.175), arrowprops=dict(arrowstyle="->", lw=2))
    
    # Contrastive Loss Box
    ax.add_patch(patches.Rectangle((0.75, 0.55), 0.1, 0.15, fill=False, ec='none'))
    ax.text(0.8, 0.625, 'Contrastive\nAlignment', ha='center', va='center', fontsize=10, rotation=90)
    ax.annotate('', xy=(0.8, 0.69), xytext=(0.8, 0.56), arrowprops=dict(arrowstyle="<->", lw=2, color='purple'))
    
    # Soft Targets feeding into Alignment
    ax.annotate('', xy=(0.85, 0.625), xytext=(0.85, 0.25), arrowprops=dict(arrowstyle="->", lw=2, color='orange', linestyle='dashed', connectionstyle="arc3,rad=-0.2"))
    ax.text(0.9, 0.4, 'Guides Loss', color='orange', fontweight='bold', rotation=270, ha='center', va='center')
    
    plt.title("Medical GRAM-CLIP Dual-Encoder Architecture", fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('/home/gheras/medical_gram_clip_project/arch_diagram.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_loss_diagram():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # standard InfoNCE
    mat1 = np.zeros((4,4))
    np.fill_diagonal(mat1, 1.0)
    cax1 = ax1.matshow(mat1, cmap='Reds', vmin=-0.2, vmax=1.2)
    ax1.set_title("Standard InfoNCE\n(Hard Targets)", fontsize=14, pad=15)
    for i in range(4):
        for j in range(4):
            val = "1.0" if i==j else "0.0"
            ax1.text(j, i, val, ha='center', va='center', color='black' if val=="0.0" else 'white', fontweight='bold')
    
    # MedCLIP/GRAM-Med Soft Targets
    mat2 = np.zeros((4,4))
    np.fill_diagonal(mat2, 1.0)
    mat2[0, 2] = 0.8  # Semantically similar pair
    mat2[2, 0] = 0.8
    mat2[1, 3] = 0.6
    mat2[3, 1] = 0.6
    cax2 = ax2.matshow(mat2, cmap='YlGn', vmin=0, vmax=1)
    ax2.set_title("MedCLIP / GRAM-Med\n(Semantic Soft Targets)", fontsize=14, pad=15)
    for i in range(4):
        for j in range(4):
            val = f"{mat2[i,j]:.1f}"
            ax2.text(j, i, val, ha='center', va='center', color='black', fontweight='bold')
    
    ax1.set_xticks(range(4)); ax1.set_yticks(range(4))
    ax1.set_xticklabels(['T1', 'T2', 'T3', 'T4']); ax1.set_yticklabels(['I1', 'I2', 'I3', 'I4'])
    ax2.set_xticks(range(4)); ax2.set_yticks(range(4))
    ax2.set_xticklabels(['T1', 'T2', 'T3', 'T4']); ax2.set_yticklabels(['I1', 'I2', 'I3', 'I4'])
    
    fig.suptitle("Contrastive Target Matrices Comparison", fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig('/home/gheras/medical_gram_clip_project/loss_diagram.png', dpi=150, bbox_inches='tight')
    plt.close()

create_arch_diagram()
create_loss_diagram()
