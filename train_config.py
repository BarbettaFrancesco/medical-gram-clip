from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainConfig:
    # --- Dataset & Environment ---
    output_dir: str = "output"            # Directory to save checkpoints and metrics
    num_workers: int = 4                  # Number of dataloader workers
    
    # --- Model Architecture ---
    vision_model_type: str = "cnn"        # "vit" or "cnn"
    vision_model_name: str = "resnet34"   # resnet34, google/vit-base-patch16-224
    projection_dim: int = 512             # Dimension of the joint embedding space
    
    # --- Training Hyperparameters ---
    loss_type: str = "gram_med"            # Loss objective to optimize (e.g., medclip, clip, gram, gram_med)
    batch_size: int = 64                  # Batch size per step
    grad_accum_steps: int = 1             # Number of steps to accumulate gradients
    epochs: int = 10                      # Total number of training epochs
    lr: float = 3e-5                      # Peak learning rate
    weight_decay: float = 1e-2            # Weight decay for AdamW
    target_temp: float = 0.05             # Temperature for soft semantic targets
    contrastive_temp: float = 0.07        # Temperature for volume computation
    
    # --- Data & Validation ---
    max_samples: Optional[int] = None     # Limit dataset size (None for full dataset)
    val_ratio: float = 0.2                # Fraction of data to use for validation
    
    # --- Optimization ---
    bf16: bool = True                    # Whether to use bfloat16 mixed precision
    vision_layers_unfrozen: int = 6       # Number of vision encoder layers to unfreeze
    text_layers_unfrozen: int = 4         # Number of text encoder layers to unfreeze
    
    # --- Checkpointing ---
    no_save_checkpoint: bool = False      # Set to True to skip saving checkpoints
