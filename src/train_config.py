import argparse

class TrainConfig:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--hf_token", type=str, required=True, help="HuggingFace token for MIMIC-CXR")
        parser.add_argument("--output_dir", type=str, required=True)
        parser.add_argument("--loss_type", type=str, choices=["clip", "gram", "medclip", "gram_med"], required=True)

        parser.add_argument("--gram_repo_path", type=str, default="external/GRAM")
        parser.add_argument("--projection_dim", type=int, default=512)
        
        parser.add_argument("--vision_model_type", type=str, default="vit", choices=["vit", "cnn"])
        parser.add_argument("--vision_model_name", type=str, default="google/vit-base-patch16-224")
        
        parser.add_argument("--contrastive_temp", type=float, default=0.07)
        parser.add_argument("--target_temp", type=float, default=0.1)

        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--grad_accum_steps", type=int, default=16)
        parser.add_argument("--epochs", type=int, default=10)
        parser.add_argument("--lr", type=float, default=5e-6)
        parser.add_argument("--weight_decay", type=float, default=1e-2)
        parser.add_argument("--val_ratio", type=float, default=0.1)
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--bf16", action="store_true")
        parser.add_argument("--no_save_checkpoint", action="store_true")
        parser.add_argument("--gradient_checkpointing", action="store_true")
        parser.add_argument("--vision_layers_unfrozen", type=int, default=-1)
        parser.add_argument("--text_layers_unfrozen", type=int, default=-1)

        args = parser.parse_args()
        for k, v in vars(args).items():
            setattr(self, k, v)
