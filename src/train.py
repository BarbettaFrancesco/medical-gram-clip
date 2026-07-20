from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import MedicalCollator, MIMICCXRDataset
from eval import retrieval_metrics
from losses import LossRouter
from model import MedicalMultimodal


sys.path.append(str(Path(__file__).resolve().parent.parent))
from train_config import TrainConfig


def train_one_epoch(
    model: MedicalMultimodal,
    loss_fn: LossRouter,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    grad_accum_steps: int,
    use_bf16: bool,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    model.train()
    total_loss = 0.0

    grad_accum_steps = max(1, grad_accum_steps)
    optimizer.zero_grad(set_to_none=True)

    use_cuda = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    progress_bar = tqdm(dataloader, desc="Training")

    for step, batch in enumerate(progress_bar):
        images = batch["images"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_cuda):
            outputs = model(images, input_ids, attention_mask)
            loss = loss_fn(
                image_embeds=outputs["image_embeds"],
                text_embeds=outputs["text_embeds"],
                logit_scale=outputs["logit_scale"],
                raw_text_embeds=outputs.get("raw_text_embeds"),
            )
            loss = loss / grad_accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        is_accumulation_step = (step + 1) % grad_accum_steps == 0
        is_last_step = (step + 1) == len(dataloader)

        if is_accumulation_step or is_last_step:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        current_loss = loss.item() * grad_accum_steps
        total_loss += current_loss
        progress_bar.set_postfix({"loss": f"{current_loss:.4f}"})

    return total_loss / max(len(dataloader), 1)


def main() -> None:
    args = TrainConfig()

    vision_model_type = getattr(args, "vision_model_type", "vit")
    output_dir = Path(args.output_dir) / vision_model_type / args.loss_type
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80, flush=True)
    print("[INFO] Starting Medical GRAM-CLIP training", flush=True)
    print(f"[INFO] Loss type: {args.loss_type}", flush=True)
    print(f"[INFO] Device: {device}", flush=True)
    print(f"[INFO] HF Token provided: {'*' * 10}")
    print(f"[INFO] Output dir: {output_dir}", flush=True)
    print("=" * 80, flush=True)

    print("[INFO] Loading dataset...", flush=True)
    dataset = MIMICCXRDataset(hf_token=args.hf_token, split="train")

    if getattr(args, "max_samples", None) is not None and len(dataset) > args.max_samples:
        print(f"[INFO] Limiting dataset to {args.max_samples} samples", flush=True)
        dataset = torch.utils.data.Subset(dataset, range(args.max_samples))

    if len(dataset) < 2:
        raise ValueError("Dataset must contain at least 2 samples.")

    val_size = max(1, int(len(dataset) * args.val_ratio))
    train_size = len(dataset) - val_size

    if train_size < 1:
        raise ValueError("Training split is empty. Increase dataset size or reduce val_ratio.")

    print(f"[INFO] Dataset loaded: {len(dataset)} samples", flush=True)
    print(f"[INFO] Train samples: {train_size}", flush=True)
    print(f"[INFO] Validation samples: {val_size}", flush=True)

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    print("[INFO] Loading BioClinicalBERT tokenizer...", flush=True)
    collator = MedicalCollator(
        vision_model_type=getattr(args, "vision_model_type", "vit"),
        vision_model_name=getattr(args, "vision_model_name", "google/vit-base-patch16-224"),
    )

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=pin_memory,
    )

    print("[INFO] Loading ViT/CNN + PubMedBERT model...", flush=True)
    model = MedicalMultimodal(
        projection_dim=args.projection_dim,
        vision_model_type=getattr(args, "vision_model_type", "vit"),
        vision_model_name=getattr(args, "vision_model_name", "google/vit-base-patch16-224"),
    )

    print("[INFO] Freezing encoders...", flush=True)
    model.freeze_encoders(
        vision_layers_unfrozen=args.vision_layers_unfrozen,
        text_layers_unfrozen=args.text_layers_unfrozen,
    )

    model.to(device)
    print("[INFO] Model ready", flush=True)

    print(f"[INFO] Initializing loss: {args.loss_type} (target_temp={getattr(args, 'target_temp', 0.1)}, contrastive_temp={getattr(args, 'contrastive_temp', 0.07)})", flush=True)
    loss_fn = LossRouter(
        args.loss_type,
        contrastive_temp=getattr(args, "contrastive_temp", 0.07),
        target_temp=getattr(args, "target_temp", 0.1),
    )

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    steps_per_epoch = max(1, (len(train_loader) + args.grad_accum_steps - 1) // args.grad_accum_steps)
    total_steps = max(1, steps_per_epoch * args.epochs)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
    )

    history: List[Dict[str, float]] = []
    best_loss = float("inf")

    # Initialize GradScaler once outside the loop (only if using CUDA and not bf16)
    scaler = torch.amp.GradScaler("cuda") if (device.type == "cuda" and not args.bf16) else None

    print("[INFO] Starting training loop...", flush=True)

    for epoch in range(args.epochs):
        print(f"\n[INFO] Epoch {epoch + 1}/{args.epochs}", flush=True)

        train_loss = train_one_epoch(
            model=model,
            loss_fn=loss_fn,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            grad_accum_steps=args.grad_accum_steps,
            use_bf16=args.bf16,
            scaler=scaler,
        )

        print("[INFO] Running retrieval evaluation...", flush=True)
        metrics = retrieval_metrics(model, val_loader, device, args.loss_type)

        row: Dict[str, float] = {
            "epoch": float(epoch + 1),
            "train_loss": train_loss,
        }
        row.update(metrics)
        history.append(row)

        print("[INFO] Epoch metrics:", flush=True)
        print(json.dumps(row, indent=2), flush=True)

        metrics_path = output_dir / "metrics.json"
        print(f"[INFO] Saving metrics to {metrics_path}", flush=True)

        with open(metrics_path, "w") as f:
            json.dump(
                {
                    "loss_type": args.loss_type,
                    "history": history,
                    "args": asdict(args),
                },
                f,
                indent=2,
            )

        if not args.no_save_checkpoint:
            if train_loss < best_loss:
                best_loss = train_loss
                checkpoint_path = output_dir / f"model_{args.loss_type}.pt"
                print(f"[INFO] New best loss: {best_loss:.4f}. Saving checkpoint to {checkpoint_path}", flush=True)
                torch.save(model.state_dict(), checkpoint_path)
                print("[INFO] Checkpoint saved", flush=True)
            else:
                print(f"[INFO] Current loss {train_loss:.4f} did not improve from best loss {best_loss:.4f}. Skipping checkpoint save.", flush=True)
        else:
            print("[INFO] Skipping checkpoint save", flush=True)

    print("\n[INFO] Training completed successfully.", flush=True)


if __name__ == "__main__":
    main()