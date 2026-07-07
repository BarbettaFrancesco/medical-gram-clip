from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import MedicalCollator, MimicCXRDataset
from eval import retrieval_metrics
from losses import LossRouter
from model import MedicalMultimodal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--loss_type", type=str, choices=["clip", "gram"], required=True)

    parser.add_argument("--gram_repo_path", type=str, default="external/GRAM")
    parser.add_argument("--projection_dim", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--vision_layers_unfrozen", type=int, default=1)
    parser.add_argument("--text_layers_unfrozen", type=int, default=2)

    return parser.parse_args()


def train_one_epoch(
    model: MedicalMultimodal,
    loss_fn: LossRouter,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    grad_accum_steps: int,
    use_bf16: bool,
) -> float:
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler_enabled = device.type == "cuda" and not use_bf16
    scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    for step, batch in enumerate(tqdm(dataloader, desc="Training")):
        images = batch["images"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            outputs = model(images, input_ids, attention_mask)
            loss = loss_fn(
                outputs["image_embeds"],
                outputs["text_embeds"],
                outputs["logit_scale"],
            )
            loss = loss / grad_accum_steps

        if scaler_enabled:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            if scaler_enabled:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * grad_accum_steps

    return total_loss / max(len(dataloader), 1)


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MimicCXRDataset(
        csv_path=args.csv_path,
        image_root=args.image_root,
        max_samples=args.max_samples,
    )

    val_size = int(len(dataset) * args.val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    collator = MedicalCollator()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
    )

    model = MedicalMultimodal(projection_dim=args.projection_dim)
    model.freeze_encoders(
        vision_layers_unfrozen=args.vision_layers_unfrozen,
        text_layers_unfrozen=args.text_layers_unfrozen,
    )
    model.to(device)

    loss_fn = LossRouter(args.loss_type, gram_repo_path=args.gram_repo_path)

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_steps = max(len(train_loader) * args.epochs // args.grad_accum_steps, 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    history: List[Dict[str, float]] = []

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model=model,
            loss_fn=loss_fn,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            grad_accum_steps=args.grad_accum_steps,
            use_bf16=args.bf16,
        )

        metrics = retrieval_metrics(model, val_loader, device)
        row: Dict[str, float] = {"epoch": float(epoch + 1), "train_loss": train_loss}
        row.update(metrics)
        history.append(row)

        print(json.dumps(row, indent=2))

        with open(output_dir / "metrics.json", "w") as f:
            json.dump(
                {
                    "loss_type": args.loss_type,
                    "history": history,
                    "args": vars(args),
                },
                f,
                indent=2,
            )

        torch.save(model.state_dict(), output_dir / f"model_epoch_{epoch + 1}.pt")


if __name__ == "__main__":
    main()