from __future__ import annotations

from typing import Dict, Iterable

import torch
from torch.nn import functional as F
from tqdm import tqdm


@torch.no_grad()
def extract_embeddings(model, dataloader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    image_embeddings = []
    text_embeddings = []

    for batch in tqdm(dataloader, desc="Extracting embeddings"):
        images = batch["images"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(images, input_ids, attention_mask)

        image_embeddings.append(outputs["image_embeds"].cpu())
        text_embeddings.append(outputs["text_embeds"].cpu())

    return torch.cat(image_embeddings, dim=0), torch.cat(text_embeddings, dim=0)


def recall_at_k(similarity: torch.Tensor, ks: Iterable[int] = (1, 5, 10)) -> Dict[str, float]:
    n = similarity.size(0)
    targets = torch.arange(n)

    metrics = {}

    for k in ks:
        actual_k = min(k, similarity.size(1))
        if actual_k <= 0:
            metrics[f"R@{k}"] = 0.0
            continue
            
        topk = similarity.topk(actual_k, dim=1).indices
        correct = topk.eq(targets[:, None]).any(dim=1).float().mean().item()
        metrics[f"R@{k}"] = correct

    return metrics


@torch.no_grad()
def retrieval_metrics(model, dataloader, device: torch.device) -> Dict[str, float]:
    image_embeds, text_embeds = extract_embeddings(model, dataloader, device)

    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)

    similarity = image_embeds @ text_embeds.T

    i2t = recall_at_k(similarity)
    t2i = recall_at_k(similarity.T)

    metrics = {}
    for key, value in i2t.items():
        metrics[f"i2t_{key}"] = value
    for key, value in t2i.items():
        metrics[f"t2i_{key}"] = value

    return metrics