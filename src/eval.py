from __future__ import annotations

from typing import Dict, Iterable

import torch
from torch.nn import functional as F
from tqdm import tqdm


@torch.no_grad()
def extract_embeddings(
    model, 
    dataloader, 
    device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    image_embeddings = []
    text_embeddings = []
    raw_text_embeddings = []

    for batch in tqdm(dataloader, desc="Extracting embeddings"):
        images = batch["images"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(images, input_ids, attention_mask)
        
        # Extract raw CLS token from the text encoder (before projector)
        raw_text_outputs = model.target_text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        raw_text_cls = raw_text_outputs.last_hidden_state[:, 0]

        image_embeddings.append(outputs["image_embeds"].cpu())
        text_embeddings.append(outputs["text_embeds"].cpu())
        raw_text_embeddings.append(raw_text_cls.cpu())

    return (
        torch.cat(image_embeddings, dim=0), 
        torch.cat(text_embeddings, dim=0),
        torch.cat(raw_text_embeddings, dim=0)
    )


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


def semantic_recall_at_k(
    similarity: torch.Tensor, 
    text_similarity: torch.Tensor, 
    threshold: float = 0.8, 
    ks: Iterable[int] = (1, 5, 10)
) -> Dict[str, float]:
    metrics = {}
    
    # A retrieved item `j` for query `i` is correct if text `j` is similar to text `i`
    positive_targets = text_similarity >= threshold

    for k in ks:
        actual_k = min(k, similarity.size(1))
        if actual_k <= 0:
            metrics[f"SemR@{k}"] = 0.0
            continue
            
        topk_indices = similarity.topk(actual_k, dim=1).indices
        
        # Gather the boolean values from positive_targets using the retrieved indices
        retrieved_is_positive = positive_targets.gather(1, topk_indices)
        
        # If at least one retrieved item is a valid target, it's a hit
        correct = retrieved_is_positive.any(dim=1).float().mean().item()
        metrics[f"SemR@{k}"] = correct

    return metrics


@torch.no_grad()
def retrieval_metrics(model, dataloader, device: torch.device) -> Dict[str, float]:
    image_embeds, text_embeds, raw_text_embeds = extract_embeddings(model, dataloader, device)

    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)
    raw_text_embeds = F.normalize(raw_text_embeds, dim=-1)

    # Standard Contrastive Similarities (using projected embeddings)
    similarity = image_embeds @ text_embeds.T
    
    # Text-Text Similarities for Semantic Recall (using raw, stable PubMedBERT embeddings)
    text_similarity = raw_text_embeds @ raw_text_embeds.T

    # Calculate standard recall
    i2t = recall_at_k(similarity)
    t2i = recall_at_k(similarity.T)
    
    # Calculate semantic recall
    sem_i2t = semantic_recall_at_k(similarity, text_similarity, threshold=0.8)
    sem_t2i = semantic_recall_at_k(similarity.T, text_similarity, threshold=0.8)

    metrics = {}
    for key, value in i2t.items():
        metrics[f"i2t_{key}"] = value
    for key, value in t2i.items():
        metrics[f"t2i_{key}"] = value
        
    for key, value in sem_i2t.items():
        metrics[f"i2t_{key}"] = value
    for key, value in sem_t2i.items():
        metrics[f"t2i_{key}"] = value

    return metrics