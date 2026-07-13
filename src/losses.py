from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Optional

import torch
from torch import nn
from torch.nn import functional as F

from gram_utils import volume_computation


class ClipInfoNCELoss(nn.Module):
    """
    Standard CLIP symmetric InfoNCE loss.

    It computes:
    - image-to-text contrastive loss
    - text-to-image contrastive loss

    The final loss is the average of the two.
    """

    def forward(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        logit_scale: torch.Tensor,
    ) -> torch.Tensor:
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)

        logits_per_image = logit_scale * image_embeds @ text_embeds.T
        logits_per_text = logits_per_image.T

        labels = torch.arange(image_embeds.size(0), device=image_embeds.device)

        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)

        return 0.5 * (loss_i2t + loss_t2i)


class GramLossAdapter(nn.Module):
    """
    Adapter for the official GRAM repository.

    This adapter imports the volume_computation function from gram_utils.
    """

    def __init__(
        self,
        contrastive_temp: float = 0.07,
        label_smoothing: float = 0.1,
    ) -> None:
        super().__init__()

        self.contrastive_temp = contrastive_temp
        self.label_smoothing = label_smoothing
        self.volume_computation: Callable = volume_computation

    def forward(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
    ) -> torch.Tensor:
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)

        volume = self.volume_computation(image_embeds, text_embeds)
        volume = volume / self.contrastive_temp

        volume_t = volume.T

        batch_size = image_embeds.size(0)
        targets = torch.arange(batch_size, device=image_embeds.device)

        loss_i2t = F.cross_entropy(
            -volume,
            targets,
            label_smoothing=self.label_smoothing,
        )

        loss_t2i = F.cross_entropy(
            -volume_t,
            targets,
            label_smoothing=self.label_smoothing,
        )

        return 0.5 * (loss_i2t + loss_t2i)

class MedClipLoss(nn.Module):
    """
    MedCLIP loss variant that computes semantic similarity from the text 
    embeddings themselves to form soft targets, preventing false negatives.
    """

    def __init__(self, target_temp: float = 0.1) -> None:
        super().__init__()
        self.target_temp = target_temp

    def forward(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        logit_scale: torch.Tensor,
        raw_text_embeds: torch.Tensor,
    ) -> torch.Tensor:
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)
        raw_text_embeds = F.normalize(raw_text_embeds, dim=-1)

        logits_per_image = logit_scale * image_embeds @ text_embeds.T
        logits_per_text = logits_per_image.T

        # Soft targets based on RAW text similarity (prevents moving target collapse)
        sim_text = raw_text_embeds @ raw_text_embeds.T
        targets = F.softmax(sim_text / self.target_temp, dim=-1)

        loss_i2t = F.cross_entropy(logits_per_image, targets)
        loss_t2i = F.cross_entropy(logits_per_text, targets)

        return 0.5 * (loss_i2t + loss_t2i)


class GramMedLoss(nn.Module):
    """
    GramMed loss: Combines GRAM volume computation with MedCLIP's logic
    of using soft targets derived from text semantic similarities.
    """

    def __init__(
        self,
        contrastive_temp: float = 0.07,
        target_temp: float = 0.1,
    ) -> None:
        super().__init__()

        self.contrastive_temp = contrastive_temp
        self.target_temp = target_temp
        self.volume_computation: Callable = volume_computation

    def forward(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        raw_text_embeds: torch.Tensor,
    ) -> torch.Tensor:
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)
        raw_text_embeds = F.normalize(raw_text_embeds, dim=-1)

        volume = self.volume_computation(image_embeds, text_embeds)
        volume = volume / self.contrastive_temp

        volume_t = volume.T

        # Soft targets from stable RAW text embeddings similarity
        sim_text = raw_text_embeds @ raw_text_embeds.T
        targets = F.softmax(sim_text / self.target_temp, dim=-1)

        loss_i2t = F.cross_entropy(-volume, targets)
        loss_t2i = F.cross_entropy(-volume_t, targets)

        return 0.5 * (loss_i2t + loss_t2i)


class LossRouter(nn.Module):
    """
    Selects which loss to use.

    Available loss types:
    - clip
    - gram
    - medclip
    - gram_med
    """

    def __init__(
        self,
        loss_type: str,
        contrastive_temp: float = 0.07,
        target_temp: float = 0.1,
    ) -> None:
        super().__init__()

        if loss_type not in {"clip", "gram", "medclip", "gram_med"}:
            raise ValueError(
                "loss_type must be one of: 'clip', 'gram', 'medclip', or 'gram_med'."
            )

        self.loss_type = loss_type

        self.clip_loss: Optional[ClipInfoNCELoss] = None
        self.gram_loss: Optional[GramLossAdapter] = None
        self.medclip_loss: Optional[MedClipLoss] = None
        self.gram_med_loss: Optional[GramMedLoss] = None

        if loss_type == "clip":
            self.clip_loss = ClipInfoNCELoss()

        elif loss_type == "gram":
            self.gram_loss = GramLossAdapter(
                contrastive_temp=contrastive_temp,
            )

        elif loss_type == "medclip":
            self.medclip_loss = MedClipLoss(target_temp=target_temp)

        elif loss_type == "gram_med":
            self.gram_med_loss = GramMedLoss(
                contrastive_temp=contrastive_temp,
                target_temp=target_temp,
            )

    def forward(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        logit_scale: torch.Tensor,
        raw_text_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.loss_type == "clip":
            assert self.clip_loss is not None
            return self.clip_loss(
                image_embeds=image_embeds,
                text_embeds=text_embeds,
                logit_scale=logit_scale,
            )

        if self.loss_type == "gram":
            assert self.gram_loss is not None
            return self.gram_loss(
                image_embeds=image_embeds,
                text_embeds=text_embeds,
            )

        if self.loss_type == "medclip":
            assert self.medclip_loss is not None
            assert raw_text_embeds is not None
            return self.medclip_loss(
                image_embeds=image_embeds,
                text_embeds=text_embeds,
                logit_scale=logit_scale,
                raw_text_embeds=raw_text_embeds,
            )

        if self.loss_type == "gram_med":
            assert self.gram_med_loss is not None
            assert raw_text_embeds is not None
            return self.gram_med_loss(
                image_embeds=image_embeds,
                text_embeds=text_embeds,
                raw_text_embeds=raw_text_embeds,
            )

        raise RuntimeError(f"Unknown loss_type: {self.loss_type}")