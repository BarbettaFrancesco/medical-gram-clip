from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Optional

import torch
from torch import nn
from torch.nn import functional as F


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

    Before using this loss, clone the official GRAM repository:

        git clone https://github.com/ispamm/GRAM external/GRAM

    This adapter imports:

        external/GRAM/utils/volume.py

    and uses the official volume_computation function.
    """

    def __init__(
        self,
        gram_repo_path: str = "external/GRAM",
        contrastive_temp: float = 0.07,
        label_smoothing: float = 0.1,
    ) -> None:
        super().__init__()

        self.contrastive_temp = contrastive_temp
        self.label_smoothing = label_smoothing

        repo = Path(gram_repo_path).resolve()

        if not repo.exists():
            raise FileNotFoundError(
                f"GRAM repository not found at:\n"
                f"{repo}\n\n"
                f"Run this command from the project root:\n"
                f"git clone https://github.com/ispamm/GRAM {gram_repo_path}"
            )

        sys.path.insert(0, str(repo))

        try:
            from utils.volume import volume_computation
            self.volume_computation: Callable = volume_computation
        except ImportError as exc:
            raise ImportError(
                f"Could not import volume_computation from the official GRAM repo.\n"
                f"Expected file:\n"
                f"{repo}/utils/volume.py\n\n"
                f"Original error: {exc}"
            )

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


class GramClipHybridLoss(nn.Module):
    """
    Hybrid loss: CLIP InfoNCE + GRAM loss.

    alpha controls the balance:

        alpha = 1.0  -> only CLIP
        alpha = 0.0  -> only GRAM
        alpha = 0.5  -> half CLIP, half GRAM

    This is useful because pure GRAM can be unstable in a two-modality
    image-text setting.
    """

    def __init__(
        self,
        gram_repo_path: str = "external/GRAM",
        contrastive_temp: float = 0.07,
        alpha: float = 0.5,
    ) -> None:
        super().__init__()

        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0.0 and 1.0.")

        self.alpha = alpha
        self.clip_loss = ClipInfoNCELoss()
        self.gram_loss = GramLossAdapter(
            gram_repo_path=gram_repo_path,
            contrastive_temp=contrastive_temp,
        )

    def forward(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        logit_scale: torch.Tensor,
    ) -> torch.Tensor:
        loss_clip = self.clip_loss(
            image_embeds=image_embeds,
            text_embeds=text_embeds,
            logit_scale=logit_scale,
        )

        loss_gram = self.gram_loss(
            image_embeds=image_embeds,
            text_embeds=text_embeds,
        )

        return self.alpha * loss_clip + (1.0 - self.alpha) * loss_gram


class LossRouter(nn.Module):
    """
    Selects which loss to use.

    Available loss types:
    - clip
    - gram
    - gram_clip
    """

    def __init__(
        self,
        loss_type: str,
        gram_repo_path: str = "external/GRAM",
        contrastive_temp: float = 0.07,
        hybrid_alpha: float = 0.5,
    ) -> None:
        super().__init__()

        if loss_type not in {"clip", "gram", "gram_clip"}:
            raise ValueError(
                "loss_type must be one of: 'clip', 'gram', or 'gram_clip'."
            )

        self.loss_type = loss_type

        self.clip_loss: Optional[ClipInfoNCELoss] = None
        self.gram_loss: Optional[GramLossAdapter] = None
        self.hybrid_loss: Optional[GramClipHybridLoss] = None

        if loss_type == "clip":
            self.clip_loss = ClipInfoNCELoss()

        elif loss_type == "gram":
            self.gram_loss = GramLossAdapter(
                gram_repo_path=gram_repo_path,
                contrastive_temp=contrastive_temp,
            )

        elif loss_type == "gram_clip":
            self.hybrid_loss = GramClipHybridLoss(
                gram_repo_path=gram_repo_path,
                contrastive_temp=contrastive_temp,
                alpha=hybrid_alpha,
            )

    def forward(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        logit_scale: torch.Tensor,
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

        if self.loss_type == "gram_clip":
            assert self.hybrid_loss is not None
            return self.hybrid_loss(
                image_embeds=image_embeds,
                text_embeds=text_embeds,
                logit_scale=logit_scale,
            )

        raise RuntimeError(f"Unknown loss_type: {self.loss_type}")