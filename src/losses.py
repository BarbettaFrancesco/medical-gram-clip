from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class ClipInfoNCELoss(nn.Module):
    def forward(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        logit_scale: torch.Tensor,
    ) -> torch.Tensor:
        logits_per_image = logit_scale * image_embeds @ text_embeds.T
        logits_per_text = logits_per_image.T
        labels = torch.arange(image_embeds.size(0), device=image_embeds.device)

        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)

        return 0.5 * (loss_i2t + loss_t2i)


class GramLossAdapter(nn.Module):
    """
    Adapter for the official GRAM repository.

    Clone first:
        git clone https://github.com/ispamm/GRAM external/GRAM
    """

    def __init__(self, gram_repo_path: str = "external/GRAM", contrastive_temp: float = 0.07) -> None:
        super().__init__()

        repo = Path(gram_repo_path).resolve()
        if not repo.exists():
            print(f"GRAM repo not found at {repo}. Cloning from GitHub...")
            import subprocess
            repo.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", "https://github.com/ispamm/GRAM", str(repo)], check=True)
            print("Successfully cloned GRAM repo.")

        sys.path.insert(0, str(repo))

        try:
            from utils.volume import volume_computation
            self.volume_computation = volume_computation
        except ImportError as exc:
            raise ImportError(
                "Could not import volume_computation from official GRAM repo. "
                "Inspect external/GRAM/utils/volume.py. "
                f"Last error: {exc}"
            )
            
        self.contrastive_temp = contrastive_temp

    def forward(self, image_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)

        # Compute volume using official function
        volume = self.volume_computation(image_embeds, text_embeds)
        volume = volume / self.contrastive_temp

        # Compute transpose volume
        volumeT = volume.T

        batch_size = image_embeds.size(0)
        targets = torch.arange(batch_size, device=image_embeds.device)

        # As defined in GRAM code: minimize volume for matched pairs (-volume)
        loss_i2t = F.cross_entropy(-volume, targets, label_smoothing=0.1)
        loss_t2i = F.cross_entropy(-volumeT, targets, label_smoothing=0.1)

        return 0.5 * (loss_i2t + loss_t2i)


class LossRouter(nn.Module):
    def __init__(self, loss_type: str, gram_repo_path: str = "external/GRAM") -> None:
        super().__init__()

        if loss_type not in {"clip", "gram"}:
            raise ValueError("loss_type must be either 'clip' or 'gram'.")

        self.loss_type = loss_type
        self.clip_loss = ClipInfoNCELoss()
        self.gram_loss = GramLossAdapter(gram_repo_path) if loss_type == "gram" else None

    def forward(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        logit_scale: torch.Tensor,
    ) -> torch.Tensor:
        if self.loss_type == "clip":
            return self.clip_loss(image_embeds, text_embeds, logit_scale)

        assert self.gram_loss is not None
        return self.gram_loss(image_embeds, text_embeds)