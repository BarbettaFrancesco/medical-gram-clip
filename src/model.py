from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel


class MedicalMultimodal(nn.Module):
    def __init__(
        self,
        vision_model_name: str = "google/vit-base-patch16-224",
        text_model_name: str = "NeuML/pubmedbert-base-embeddings",
        projection_dim: int = 512,
    ) -> None:
        super().__init__()

        self.vision_encoder = AutoModel.from_pretrained(vision_model_name)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)

        hidden_dim = 768
        self.shared_projector = nn.Linear(hidden_dim, projection_dim)
        self.logit_scale = nn.Parameter(torch.tensor(1 / 0.07).log())

    def freeze_encoders(self, vision_layers_unfrozen: int = 1, text_layers_unfrozen: int = 2) -> None:
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # ViT encoder blocks
        if hasattr(self.vision_encoder, "encoder"):
            for block in self.vision_encoder.encoder.layer[-vision_layers_unfrozen:]:
                for param in block.parameters():
                    param.requires_grad = True

        # BERT encoder blocks
        if hasattr(self.text_encoder, "encoder"):
            for block in self.text_encoder.encoder.layer[-text_layers_unfrozen:]:
                for param in block.parameters():
                    param.requires_grad = True

        for param in self.shared_projector.parameters():
            param.requires_grad = True

        self.logit_scale.requires_grad = True

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.vision_encoder(pixel_values=images)
        cls = outputs.last_hidden_state[:, 0]
        projected = self.shared_projector(cls)
        return F.normalize(projected, dim=-1)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0]
        projected = self.shared_projector(cls)
        return F.normalize(projected, dim=-1)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        image_embeds = self.encode_image(images)
        text_embeds = self.encode_text(input_ids, attention_mask)

        return {
            "image_embeds": image_embeds,
            "text_embeds": text_embeds,
            "logit_scale": self.logit_scale.exp().clamp(max=100),
        }