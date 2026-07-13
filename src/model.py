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
        vision_model_type: str = "vit",
    ) -> None:
        super().__init__()

        self.vision_model_type = vision_model_type

        if self.vision_model_type == "cnn":
            import timm
            self.vision_encoder = timm.create_model(vision_model_name, pretrained=True, num_classes=0)
            vision_hidden_dim = self.vision_encoder.num_features
        else:
            self.vision_encoder = AutoModel.from_pretrained(vision_model_name)
            vision_hidden_dim = 768

        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        
        # Instantiate a separate, completely frozen text encoder specifically to 
        # generate stable ground truth semantic targets, preventing target collapse.
        self.target_text_encoder = AutoModel.from_pretrained(text_model_name)
        self.target_text_encoder.eval()
        for param in self.target_text_encoder.parameters():
            param.requires_grad = False

        text_hidden_dim = 768
        self.image_projector = nn.Linear(vision_hidden_dim, projection_dim)
        self.text_projector = nn.Linear(text_hidden_dim, projection_dim)
        self.logit_scale = nn.Parameter(torch.tensor(1 / 0.07).log())

    def freeze_encoders(self, vision_layers_unfrozen: int = 1, text_layers_unfrozen: int = 2) -> None:
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        for param in self.text_encoder.parameters():
            param.requires_grad = False

        if self.vision_model_type == "vit":
            # ViT encoder blocks
            if hasattr(self.vision_encoder, "encoder"):
                for block in self.vision_encoder.encoder.layer[-vision_layers_unfrozen:]:
                    for param in block.parameters():
                        param.requires_grad = True
        elif self.vision_model_type == "cnn":
            if vision_layers_unfrozen > 0:
                children = list(self.vision_encoder.children())
                for child in children[-vision_layers_unfrozen:]:
                    for param in child.parameters():
                        param.requires_grad = True

        # BERT encoder blocks
        if hasattr(self.text_encoder, "encoder"):
            for block in self.text_encoder.encoder.layer[-text_layers_unfrozen:]:
                for param in block.parameters():
                    param.requires_grad = True

        for param in self.image_projector.parameters():
            param.requires_grad = True
            
        for param in self.text_projector.parameters():
            param.requires_grad = True

        self.logit_scale.requires_grad = True

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        if self.vision_model_type == "cnn":
            cls = self.vision_encoder(images)
        else:
            outputs = self.vision_encoder(pixel_values=images)
            cls = outputs.last_hidden_state[:, 0]
            
        projected = self.image_projector(cls)
        return F.normalize(projected, dim=-1)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0]
        projected = self.text_projector(cls)
        return F.normalize(projected, dim=-1)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        image_embeds = self.encode_image(images)
        
        # The active text encoder being fine-tuned and projected
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        active_text_cls = text_outputs.last_hidden_state[:, 0]
        projected_text = self.text_projector(active_text_cls)
        text_embeds = F.normalize(projected_text, dim=-1)

        # The frozen target text encoder for stable ground truth semantics
        with torch.no_grad():
            target_outputs = self.target_text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            raw_text_cls = target_outputs.last_hidden_state[:, 0]

        return {
            "image_embeds": image_embeds,
            "text_embeds": text_embeds,
            "raw_text_embeds": F.normalize(raw_text_cls, dim=-1),
            "logit_scale": self.logit_scale.exp().clamp(max=100),
        }