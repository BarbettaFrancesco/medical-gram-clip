from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from dotenv import load_dotenv
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer


load_dotenv()


class MIMICCXRDataset(Dataset):
    """
    Dataset wrapper for MIMIC-CXR from Hugging Face.

    Each item returns:
        {
            "image": PIL image,
            "text": report string
        }
    """

    def __init__(
        self,
        hf_token: str | None = None,
        split: str = "train",
        dataset_name: str = "MLforHealthcare/mimic-cxr",
    ) -> None:
        super().__init__()

        if hf_token is None:
            hf_token = os.environ.get("HF_TOKEN")

        if not hf_token:
            raise ValueError(
                "Hugging Face token not found. "
                "Pass --hf_token YOUR_TOKEN or create a .env file with HF_TOKEN=YOUR_TOKEN"
            )

        self.data = load_dataset(
            dataset_name,
            split=split,
            token=hf_token,
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]

        image = item["image"]
        text = item["reports"]

        if isinstance(text, list):
            text = " ".join(str(t) for t in text)

        if text is None:
            text = ""

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        image = image.convert("RGB")

        return {
            "image": image,
            "text": str(text),
        }


class MedicalCollator:
    """
    Collator for the DataLoader.

    It converts:
    - PIL images into ViT pixel_values
    - report strings into tokenizer input_ids and attention_mask
    """

    def __init__(
        self,
        vision_model_name: str = "google/vit-base-patch16-224",
        text_model_name: str = "NeuML/pubmedbert-base-embeddings",
        max_length: int = 256,
        vision_model_type: str = "vit",
    ) -> None:
        self.vision_model_type = vision_model_type

        if self.vision_model_type == "cnn":
            import timm
            from torchvision import transforms
            model_config = timm.data.resolve_model_data_config(vision_model_name)
            self.image_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=model_config["mean"], std=model_config["std"]),
            ])
        else:
            self.image_processor = AutoImageProcessor.from_pretrained(vision_model_name, use_fast=True)

        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images = [sample["image"] for sample in batch]
        texts = [sample["text"] for sample in batch]

        if self.vision_model_type == "cnn":
            image_tensors = [self.image_transform(img) for img in images]
            pixel_values = torch.stack(image_tensors)
        else:
            image_inputs = self.image_processor(
                images=images,
                return_tensors="pt",
            )
            pixel_values = image_inputs["pixel_values"]

        text_inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "images": pixel_values,
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
        }