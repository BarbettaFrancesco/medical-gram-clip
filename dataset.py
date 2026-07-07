from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def extract_section(report: str) -> str:
    """
    Extract FINDINGS or IMPRESSION from a radiology report.
    Falls back to the original cleaned report if sections are not found.
    """
    if not isinstance(report, str):
        return ""

    text = report.replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)

    patterns = [
        r"findings\s*:(.*?)(impression\s*:|$)",
        r"impression\s*:(.*)$",
    ]

    extracted = []
    lower = text.lower()

    for pattern in patterns:
        match = re.search(pattern, lower, flags=re.IGNORECASE)
        if match:
            extracted.append(match.group(1).strip())

    clean = " ".join(extracted).strip()
    clean = re.sub(r"\s+", " ", clean)

    return clean if len(clean.split()) >= 5 else text


class MimicCXRDataset(Dataset):
    """
    Dataset for a Kaggle-style MIMIC-CXR subset.

    Expected CSV columns are flexible:
    - image path: one of ["image_path", "path", "jpg_path", "dicom_path"]
    - view position: one of ["ViewPosition", "view_position", "view"]
    - report text: one of ["report", "text", "findings", "impression"]
    """

    def __init__(
        self,
        csv_path: str | Path,
        image_root: str | Path,
        split: str | None = None,
        max_samples: int | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.image_root = Path(image_root)

        df = pd.read_csv(self.csv_path)

        view_col = self._find_column(df, ["ViewPosition", "view_position", "view"])
        df = df[df[view_col].astype(str).str.upper().isin(["PA", "AP"])].copy()

        if split is not None and "split" in df.columns:
            df = df[df["split"].astype(str).str.lower() == split.lower()].copy()

        self.path_col = self._find_column(df, ["image_path", "path", "jpg_path", "dicom_path"])
        self.text_col = self._find_column(df, ["report", "text", "findings", "impression"])

        df["clean_text"] = df[self.text_col].astype(str).apply(extract_section)
        df = df[df["clean_text"].str.split().str.len() >= 5].copy()

        if max_samples is not None:
            df = df.head(max_samples).copy()

        self.df = df.reset_index(drop=True)

        self.transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    @staticmethod
    def _find_column(df: pd.DataFrame, candidates: List[str]) -> str:
        for col in candidates:
            if col in df.columns:
                return col
        raise ValueError(f"Missing required column. Tried: {candidates}. Found: {list(df.columns)}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        image_path = Path(str(row[self.path_col]))

        if not image_path.is_absolute():
            image_path = self.image_root / image_path

        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)

        return {
            "image": image_tensor,
            "text": str(row["clean_text"]),
            "index": idx,
        }


class MedicalCollator:
    def __init__(
        self,
        tokenizer_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        max_length: int = 128,
    ) -> None:
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images = torch.stack([item["image"] for item in batch])
        texts = [item["text"] for item in batch]
        indices = torch.tensor([item["index"] for item in batch], dtype=torch.long)

        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "images": images,
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "indices": indices,
        }