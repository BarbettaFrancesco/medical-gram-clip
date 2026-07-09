import os
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from dotenv import load_dotenv

# Carica le variabili nascoste dal file .env
load_dotenv()

class MIMICCXRDataset(Dataset):
    def __init__(self, split="train"):
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise ValueError("Token HF_TOKEN non trovato! Crea un file .env e inseriscilo.")
            
        self.data = load_dataset("MLforHealthcare/mimic-cxr", split=split, token=hf_token)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        elemento = self.data[idx]
        return elemento['image'], elemento['reports']
