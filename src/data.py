"""
Amazon Review Polarity data pipeline.

Each sample: polarity (1=negative, 2=positive), title, text.
Title and text are concatenated into a single string.
"""

import random
from typing import Optional

import tiktoken
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from src.config import DataConfig


# ---------------------------------------------------------------------------
# Raw data loading
# ---------------------------------------------------------------------------

def load_amazon_reviews(
    num_samples: int = 50_000,
    test_split: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Returns (train_samples, test_samples) where each sample is
    {"text": str, "label": int}  (label: 0=negative, 1=positive).
    """
    dataset = load_dataset("amazon_polarity", split="train")

    # Shuffle & take num_samples
    indices = list(range(len(dataset)))
    random.seed(seed)
    random.shuffle(indices)
    indices = indices[:num_samples]

    samples = []
    for i in indices:
        row = dataset[i]
        combined = (row["title"] + " " + row["content"]).strip()
        label = row["label"]          # 0 or 1 in HF version
        samples.append({"text": combined, "label": label})

    n_test = int(num_samples * test_split)
    train_samples = samples[n_test:]
    test_samples = samples[:n_test]
    return train_samples, test_samples


# ---------------------------------------------------------------------------
# PyTorch Dataset for the transformer (language modelling)
# ---------------------------------------------------------------------------

class ReviewLMDataset(Dataset):
    """
    Next-token prediction dataset built from Amazon review text.
    Labels (sentiment) are NOT used here — the transformer learns
    pure language modelling.
    """

    def __init__(
        self,
        samples: list[dict],
        max_seq_len: int = 128,
        tokenizer: Optional[tiktoken.Encoding] = None,
    ):
        if tokenizer is None:
            tokenizer = tiktoken.get_encoding("cl100k_base")
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_id = 0   # cl100k_base does not have a dedicated pad token; use 0

        self.encoded: list[list[int]] = []
        for s in samples:
            ids = tokenizer.encode(s["text"], disallowed_special=())
            if len(ids) < 2:
                continue
            self.encoded.append(ids)

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ids = self.encoded[idx]
        # Crop to max_seq_len + 1 (we need input + target)
        ids = ids[: self.max_seq_len + 1]

        input_ids = ids[:-1]
        target_ids = ids[1:]

        # Pad to max_seq_len
        pad_len = self.max_seq_len - len(input_ids)
        input_ids = input_ids + [self.pad_id] * pad_len
        target_ids = target_ids + [-100] * pad_len   # -100 = ignore in CE loss

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long),
        )


def get_lm_dataloaders(
    train_samples: list[dict],
    test_samples: list[dict],
    config: DataConfig,
    batch_size: int = 64,
) -> tuple[DataLoader, DataLoader]:
    tokenizer = tiktoken.get_encoding("cl100k_base")
    train_ds = ReviewLMDataset(train_samples, config.max_seq_len, tokenizer)
    test_ds = ReviewLMDataset(test_samples, config.max_seq_len, tokenizer)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Activation dataset (for SAE training)
# ---------------------------------------------------------------------------

class ActivationDataset(Dataset):
    """Wraps a pre-collected tensor of activations."""

    def __init__(self, activations: torch.Tensor):
        self.activations = activations

    def __len__(self) -> int:
        return len(self.activations)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.activations[idx]


if __name__ == "__main__":
    print("Loading 50k Amazon reviews…")
    train, test = load_amazon_reviews(num_samples=50_000, test_split=0.2)
    print(f"Train: {len(train)}  Test: {len(test)}")
    print("Sample:", train[0])
