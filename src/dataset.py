"""Loads the labelled CSV described in Document/LABEL-SCHEME (2).md:

    comment_id,text,severity_id,target_id,has_pii,link_flagged,annotator,split,notes

`split` (train/eval) is assigned once when the row is written and must never
be reassigned here — that boundary is what keeps eval data out of training.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

REQUIRED_COLUMNS = {"comment_id", "text", "severity_id", "target_id", "split"}


def load_split(csv_path: str | Path, split: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")
    return df[df["split"] == split].reset_index(drop=True)


class ModerationDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
    ) -> None:
        self.texts = df["text"].astype(str).tolist()
        self.severity_ids = df["severity_id"].astype(int).tolist()
        self.target_ids = df["target_id"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[index],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "severity_id": torch.tensor(self.severity_ids[index], dtype=torch.long),
            "target_id": torch.tensor(self.target_ids[index], dtype=torch.long),
        }
