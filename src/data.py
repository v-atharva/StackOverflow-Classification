"""Dataset utilities for Stack Overflow quality classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from . import config, preprocessing, vocab as vocab_mod


def read_dataframe(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def stratified_split(
    df: pd.DataFrame,
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "sizes must sum to 1"
    stratify = df["Y"]
    train_df, temp_df = train_test_split(
        df,
        train_size=train_size,
        stratify=stratify,
        random_state=random_state,
    )
    stratify_temp = temp_df["Y"]
    relative_val_size = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=relative_val_size,
        stratify=stratify_temp,
        random_state=random_state,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(
        drop=True
    )


def prepare_tokens(df: pd.DataFrame) -> List[List[str]]:
    return [preprocessing.preprocess(row["Title"], row["Body"]) for _, row in df.iterrows()]


class StackOverflowTextDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokens: Sequence[List[str]],
        vocab: vocab_mod.Vocabulary,
        max_len: int,
    ) -> None:
        self.vocab = vocab
        self.max_len = max_len
        self.pad_index = vocab.pad_index
        self.examples = []
        for row, token_list in zip(df.itertuples(index=False), tokens):
            encoded = vocab.encode(token_list[:max_len])
            if not encoded:
                encoded = [self.pad_index]
            label = config.LABEL_TO_ID[getattr(row, "Y")]
            meta = {"Id": getattr(row, "Id"), "Y": getattr(row, "Y"), "tokens": token_list}
            self.examples.append((encoded, label, meta))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        return self.examples[idx]


@dataclass
class TextBatch:
    tokens: torch.Tensor
    lengths: torch.Tensor
    labels: torch.Tensor
    metas: List[dict]


def collate_batch(batch, pad_index: int) -> TextBatch:
    lengths = torch.tensor([len(seq) for seq, _, _ in batch], dtype=torch.long)
    max_len = int(lengths.max().item())
    padded = torch.full((len(batch), max_len), pad_index, dtype=torch.long)
    labels = torch.tensor([label for _, label, _ in batch], dtype=torch.long)
    metas = [meta for _, _, meta in batch]
    for i, (seq, _, _) in enumerate(batch):
        padded[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return TextBatch(tokens=padded, lengths=lengths, labels=labels, metas=metas)


def build_datasets(
    train_csv: str,
    valid_csv: str,
    max_vocab_size: int = 40000,
    min_freq: int = 2,
    max_len: int = 256,
) -> Tuple[
    StackOverflowTextDataset,
    StackOverflowTextDataset,
    vocab_mod.Vocabulary,
]:
    train_df = read_dataframe(train_csv)
    valid_df = read_dataframe(valid_csv)
    train_tokens = prepare_tokens(train_df)
    vocab = vocab_mod.Vocabulary.build(train_tokens, max_size=max_vocab_size, min_freq=min_freq)
    valid_tokens = prepare_tokens(valid_df)
    train_dataset = StackOverflowTextDataset(train_df, train_tokens, vocab, max_len=max_len)
    valid_dataset = StackOverflowTextDataset(valid_df, valid_tokens, vocab, max_len=max_len)
    return train_dataset, valid_dataset, vocab
