"""Dataset utilities for Stack Overflow quality classification."""

from __future__ import annotations

import hashlib
import logging
import pickle
import random
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from . import config, preprocessing, vocab as vocab_mod

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

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
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Tokenisation (with parallelism + disk caching)
# ---------------------------------------------------------------------------

def _preprocess_row(title: str, body: str) -> List[str]:
    return preprocessing.preprocess(
        title if isinstance(title, str) else "",
        body if isinstance(body, str) else "",
    )


def prepare_tokens(
    df: pd.DataFrame,
    *,
    cache_dir: str | Path | None = None,
    n_jobs: int = 1,
) -> List[List[str]]:
    """Tokenise title+body for every row.

    * Uses ``df.apply`` instead of ``iterrows`` for speed.
    * Optionally caches results to *cache_dir* keyed by a content hash.
    * When *n_jobs* > 1 uses ``joblib.Parallel`` for true parallelism.
    """
    # --- Disk cache ---
    cache_path: Path | None = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Deterministic hash of the first+last few ids and shape
        key_data = f"{df.shape}-{df['Id'].iloc[0]}-{df['Id'].iloc[-1]}"
        digest = hashlib.md5(key_data.encode()).hexdigest()[:12]
        cache_path = cache_dir / f"tokens_{digest}.pkl"
        if cache_path.exists():
            logger.info("Loading cached tokens from %s", cache_path)
            with open(cache_path, "rb") as fh:
                return pickle.load(fh)

    # --- Parallel path ---
    if n_jobs != 1:
        try:
            from joblib import Parallel, delayed

            tokens = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_preprocess_row)(row["Title"], row["Body"])
                for _, row in df.iterrows()
            )
        except ImportError:
            logger.warning("joblib not available; falling back to sequential tokenisation")
            tokens = df.apply(
                lambda row: _preprocess_row(row["Title"], row["Body"]), axis=1
            ).tolist()
    else:
        tokens = df.apply(
            lambda row: _preprocess_row(row["Title"], row["Body"]), axis=1
        ).tolist()

    # --- Write cache ---
    if cache_path is not None:
        with open(cache_path, "wb") as fh:
            pickle.dump(tokens, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Cached tokenised data to %s", cache_path)

    return tokens


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class StackOverflowTextDataset(Dataset):
    """PyTorch dataset wrapping encoded tokens, labels and metadata.

    When *token_dropout* > 0, randomly replaces that fraction of tokens with
    ``<unk>`` during *training* only (controlled by ``self.training`` flag).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokens: Sequence[List[str]],
        vocab: vocab_mod.Vocabulary,
        max_len: int,
        token_dropout: float = 0.0,
    ) -> None:
        self.vocab = vocab
        self.max_len = max_len
        self.pad_index = vocab.pad_index
        self.unk_index = vocab.unk_index
        self.token_dropout = token_dropout
        self.training = False  # toggled externally

        self.examples = []
        for row, token_list in zip(df.itertuples(index=False), tokens):
            encoded = vocab.encode(token_list[:max_len])
            if not encoded:
                encoded = [self.pad_index]
            label = config.LABEL_TO_ID[getattr(row, "Y")]
            meta = {
                "Id": getattr(row, "Id"),
                "Y": getattr(row, "Y"),
                "tokens": token_list,
            }
            self.examples.append((encoded, label, meta))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        encoded, label, meta = self.examples[idx]
        if self.training and self.token_dropout > 0:
            encoded = [
                tok if random.random() > self.token_dropout else self.unk_index
                for tok in encoded
            ]
        return encoded, label, meta


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# High‑level builder
# ---------------------------------------------------------------------------

def build_datasets(
    train_csv: str,
    valid_csv: str,
    max_vocab_size: int = 40000,
    min_freq: int = 2,
    max_len: int = 256,
    token_dropout: float = 0.0,
    cache_dir: str | Path | None = "artifacts/token_cache",
    n_jobs: int = 1,
) -> Tuple[
    StackOverflowTextDataset,
    StackOverflowTextDataset,
    vocab_mod.Vocabulary,
]:
    train_df = read_dataframe(train_csv)
    valid_df = read_dataframe(valid_csv)
    train_tokens = prepare_tokens(train_df, cache_dir=cache_dir, n_jobs=n_jobs)
    vocab = vocab_mod.Vocabulary.build(
        train_tokens, max_size=max_vocab_size, min_freq=min_freq
    )
    valid_tokens = prepare_tokens(valid_df, cache_dir=cache_dir, n_jobs=n_jobs)
    train_dataset = StackOverflowTextDataset(
        train_df, train_tokens, vocab, max_len=max_len, token_dropout=token_dropout
    )
    valid_dataset = StackOverflowTextDataset(
        valid_df, valid_tokens, vocab, max_len=max_len, token_dropout=0.0
    )
    return train_dataset, valid_dataset, vocab
