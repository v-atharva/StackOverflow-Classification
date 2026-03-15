"""Vocabulary utilities with pretrained embedding support."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch

logger = logging.getLogger(__name__)

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


@dataclass
class Vocabulary:
    stoi: Dict[str, int]
    itos: List[str]

    @classmethod
    def build(
        cls,
        token_iter: Iterable[List[str]],
        max_size: int = 40000,
        min_freq: int = 2,
    ) -> "Vocabulary":
        counter = Counter()
        for tokens in token_iter:
            counter.update(tokens)
        itos = [PAD_TOKEN, UNK_TOKEN]
        for token, freq in counter.most_common():
            if freq < min_freq or len(itos) >= max_size:
                continue
            itos.append(token)
        stoi = {token: idx for idx, token in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    @classmethod
    def from_itos(cls, itos: List[str]) -> "Vocabulary":
        stoi = {token: idx for idx, token in enumerate(itos)}
        return cls(stoi=stoi, itos=list(itos))

    @property
    def pad_index(self) -> int:
        return self.stoi[PAD_TOKEN]

    @property
    def unk_index(self) -> int:
        return self.stoi[UNK_TOKEN]

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.stoi.get(token, self.unk_index) for token in tokens]

    def __len__(self) -> int:
        return len(self.itos)

    # ------------------------------------------------------------------
    # Pretrained embedding helpers
    # ------------------------------------------------------------------
    def load_pretrained_vectors(
        self,
        path: str,
        embedding_dim: int,
        *,
        binary: bool = False,
    ) -> torch.Tensor:
        """Load vectors from a GloVe/fastText text file and return an
        ``(vocab_size, embedding_dim)`` Tensor aligned to ``self.itos``.

        Tokens not found in the pretrained file are initialised from
        ``N(0, 0.25)`` (the default used by many NLP papers).

        For GloVe:  ``vectors/glove.6B.300d.txt``
        For fastText: ``vectors/crawl-300d-2M.vec``
        """
        vectors = torch.randn(len(self), embedding_dim) * 0.25
        # Pad token should always be zeros
        vectors[self.pad_index] = 0.0

        found = 0
        logger.info("Loading pretrained vectors from %s …", path)
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                word = parts[0]
                if word in self.stoi:
                    try:
                        vec = torch.tensor(
                            [float(x) for x in parts[1:]], dtype=torch.float32
                        )
                        if vec.size(0) == embedding_dim:
                            vectors[self.stoi[word]] = vec
                            found += 1
                    except ValueError:
                        continue
        coverage = found / max(len(self) - 2, 1) * 100  # exclude pad/unk
        logger.info(
            "Loaded %d / %d vocab tokens (%.1f%% coverage)",
            found,
            len(self) - 2,
            coverage,
        )
        return vectors
