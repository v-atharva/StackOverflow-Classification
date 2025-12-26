"""Vocabulary utilities."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List


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
