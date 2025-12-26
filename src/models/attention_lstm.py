"""Attention-based BiLSTM model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBiLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 200,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.3,
        num_labels: int = 3,
        padding_idx: int = 0,
        attention_dim: int = 128,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1, bias=False),
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor, return_attention: bool = False):
        embedded = self.embedding(tokens)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        mask = tokens != 0
        attn_logits = self.attention(output).squeeze(-1)
        attn_logits = attn_logits.masked_fill(~mask, -1e9)
        weights = F.softmax(attn_logits, dim=1)
        context = torch.sum(output * weights.unsqueeze(-1), dim=1)
        context = self.dropout(context)
        logits = self.fc(context)
        if return_attention:
            return logits, weights
        return logits

