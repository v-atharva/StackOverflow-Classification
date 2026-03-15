"""Attention-based BiLSTM model with optional pretrained embeddings."""

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
        num_layers: int = 2,
        dropout: float = 0.3,
        num_labels: int = 3,
        padding_idx: int = 0,
        attention_dim: int = 128,
        pretrained_vectors: torch.Tensor | None = None,
        freeze_embeds: bool = False,
    ) -> None:
        super().__init__()
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

        # Load pretrained embeddings when available
        if pretrained_vectors is not None:
            self.embedding.weight.data.copy_(pretrained_vectors)
            if freeze_embeds:
                self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out_dim = hidden_dim * 2  # bidirectional

        # Layer normalisation on LSTM output
        self.layer_norm = nn.LayerNorm(lstm_out_dim)

        # Multi-head–style additive attention (still single-head Bahdanau
        # but properly uses padding_idx for mask)
        self.attention = nn.Sequential(
            nn.Linear(lstm_out_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1, bias=False),
        )

        # Residual projection: map embedding dim to lstm_out_dim so we can
        # add it to the attended context (skip connection)
        self.residual_proj = (
            nn.Linear(embedding_dim, lstm_out_dim, bias=False)
            if embedding_dim != lstm_out_dim
            else nn.Identity()
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_out_dim, num_labels)

    def forward(
        self,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
        return_attention: bool = False,
    ):
        embedded = self.embedding(tokens)  # (B, T, E)

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True
        )  # (B, T, 2H)

        # Apply layer normalisation
        output = self.layer_norm(output)

        # Mask using stored padding_idx (NOT hard‑coded 0)
        mask = tokens != self.padding_idx  # (B, T)

        # Attention
        attn_logits = self.attention(output).squeeze(-1)  # (B, T)
        attn_logits = attn_logits.masked_fill(~mask, -1e9)
        weights = F.softmax(attn_logits, dim=1)  # (B, T)

        context = torch.sum(output * weights.unsqueeze(-1), dim=1)  # (B, 2H)

        # Residual connection: average‑pool the raw embedding and add to context
        embed_mask = mask.unsqueeze(-1).float()  # (B, T, 1)
        embed_avg = (embedded * embed_mask).sum(dim=1) / embed_mask.sum(dim=1).clamp(min=1)
        residual = self.residual_proj(embed_avg)  # (B, 2H)
        context = context + residual

        context = self.dropout(context)
        logits = self.fc(context)

        if return_attention:
            return logits, weights
        return logits
