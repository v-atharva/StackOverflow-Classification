"""Visualise attention weights for a single question."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from . import config, preprocessing, vocab as vocab_mod
from .models.attention_lstm import AttentionBiLSTM

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualise attention for a Stack Overflow question.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--csv", type=str, default="valid.csv")
    parser.add_argument("--question_id", type=int, help="Question Id to visualise.")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save the figure.")
    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    run_config = ckpt["config"]
    vocab = vocab_mod.Vocabulary.from_itos(ckpt["vocab"])
    model = AttentionBiLSTM(
        len(vocab),
        embedding_dim=run_config.get("embedding_dim", 200),
        hidden_dim=run_config.get("hidden_dim", 128),
        attention_dim=run_config.get("attention_dim", 128),
        dropout=run_config.get("dropout", 0.3),
        num_labels=len(config.LABELS),
        padding_idx=vocab.pad_index,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, vocab, run_config


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, run_config = load_model(args.checkpoint, device)
    df = pd.read_csv(args.csv)
    if args.question_id:
        row = df.loc[df["Id"] == args.question_id]
        if row.empty:
            raise ValueError(f"Question id {args.question_id} not found in {args.csv}")
        row = row.iloc[0]
    else:
        row = df.sample(1, random_state=42).iloc[0]
    tokens = preprocessing.preprocess(row["Title"], row["Body"])
    max_len = run_config.get("max_len", 256)
    tokens = tokens[:max_len]
    encoded = vocab.encode(tokens)
    if not encoded:
        encoded = [vocab.pad_index]
    tokens_tensor = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(device)
    lengths = torch.tensor([len(encoded)], dtype=torch.long).to(device)
    with torch.no_grad():
        logits, attn = model(tokens_tensor, lengths, return_attention=True)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_label = config.ID_TO_LABEL[int(probs.argmax())]
        attention = attn.squeeze(0).cpu().numpy()[: len(tokens)]
    print(f"Question Id: {row['Id']}, true_label={row['Y']}, predicted={pred_label}")
    print("Top attention tokens:")
    token_weights = list(zip(tokens, attention))
    token_weights.sort(key=lambda x: x[1], reverse=True)
    for token, weight in token_weights[:20]:
        print(f"{token:20s} -> {weight:.4f}")

    fig, ax = plt.subplots(figsize=(min(12, len(tokens) * 0.4), 3))
    ax.bar(range(len(tokens)), attention, color="teal")
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90, fontsize=8)
    ax.set_ylabel("Attention weight")
    ax.set_title(f"Attention over tokens (pred={pred_label}, true={row['Y']})")
    fig.tight_layout()
    output_path = args.output or (config.PLOTS_DIR / f"attention_{row['Id']}.png")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Attention plot saved to {output_path}")


if __name__ == "__main__":
    main()

