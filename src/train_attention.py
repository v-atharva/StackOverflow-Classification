"""Train the attention-based BiLSTM classifier."""

from __future__ import annotations

import argparse
import json
import os
import random
from functools import partial
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import config, data, vocab as vocab_mod
from .models.attention_lstm import AttentionBiLSTM

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train attention-based BiLSTM on Stack Overflow data.")
    parser.add_argument("--train_csv", type=str, default="train.csv")
    parser.add_argument("--valid_csv", type=str, default="valid.csv")
    parser.add_argument("--max_vocab_size", type=int, default=40000)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--embedding_dim", type=int, default=200)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--attention_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--run_name", type=str, default="attention_bilstm")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--class_weights", action="store_true", help="Use class weights in the loss.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_class_weights(labels: List[int]) -> torch.Tensor:
    counts = np.bincount(labels)
    total = counts.sum()
    weights = total / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def train_epoch(model, loader, criterion, optimizer, device, grad_clip: float) -> float:
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="Training", leave=False):
        tokens = batch.tokens.to(device)
        lengths = batch.lengths.to(device)
        labels = batch.labels.to(device)
        optimizer.zero_grad()
        logits = model(tokens, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * tokens.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels: List[int] = []
    all_preds: List[int] = []
    all_probs: List[np.ndarray] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            tokens = batch.tokens.to(device)
            lengths = batch.lengths.to(device)
            labels = batch.labels.to(device)
            logits = model(tokens, lengths)
            loss = criterion(logits, labels)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            total_loss += loss.item() * tokens.size(0)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())
    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    accuracy = accuracy_score(all_labels, all_preds)
    cls_report = classification_report(
        all_labels,
        all_preds,
        target_names=config.LABELS,
        output_dict=True,
        zero_division=0,
    )
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "report": cls_report,
        "labels": all_labels,
        "preds": all_preds,
        "probs": all_probs,
    }


def plot_confusion_matrix(labels, preds, run_name: str) -> Path:
    matrix = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=config.LABELS, yticklabels=config.LABELS, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plot_path = config.PLOTS_DIR / f"{run_name}_confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    return plot_path


def main():
    args = parse_args()
    set_seed(args.seed)
    train_df = data.read_dataframe(args.train_csv)
    valid_df = data.read_dataframe(args.valid_csv)
    train_tokens = data.prepare_tokens(train_df)
    vocab = vocab_mod.Vocabulary.build(
        train_tokens, max_size=args.max_vocab_size, min_freq=args.min_freq
    )
    valid_tokens = data.prepare_tokens(valid_df)
    train_dataset = data.StackOverflowTextDataset(train_df, train_tokens, vocab, args.max_len)
    valid_dataset = data.StackOverflowTextDataset(valid_df, valid_tokens, vocab, args.max_len)
    pad_index = vocab.pad_index
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=partial(data.collate_batch, pad_index=pad_index),
        num_workers=args.num_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=partial(data.collate_batch, pad_index=pad_index),
        num_workers=args.num_workers,
    )

    device = torch.device(args.device)
    model = AttentionBiLSTM(
        len(vocab),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        attention_dim=args.attention_dim,
        dropout=args.dropout,
        num_labels=len(config.LABELS),
        padding_idx=pad_index,
    ).to(device)

    train_labels = [label for _, label, _ in train_dataset.examples]
    if args.class_weights:
        weights = compute_class_weights(train_labels).to(device)
    else:
        weights = torch.ones(len(config.LABELS), dtype=torch.float32).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    history = []
    best_f1 = -1.0
    patience = args.early_stopping_patience
    best_epoch = 0
    checkpoint_path = config.CHECKPOINT_DIR / f"{args.run_name}_best.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, args.grad_clip)
        eval_result = evaluate(model, valid_loader, criterion, device)
        metrics_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": eval_result["loss"],
            "val_accuracy": eval_result["accuracy"],
            "val_macro_f1": eval_result["macro_f1"],
        }
        history.append(metrics_row)
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} "
            f"val_loss={eval_result['loss']:.4f} macro_f1={eval_result['macro_f1']:.4f}"
        )
        if eval_result["macro_f1"] > best_f1:
            best_f1 = eval_result["macro_f1"]
            best_epoch = epoch
            patience = args.early_stopping_patience
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "vocab": vocab.itos,
                    "config": vars(args),
                },
                checkpoint_path,
            )
            valid_ids = [meta["Id"] for _, _, meta in valid_dataset.examples]
            valid_true = [meta["Y"] for _, _, meta in valid_dataset.examples]
            predictions_df = pd.DataFrame({"Id": valid_ids, "true_label": valid_true})
            predictions_df["pred_label"] = [
                config.ID_TO_LABEL[idx] for idx in eval_result["preds"]
            ]
            prob_array = np.array(eval_result["probs"])
            for idx, label in enumerate(config.LABELS):
                predictions_df[f"prob_{label}"] = prob_array[:, idx]
            pred_path = config.METRICS_DIR / f"{args.run_name}_predictions.csv"
            predictions_df.to_csv(pred_path, index=False)
            plot_confusion_matrix(eval_result["labels"], eval_result["preds"], args.run_name)
            metrics_path = config.METRICS_DIR / f"{args.run_name}_metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(
                    {
                        "best_epoch": epoch,
                        "macro_f1": eval_result["macro_f1"],
                        "accuracy": eval_result["accuracy"],
                        "report": eval_result["report"],
                        "history": history,
                    },
                    f,
                    indent=2,
                )
        else:
            patience -= 1
            if patience <= 0:
                print("Early stopping triggered.")
                break

    print(f"Best validation macro-F1: {best_f1:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
