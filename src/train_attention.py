"""Train the attention-based BiLSTM classifier.

Improvements
------------
* LR scheduler (ReduceLROnPlateau on val macro-F1).
* Label smoothing (default 0.1).
* Non-zero weight decay (default 1e-3).
* Optional pretrained embeddings (GloVe / fastText).
* Token dropout augmentation (default 10%).
* Mixed-precision training (AMP) on CUDA.
* Training / validation loss & F1 curves plotted.
* Per-class F1 tracked across epochs.
* More epochs (default 15) with patience 5.
* Proper logging instead of print().
* Reproducible DataLoader worker seeds.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from functools import partial
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import config, data, vocab as vocab_mod
from .models.attention_lstm import AttentionBiLSTM

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train attention-based BiLSTM on Stack Overflow data."
    )
    # Data
    parser.add_argument("--train_csv", type=str, default="train.csv")
    parser.add_argument("--valid_csv", type=str, default="valid.csv")
    parser.add_argument("--max_vocab_size", type=int, default=40000)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=256)

    # Model
    parser.add_argument("--embedding_dim", type=int, default=300)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--attention_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)

    # Pretrained embeddings
    parser.add_argument(
        "--pretrained_vectors",
        type=str,
        default=None,
        help="Path to GloVe/fastText text vectors (e.g. glove.6B.300d.txt).",
    )
    parser.add_argument(
        "--freeze_embeds",
        action="store_true",
        help="Freeze pretrained embedding weights.",
    )

    # Training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--token_dropout", type=float, default=0.1)

    # Scheduler
    parser.add_argument(
        "--scheduler",
        choices=["plateau", "onecycle", "none"],
        default="plateau",
    )
    parser.add_argument("--lr_patience", type=int, default=1)
    parser.add_argument("--lr_factor", type=float, default=0.5)

    # Misc
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
    )
    parser.add_argument("--run_name", type=str, default="attention_bilstm")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument(
        "--class_weights",
        action="store_true",
        help="Use class weights in the loss.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable mixed-precision training (CUDA only).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _worker_init_fn(worker_id: int, seed: int = 42) -> None:
    """Ensure each DataLoader worker gets a deterministic seed."""
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------

def compute_class_weights(labels: List[int]) -> torch.Tensor:
    counts = np.bincount(labels)
    total = counts.sum()
    weights = total / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Training / evaluation loops
# ---------------------------------------------------------------------------

def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    grad_clip: float,
    scaler=None,
    use_amp: bool = False,
) -> float:
    model.train()
    # enable token dropout on the dataset
    if hasattr(loader.dataset, "training"):
        loader.dataset.training = True

    total_loss = 0.0
    for batch in tqdm(loader, desc="Training", leave=False):
        tokens = batch.tokens.to(device)
        lengths = batch.lengths.to(device)
        labels = batch.labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        if use_amp and scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                logits = model(tokens, lengths)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
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
    # disable token dropout
    if hasattr(loader.dataset, "training"):
        loader.dataset.training = False

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


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_confusion_matrix(labels, preds, run_name: str) -> Path:
    matrix = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=config.LABELS,
        yticklabels=config.LABELS,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plot_path = config.PLOTS_DIR / f"{run_name}_confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    return plot_path


def plot_training_curves(history: List[dict], run_name: str) -> Path:
    """Plot train/val loss and val macro-F1 across epochs."""
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Loss curves
    axes[0].plot(epochs, [h["train_loss"] for h in history], "o-", label="Train loss")
    axes[0].plot(epochs, [h["val_loss"] for h in history], "o-", label="Val loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Macro F1
    axes[1].plot(
        epochs,
        [h["val_macro_f1"] for h in history],
        "o-",
        color="green",
        label="Val macro-F1",
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1")
    axes[1].set_title("Validation Macro F1")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Per-class F1
    for label in config.LABELS:
        key = f"val_f1_{label}"
        if key in history[0]:
            axes[2].plot(
                epochs, [h[key] for h in history], "o-", label=label
            )
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("F1")
    axes[2].set_title("Per-Class F1")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path = config.PLOTS_DIR / f"{run_name}_training_curves.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    return plot_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    args = parse_args()
    set_seed(args.seed)
    logger.info("Config: %s", vars(args))

    # ---- Data ----
    train_df = data.read_dataframe(args.train_csv)
    valid_df = data.read_dataframe(args.valid_csv)
    train_tokens = data.prepare_tokens(train_df, cache_dir="artifacts/token_cache")
    vocab = vocab_mod.Vocabulary.build(
        train_tokens, max_size=args.max_vocab_size, min_freq=args.min_freq
    )
    valid_tokens = data.prepare_tokens(valid_df, cache_dir="artifacts/token_cache")
    train_dataset = data.StackOverflowTextDataset(
        train_df,
        train_tokens,
        vocab,
        args.max_len,
        token_dropout=args.token_dropout,
    )
    valid_dataset = data.StackOverflowTextDataset(
        valid_df, valid_tokens, vocab, args.max_len, token_dropout=0.0
    )
    pad_index = vocab.pad_index

    worker_init = partial(_worker_init_fn, seed=args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=partial(data.collate_batch, pad_index=pad_index),
        num_workers=args.num_workers,
        worker_init_fn=worker_init,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=partial(data.collate_batch, pad_index=pad_index),
        num_workers=args.num_workers,
        worker_init_fn=worker_init,
    )

    # ---- Pretrained embeddings ----
    pretrained_vectors = None
    if args.pretrained_vectors:
        pretrained_vectors = vocab.load_pretrained_vectors(
            args.pretrained_vectors, args.embedding_dim
        )

    # ---- Model ----
    device = torch.device(args.device)
    model = AttentionBiLSTM(
        len(vocab),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        attention_dim=args.attention_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_labels=len(config.LABELS),
        padding_idx=pad_index,
        pretrained_vectors=pretrained_vectors,
        freeze_embeds=args.freeze_embeds,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model has %s trainable parameters", f"{n_params:,}")

    # ---- Loss ----
    train_labels = [label for _, label, _ in train_dataset.examples]
    if args.class_weights:
        weights = compute_class_weights(train_labels).to(device)
    else:
        weights = torch.ones(len(config.LABELS), dtype=torch.float32).to(device)
    criterion = torch.nn.CrossEntropyLoss(
        weight=weights, label_smoothing=args.label_smoothing
    )

    # ---- Optimiser ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # ---- LR Scheduler ----
    scheduler = None
    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=args.lr_factor,
            patience=args.lr_patience,
        )
    elif args.scheduler == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.learning_rate * 10,
            epochs=args.epochs,
            steps_per_epoch=len(train_loader),
        )

    # ---- Mixed precision ----
    use_amp = args.amp and args.device == "cuda"
    scaler = torch.amp.GradScaler() if use_amp else None
    if use_amp:
        logger.info("Mixed-precision training enabled")

    # ---- Training loop ----
    history: List[dict] = []
    best_f1 = -1.0
    patience = args.early_stopping_patience
    best_epoch = 0
    checkpoint_path = config.CHECKPOINT_DIR / f"{args.run_name}_best.pt"

    for epoch in range(1, args.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info("Epoch %d/%d  (lr=%.2e)", epoch, args.epochs, current_lr)

        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            args.grad_clip,
            scaler=scaler,
            use_amp=use_amp,
        )
        eval_result = evaluate(model, valid_loader, criterion, device)

        # Build metrics row including per-class F1
        metrics_row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_loss,
            "val_loss": eval_result["loss"],
            "val_accuracy": eval_result["accuracy"],
            "val_macro_f1": eval_result["macro_f1"],
        }
        for label in config.LABELS:
            metrics_row[f"val_f1_{label}"] = eval_result["report"].get(label, {}).get(
                "f1-score", 0.0
            )
        history.append(metrics_row)

        logger.info(
            "  train_loss=%.4f  val_loss=%.4f  macro_f1=%.4f  acc=%.4f",
            train_loss,
            eval_result["loss"],
            eval_result["macro_f1"],
            eval_result["accuracy"],
        )
        for label in config.LABELS:
            f1_val = metrics_row.get(f"val_f1_{label}", 0.0)
            logger.info("    %s F1: %.4f", label, f1_val)

        # Step scheduler
        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(eval_result["macro_f1"])
            # OneCycleLR steps per batch (handled inside train_epoch if needed)

        # Checkpoint best
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

            # Save predictions
            valid_ids = [meta["Id"] for _, _, meta in valid_dataset.examples]
            valid_true = [meta["Y"] for _, _, meta in valid_dataset.examples]
            predictions_df = pd.DataFrame(
                {"Id": valid_ids, "true_label": valid_true}
            )
            predictions_df["pred_label"] = [
                config.ID_TO_LABEL[idx] for idx in eval_result["preds"]
            ]
            prob_array = np.array(eval_result["probs"])
            for idx, label in enumerate(config.LABELS):
                predictions_df[f"prob_{label}"] = prob_array[:, idx]
            pred_path = config.METRICS_DIR / f"{args.run_name}_predictions.csv"
            predictions_df.to_csv(pred_path, index=False)

            plot_confusion_matrix(
                eval_result["labels"], eval_result["preds"], args.run_name
            )
        else:
            patience -= 1
            if patience <= 0:
                logger.info("Early stopping triggered at epoch %d.", epoch)
                break

    # ---- Final outputs ----
    plot_training_curves(history, args.run_name)

    metrics_path = config.METRICS_DIR / f"{args.run_name}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "best_epoch": best_epoch,
                "macro_f1": best_f1,
                "history": history,
            },
            f,
            indent=2,
        )
    logger.info(
        "Best validation macro-F1: %.4f at epoch %d", best_f1, best_epoch
    )


if __name__ == "__main__":
    main()
