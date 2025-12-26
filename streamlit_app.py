"""Streamlit UI for Stack Overflow question quality classification."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch

from src import config, preprocessing, vocab as vocab_mod
from src.models.attention_lstm import AttentionBiLSTM

PROJECT_ROOT = Path(__file__).parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "artifacts" / "checkpoints" / "attn_bilstm_best.pt"


@st.cache_resource(show_spinner=True)
def load_model(checkpoint_path: Path):
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Run src.train_attention first."
        )
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    vocab = vocab_mod.Vocabulary.from_itos(ckpt["vocab"])
    cfg = ckpt["config"]
    model = AttentionBiLSTM(
        len(vocab),
        embedding_dim=cfg.get("embedding_dim", 200),
        hidden_dim=cfg.get("hidden_dim", 128),
        attention_dim=cfg.get("attention_dim", 128),
        dropout=cfg.get("dropout", 0.3),
        num_labels=len(config.LABELS),
        padding_idx=vocab.pad_index,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    max_len = cfg.get("max_len", 200)
    return model, vocab, max_len


def predict_quality(
    model: AttentionBiLSTM,
    vocab: vocab_mod.Vocabulary,
    max_len: int,
    title: str,
    body: str,
) -> Tuple[str, np.ndarray, list[str], np.ndarray]:
    model.eval()
    tokens = preprocessing.preprocess(title, body)[:max_len]
    encoded = vocab.encode(tokens)
    if not encoded:
        encoded = [vocab.pad_index]
    input_tensor = torch.tensor([encoded], dtype=torch.long)
    lengths = torch.tensor([len(encoded)], dtype=torch.long)
    with torch.no_grad():
        logits, attn = model(input_tensor, lengths, return_attention=True)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        attention = attn.squeeze(0).cpu().numpy()[: len(tokens)]
    pred_label = config.ID_TO_LABEL[int(probs.argmax())]
    return pred_label, probs, tokens, attention


def main():
    st.set_page_config(page_title="Stack Overflow Quality Classifier", layout="wide")
    st.title("Stack Overflow Question Quality Classifier")
    st.markdown(
        "Enter a Stack Overflow question title and body to estimate whether it is "
        "**High Quality (HQ)**, **Low Quality – needs edits (LQ_EDIT)**, or "
        "**Low Quality – should be closed (LQ_CLOSE)**."
    )

    checkpoint_path = st.sidebar.text_input(
        "Checkpoint path",
        value=str(DEFAULT_CHECKPOINT),
        help="Path to the saved attention BiLSTM checkpoint (.pt).",
    )

    try:
        model, vocab, max_len = load_model(Path(checkpoint_path))
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to load model: {exc}")
        st.stop()

    with st.form("question_form"):
        title = st.text_input("Question title", "")
        body = st.text_area(
            "Question body",
            "",
            height=250,
            help="Paste the Markdown/HTML body of the question.",
        )
        submitted = st.form_submit_button("Predict Quality")

    if submitted:
        if not title.strip() and not body.strip():
            st.warning("Please provide at least a title or body.")
            st.stop()
        pred_label, probs, tokens, attention = predict_quality(
            model, vocab, max_len, title, body
        )
        st.subheader("Prediction")
        st.success(f"Predicted label: **{pred_label}**")
        st.caption("Confidence scores (softmax probabilities):")
        prob_table = {
            "Class": config.LABELS,
            "Confidence": [f"{p:.3f}" for p in probs],
        }
        st.table(prob_table)

        if tokens:
            st.subheader("Attention highlights")
            attn_df = pd.DataFrame({"Token": tokens, "Attention": attention})
            top_tokens = attn_df.sort_values("Attention", ascending=False).head(20)
            st.caption("Top attention tokens")
            st.table(
                {
                    "Token": top_tokens["Token"].tolist(),
                    "Attention": [f"{w:.4f}" for w in top_tokens["Attention"]],
                }
            )
            st.caption("Attention distribution across the sequence")
            st.bar_chart(attn_df.set_index("Token"))


if __name__ == "__main__":
    main()
