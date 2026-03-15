"""Streamlit UI for Stack Overflow question quality classification with Data Storytelling."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from scipy.stats import gaussian_kde

from src import config, preprocessing, vocab as vocab_mod
from src.features import structural_features
from src.models.attention_lstm import AttentionBiLSTM

PROJECT_ROOT = Path(__file__).parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "artifacts" / "checkpoints" / "attention_bilstm_best.pt"


# ---------------------------------------------------------------------------
# Caching / Loading Logic
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading Attention Model...")
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
        embedding_dim=cfg.get("embedding_dim", 300),
        hidden_dim=cfg.get("hidden_dim", 128),
        attention_dim=cfg.get("attention_dim", 128),
        num_layers=cfg.get("num_layers", 2),
        dropout=cfg.get("dropout", 0.3),
        num_labels=len(config.LABELS),
        padding_idx=vocab.pad_index,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    max_len = cfg.get("max_len", 256)
    return model, vocab, max_len


@st.cache_data(show_spinner="Loading EDA Data Sample... (takes a few seconds)")
def load_eda_data(sample_size: int = 4000) -> pd.DataFrame:
    """Load a sample of train data and extract handcrafted features for EDA."""
    train_path = PROJECT_ROOT / "dataset" / "train.csv"
    if not train_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(train_path)
    
    # Sampled evenly across the 3 labels
    dfs = []
    for label in config.LABELS:
        sub = df[df["Y"] == label]
        dfs.append(sub.sample(n=min(len(sub), sample_size // 3), random_state=42))
    sampled_df = pd.concat(dfs).reset_index(drop=True)
    
    # Compute hand-crafted features
    feats = sampled_df.apply(structural_features, axis=1).tolist()
    feat_df = pd.DataFrame(feats)
    
    # Merge label in
    feat_df["Y"] = sampled_df["Y"]
    return feat_df


# ---------------------------------------------------------------------------
# Prediction Tab
# ---------------------------------------------------------------------------

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


def render_prediction_tab(model, vocab, max_len):
    st.markdown(
        "Enter a Stack Overflow question title and body to estimate whether it is "
        "**High Quality (HQ)**, **Low Quality – needs edits (LQ_EDIT)**, or "
        "**Low Quality – should be closed (LQ_CLOSE)**."
    )

    with st.form("question_form"):
        title = st.text_input("Question title", "How to map a list in Python?")
        body = st.text_area(
            "Question body",
            "<p>I have a list <code>[1, 2, 3]</code> and want to double everything to <code>[2, 4, 6]</code>. Plz help thanks.</p>",
            height=200,
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
        
        # Color based on severity
        color = "green" if pred_label == "HQ" else ("orange" if pred_label == "LQ_EDIT" else "red")
        st.markdown(f"### Predicted label: <span style='color:{color}'>**{pred_label}**</span>", unsafe_allow_html=True)
        
        st.caption("Confidence scores (softmax probabilities):")
        
        # Format as nice progress bars rather than tables
        cols = st.columns(3)
        for i, (label, prob) in enumerate(zip(config.LABELS, probs)):
            with cols[i]:
                st.metric(label, f"{prob:.1%}")
                st.progress(float(prob))

        if tokens:
            st.divider()
            st.subheader("Neural Attention Maps 🧠")
            st.markdown("This bar chart shows exactly **which words** the BiLSTM model weighed the highest when making its decision.")
            attn_df = pd.DataFrame({"Token": tokens, "Attention": attention})
            
            # Interactive Plotly chart for attention map
            fig = px.bar(
                attn_df, x="Token", y="Attention", 
                title="Attention distribution across the sequence",
                color="Attention",
                color_continuous_scale="Viridis",
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Data EDA Tab
# ---------------------------------------------------------------------------

def render_eda_tab():
    st.header("Exploratory Data Analysis: What makes a good question?")
    st.markdown("""
    In the data engineering phase, I extracted high-signal **structural features** to detect quality based purely on 
    how a question is formatted and constructed, independently from what the neural net learns.
    
    This dashboard explores a sample of the training dataset, demonstrating the clear separating boundaries of the quality classes.
    """)
    
    df = load_eda_data()
    if df.empty:
        st.error("No training data found at `dataset/train.csv` to generate EDA.")
        return
        
    st.info(f"Loaded a representative sample of {len(df):,} Stack Overflow questions for analysis.")
    
    # ---------------------------------------------------------
    # Insight 1: Structure & Effort
    # ---------------------------------------------------------
    st.subheader("1. The Effort Heuristic: Structure & Formatting")
    st.markdown("""
    High quality questions are typically structured with care. Using lists, paragraphs, and high-readability phrasing indicates 
    that the user spent time articulating their problem, rather than performing a memory dump.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.box(
            df, x="Y", y="num_paragraphs", 
            color="Y",
            title="Number of Paragraphs by Quality",
            category_orders={"Y": ["HQ", "LQ_EDIT", "LQ_CLOSE"]},
            points="outliers"
        )
        fig1.update_yaxes(range=[-1, 15])  # Cap outliers for viewing
        st.plotly_chart(fig1, use_container_width=True)
        st.caption("Observation: HQ questions format their thoughts into multiple paragraphs, while LQ_CLOSE questions are often completely unformatted walls of text.")

    with col2:
        # Flesch reading ease
        fig2 = px.violin(
            df[df["flesch_reading_ease"].between(-50, 150)], 
            x="Y", y="flesch_reading_ease",
            color="Y",
            title="Readability Score (Flesch)",
            box=True,
            category_orders={"Y": ["HQ", "LQ_EDIT", "LQ_CLOSE"]}
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Observation: LQ_EDIT and LQ_CLOSE text tends to have lower reading ease scores globally or show erratic variance driven by code dumping outside of `<code>` blocks.")

    st.divider()

    # ---------------------------------------------------------
    # Insight 2: The Politeness Paradox
    # ---------------------------------------------------------
    st.subheader("2. The Politeness Paradox")
    st.markdown("""
    When doing QA quality modeling, a fascinating phenomenon is the **Politeness Paradox**. 
    I engineered a regex feature `politeness_score` watching for words like *please*, *thanks*, *help me*, and *urgent*.
    """)
    
    # Calculate % of posts in each class that have politeness words
    polite_prop = df.groupby("Y")["politeness_score"].apply(lambda x: (x > 0).mean()).reset_index()
    fig3 = px.bar(
        polite_prop, x="Y", y="politeness_score", 
        color="Y",
        title="% of Posts containing 'Please/Thanks/Help'",
        text_auto=".1%",
        category_orders={"Y": ["HQ", "LQ_EDIT", "LQ_CLOSE"]}
    )
    fig3.update_layout(yaxis_title="Percentage Formatted Polite")
    st.plotly_chart(fig3, use_container_width=True)
    st.info("💡 **Observation**: Low Quality questions (LQ_CLOSE) are significantly more likely to use begging language ('plz help urgent'). High quality questions rely on precise problem descriptions rather than polite appeals.")

    st.divider()

    # ---------------------------------------------------------
    # Insight 3: Code-to-Text Balance
    # ---------------------------------------------------------
    st.subheader("3. Finding the Code-to-Text Golden Ratio")
    st.markdown("""
    A great programming question usually contains code, but how much code is too much? 
    I measured the proportion of characters inside `<code>` blocks relative to the entire question body length.
    """)
    
    # Addded a small eps for plotting zeros in log scale or just use linear histogram
    fig4 = px.histogram(
        df, x="body_code_ratio", color="Y",
        marginal="violin",
        barmode="overlay",
        title="Distribution of Code Character Ratio",
        category_orders={"Y": ["HQ", "LQ_EDIT", "LQ_CLOSE"]},
        nbins=50,
        opacity=0.6
    )
    st.plotly_chart(fig4, use_container_width=True)
    st.success("💡 **Observation**: Look at the heavy spike at `0.0`. A massive proportion of LQ_CLOSE questions provide **no code at all**, making them impossible to debug. Conversely, some hit ~0.95 (95% code), meaning it's just a raw script dump. HQ questions form a healthy bell curve between 10% and 50% code.")

    st.divider()

    # ---------------------------------------------------------
    # Insight 4: Title-Body Similarity
    # ---------------------------------------------------------
    st.subheader("4. Title & Body Cohesion")
    st.markdown("""
    A strong signal of effort is whether the title actually reflects the content of the body. 
    By tokenizing the text and computing the **Jaccard similarity coefficient**, I can measure vocabulary overlap.
    """)

    fig5 = px.density_contour(
        df, x="title_token_count", y="title_body_overlap",
        color="Y",
        title="Title Length vs Vocabulary Overlap",
        category_orders={"Y": ["HQ", "LQ_EDIT", "LQ_CLOSE"]},
        marginal_x="histogram",
        marginal_y="histogram"
    )
    st.plotly_chart(fig5, use_container_width=True)
    st.info("💡 **Observation**: LQ_CLOSE questions often have very short, 1-3 word titles (e.g. 'Java error') that have little overlap with the actual problem text. HQ questions have longer, descriptive titles that share vocabulary with the body.")


# ---------------------------------------------------------------------------
# App Shell
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Stack Overflow Quality Insights", layout="wide", page_icon="📈")
    
    st.title("Stack Overflow Question Quality & Insights")
    
    # Sidebar
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/e/ef/Stack_Overflow_icon.svg", width=50)
    st.sidebar.markdown("### Model Configuration")
    checkpoint_path = st.sidebar.text_input(
        "Checkpoint path",
        value=str(DEFAULT_CHECKPOINT),
        help="Path to the saved attention BiLSTM checkpoint (.pt).",
    )
    
    # Define tabs
    tab1, tab2 = st.tabs(["🤖 AI Predictor", "📊 Data EDA"])
    
    with tab1:
        try:
            model, vocab, max_len = load_model(Path(checkpoint_path))
            render_prediction_tab(model, vocab, max_len)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to load model: {exc}")
            st.info("Ensure you have run `python -m src.train_attention` to generate a checkpoint.")
            
    with tab2:
        render_eda_tab()


if __name__ == "__main__":
    main()
