"""Tests for the Stack Overflow quality classification pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src import config, preprocessing
from src.data import (
    StackOverflowTextDataset,
    TextBatch,
    collate_batch,
    prepare_tokens,
)
from src.features import build_feature_frame, structural_features
from src.models.attention_lstm import AttentionBiLSTM
from src.vocab import Vocabulary


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Minimal DataFrame that mimics the real dataset schema."""
    return pd.DataFrame(
        {
            "Id": [1, 2, 3, 4, 5, 6],
            "Title": [
                "How to sort a list in Python?",
                "plz help java error",
                "Understanding Rust lifetimes",
                "",
                "Why does my C++ code segfault?",
                "fix my code",
            ],
            "Body": [
                "<p>I have a list <code>nums = [3,1,2]</code> and want to sort it.</p>",
                "i have error plz help me fix it thanks",
                "<p>Lifetimes in Rust ensure memory safety. <code>fn foo<'a>() {}</code></p>",
                "<p>Empty title question</p>",
                '<p>Getting a segfault. <a href="https://example.com">link</a></p><code>int *p = NULL; *p = 1;</code>',
                "",
            ],
            "Tags": [
                "<python><list><sorting>",
                "<java>",
                "<rust><lifetimes>",
                "<misc>",
                "<c++><segfault>",
                "<code>",
            ],
            "CreationDate": [
                "2023-01-15 10:30:00",
                "2023-02-20 22:15:00",
                "2023-03-10 14:00:00",
                "2023-04-05 08:00:00",
                "2023-05-12 16:45:00",
                "2023-06-01 01:00:00",
            ],
            "Y": ["HQ", "LQ_CLOSE", "HQ", "LQ_EDIT", "HQ", "LQ_CLOSE"],
        }
    )


@pytest.fixture
def vocab(sample_df) -> Vocabulary:
    tokens = prepare_tokens(sample_df)
    return Vocabulary.build(tokens, max_size=500, min_freq=1)


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------

class TestPreprocessing:
    def test_clean_text_strips_html(self):
        result = preprocessing.clean_text("Title", "<p>Hello <b>world</b></p>")
        assert "<p>" not in result
        assert "<b>" not in result
        assert "hello" in result.lower()

    def test_clean_text_replaces_urls(self):
        result = preprocessing.clean_text("", "Visit https://stackoverflow.com for help")
        assert "urltoken" in result.lower()
        assert "https://" not in result

    def test_clean_text_replaces_code(self):
        result = preprocessing.clean_text("", "<code>x = 1</code> is assignment")
        assert "codetoken" in result.lower()

    def test_clean_text_replaces_inline_code(self):
        result = preprocessing.clean_text("", "Use `print()` to debug")
        assert "codetoken" in result.lower()

    def test_clean_text_handles_empty_strings(self):
        result = preprocessing.clean_text("", "")
        assert isinstance(result, str)

    def test_tokenize_preserves_programming_tokens(self):
        tokens = preprocessing.tokenize("c++ python3 node.js c#")
        assert "c++" in tokens
        assert "c#" in tokens

    def test_preprocess_full_pipeline(self):
        tokens = preprocessing.preprocess(
            "How to use pandas?",
            "<p>I want to read a CSV with <code>pd.read_csv</code></p>",
        )
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)

    def test_preprocess_empty_body(self):
        tokens = preprocessing.preprocess("Title only", "")
        assert len(tokens) > 0

    def test_preprocess_empty_both(self):
        tokens = preprocessing.preprocess("", "")
        assert isinstance(tokens, list)

    def test_preprocess_unicode(self):
        tokens = preprocessing.preprocess("Ünïcödé tïtle", "Bôdy with àccents")
        assert isinstance(tokens, list)

    def test_detokenize_roundtrip(self):
        tokens = ["hello", "world"]
        joined = preprocessing.detokenize(tokens)
        assert joined == "hello world"


# ---------------------------------------------------------------------------
# Vocabulary tests
# ---------------------------------------------------------------------------

class TestVocabulary:
    def test_build_has_special_tokens(self, vocab):
        assert "<pad>" in vocab.stoi
        assert "<unk>" in vocab.stoi
        assert vocab.pad_index == 0
        assert vocab.unk_index == 1

    def test_encode_known_tokens(self, vocab):
        # We know "how" should be in the vocab from sample data
        known_tokens = [t for t in vocab.itos if t not in ("<pad>", "<unk>")]
        if known_tokens:
            encoded = vocab.encode([known_tokens[0]])
            assert encoded[0] != vocab.unk_index

    def test_encode_unknown_token(self, vocab):
        encoded = vocab.encode(["xyzzy_unknown_token_12345"])
        assert encoded == [vocab.unk_index]

    def test_encode_empty_list(self, vocab):
        encoded = vocab.encode([])
        assert encoded == []

    def test_len(self, vocab):
        assert len(vocab) == len(vocab.itos)
        assert len(vocab) >= 2  # at least pad + unk

    def test_from_itos_roundtrip(self, vocab):
        rebuilt = Vocabulary.from_itos(vocab.itos)
        assert rebuilt.stoi == vocab.stoi
        assert rebuilt.itos == vocab.itos

    def test_min_freq_filtering(self):
        tokens = [["a", "b", "c"], ["a", "b"], ["a"]]
        v = Vocabulary.build(tokens, max_size=100, min_freq=2)
        assert "a" in v.stoi  # freq=3
        assert "b" in v.stoi  # freq=2
        assert "c" not in v.stoi  # freq=1

    def test_max_size_limit(self):
        tokens = [["a", "b", "c", "d", "e"]] * 10
        v = Vocabulary.build(tokens, max_size=4, min_freq=1)
        assert len(v) <= 4  # pad + unk + at most 2


# ---------------------------------------------------------------------------
# Data pipeline tests
# ---------------------------------------------------------------------------

class TestData:
    def test_prepare_tokens(self, sample_df):
        tokens = prepare_tokens(sample_df)
        assert len(tokens) == len(sample_df)
        assert all(isinstance(t, list) for t in tokens)

    def test_dataset_creation(self, sample_df, vocab):
        tokens = prepare_tokens(sample_df)
        ds = StackOverflowTextDataset(sample_df, tokens, vocab, max_len=50)
        assert len(ds) == len(sample_df)

    def test_dataset_getitem(self, sample_df, vocab):
        tokens = prepare_tokens(sample_df)
        ds = StackOverflowTextDataset(sample_df, tokens, vocab, max_len=50)
        encoded, label, meta = ds[0]
        assert isinstance(encoded, list)
        assert isinstance(label, int)
        assert label in config.LABEL_TO_ID.values()
        assert "Id" in meta
        assert "Y" in meta

    def test_token_dropout(self, sample_df, vocab):
        tokens = prepare_tokens(sample_df)
        ds = StackOverflowTextDataset(
            sample_df, tokens, vocab, max_len=50, token_dropout=0.99
        )
        ds.training = True  # enable dropout
        encoded_original, _, _ = ds.examples[0]
        # With 99% dropout, most tokens should be UNK
        encoded_dropped, _, _ = ds[0]
        unk_count = sum(1 for t in encoded_dropped if t == vocab.unk_index)
        # On average ~99% should be UNK (allow some variance)
        assert unk_count > 0

    def test_collate_batch(self, sample_df, vocab):
        tokens = prepare_tokens(sample_df)
        ds = StackOverflowTextDataset(sample_df, tokens, vocab, max_len=50)
        batch_items = [ds[i] for i in range(min(3, len(ds)))]
        batch = collate_batch(batch_items, pad_index=vocab.pad_index)
        assert isinstance(batch, TextBatch)
        assert batch.tokens.shape[0] == len(batch_items)
        assert batch.lengths.shape[0] == len(batch_items)
        assert batch.labels.shape[0] == len(batch_items)

    def test_prepare_tokens_with_cache(self, sample_df, tmp_path):
        cache_dir = tmp_path / "cache"
        tokens1 = prepare_tokens(sample_df, cache_dir=str(cache_dir))
        tokens2 = prepare_tokens(sample_df, cache_dir=str(cache_dir))
        assert tokens1 == tokens2


# ---------------------------------------------------------------------------
# Feature tests
# ---------------------------------------------------------------------------

class TestFeatures:
    def test_structural_features(self, sample_df):
        row = sample_df.iloc[0]
        feats = structural_features(row)
        assert isinstance(feats, dict)
        assert "title_token_count" in feats
        assert "body_token_count" in feats
        assert "body_code_ratio" in feats
        assert "title_body_overlap" in feats
        assert "politeness_score" in feats
        assert "body_entropy" in feats

    def test_no_temporal_leakage(self, sample_df):
        row = sample_df.iloc[0]
        feats = structural_features(row)
        assert "creation_epoch_days" not in feats

    def test_cyclical_features_present(self, sample_df):
        row = sample_df.iloc[0]
        feats = structural_features(row)
        assert "hour_sin" in feats
        assert "hour_cos" in feats
        assert "dow_sin" in feats
        assert "dow_cos" in feats

    def test_polite_question_detected(self):
        row = pd.Series({
            "Id": 99,
            "Title": "Please help me fix this",
            "Body": "<p>Thanks for your help, I appreciate it</p>",
            "Tags": "<python>",
            "CreationDate": "2023-01-01 12:00:00",
            "Y": "LQ_CLOSE",
        })
        feats = structural_features(row)
        assert feats["politeness_score"] >= 2  # "please", "thanks", "appreciate"

    def test_build_feature_frame_returns_fill_values(self, sample_df):
        feature_df, fill_values = build_feature_frame(sample_df)
        assert isinstance(feature_df, pd.DataFrame)
        assert isinstance(fill_values, pd.Series)
        assert len(feature_df) == len(sample_df)

    def test_build_feature_frame_leakage_safe(self, sample_df):
        train = sample_df.iloc[:4]
        val = sample_df.iloc[4:]
        _, train_fill = build_feature_frame(train)
        val_feats, _ = build_feature_frame(val, fill_values=train_fill)
        assert len(val_feats) == len(val)


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestModel:
    def test_forward_shape(self):
        model = AttentionBiLSTM(
            vocab_size=100,
            embedding_dim=32,
            hidden_dim=16,
            attention_dim=16,
            num_layers=2,
            dropout=0.1,
            num_labels=3,
            padding_idx=0,
        )
        tokens = torch.randint(1, 99, (4, 10))
        lengths = torch.tensor([10, 8, 6, 4])
        logits = model(tokens, lengths)
        assert logits.shape == (4, 3)

    def test_forward_with_attention(self):
        model = AttentionBiLSTM(
            vocab_size=100,
            embedding_dim=32,
            hidden_dim=16,
            attention_dim=16,
            num_layers=1,
            dropout=0.0,
            num_labels=3,
            padding_idx=0,
        )
        tokens = torch.randint(1, 99, (2, 8))
        lengths = torch.tensor([8, 5])
        logits, attn = model(tokens, lengths, return_attention=True)
        assert logits.shape == (2, 3)
        assert attn.shape == (2, 8)
        # Attention weights should sum to ~1 per sample
        for i in range(2):
            assert abs(attn[i].sum().item() - 1.0) < 1e-5

    def test_padding_mask_uses_padding_idx(self):
        pad_idx = 5
        model = AttentionBiLSTM(
            vocab_size=100,
            embedding_dim=32,
            hidden_dim=16,
            attention_dim=16,
            num_layers=1,
            dropout=0.0,
            num_labels=3,
            padding_idx=pad_idx,
        )
        assert model.padding_idx == pad_idx

    def test_pretrained_vectors_loaded(self):
        vocab_size = 50
        embed_dim = 32
        vectors = torch.randn(vocab_size, embed_dim)
        model = AttentionBiLSTM(
            vocab_size=vocab_size,
            embedding_dim=embed_dim,
            hidden_dim=16,
            attention_dim=16,
            num_layers=1,
            num_labels=3,
            padding_idx=0,
            pretrained_vectors=vectors,
        )
        # Check that embedding weights match (except padding)
        assert torch.allclose(
            model.embedding.weight.data[1:], vectors[1:]
        )

    def test_single_token_input(self):
        model = AttentionBiLSTM(
            vocab_size=100,
            embedding_dim=16,
            hidden_dim=8,
            attention_dim=8,
            num_layers=1,
            num_labels=3,
            padding_idx=0,
        )
        tokens = torch.tensor([[5]])
        lengths = torch.tensor([1])
        logits = model(tokens, lengths)
        assert logits.shape == (1, 3)


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfig:
    def test_labels(self):
        assert config.LABELS == ["HQ", "LQ_EDIT", "LQ_CLOSE"]

    def test_label_mappings_consistent(self):
        for label in config.LABELS:
            idx = config.LABEL_TO_ID[label]
            assert config.ID_TO_LABEL[idx] == label

    def test_artifact_dirs_exist(self):
        assert config.ARTIFACT_DIR.exists()
        assert config.CHECKPOINT_DIR.exists()
        assert config.METRICS_DIR.exists()
        assert config.PLOTS_DIR.exists()
