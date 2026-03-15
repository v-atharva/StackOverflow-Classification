"""Handcrafted feature engineering utilities for baseline models.

Improvements over original version
-----------------------------------
* **Body-code ratio** now measures actual code *characters* rather than
  counting ``<code>`` tag occurrences.
* **Temporal leakage** removed — ``creation_epoch_days`` replaced with
  safe cyclical features (hour, dayofweek, month).
* **New high-signal features** added: title–body overlap, num_paragraphs,
  num_list_items, politeness markers, body-to-title length ratio,
  code-to-text character ratio, and more.
* **Data-leakage-safe imputation** — ``build_feature_frame`` now accepts
  optional ``fill_values`` computed on the training set and applies them
  to any split, instead of calling ``fillna(df.mean())`` independently.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from . import preprocessing

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------
SENTENCE_SPLIT = re.compile(r"[.!?]+")
WORD_PATTERN = re.compile(r"[A-Za-z0-9_#+\-]+")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
CODE_SNIPPET_PATTERN = re.compile(r"<code>.*?</code>", re.DOTALL | re.IGNORECASE)
LIST_ITEM_PATTERN = re.compile(r"<li>|^\s*[-*]\s", re.MULTILINE)
PARAGRAPH_PATTERN = re.compile(r"<p>|</p>|\n\s*\n")
IMG_PATTERN = re.compile(r"<img\b", re.IGNORECASE)
POLITENESS_WORDS = re.compile(
    r"\b(please|thanks|thank you|grateful|appreciate|help me|sorry|urgent)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Readability
# ---------------------------------------------------------------------------

def count_syllables(word: str) -> int:
    word = word.lower()
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def readability_metrics(text: str) -> Dict[str, float]:
    sentences = [s for s in SENTENCE_SPLIT.split(text) if s.strip()]
    num_sentences = max(len(sentences), 1)
    words = WORD_PATTERN.findall(text)
    num_words = max(len(words), 1)
    syllables = sum(count_syllables(word) for word in words)
    flesch_reading_ease = (
        206.835
        - 1.015 * (num_words / num_sentences)
        - 84.6 * (syllables / num_words)
    )
    flesch_kincaid_grade = (
        0.39 * (num_words / num_sentences)
        + 11.8 * (syllables / num_words)
        - 15.59
    )
    avg_word_length = sum(len(w) for w in words) / num_words
    return {
        "flesch_reading_ease": flesch_reading_ease,
        "flesch_kincaid_grade": flesch_kincaid_grade,
        "avg_sentence_length": num_words / num_sentences,
        "avg_word_length": avg_word_length,
    }


# ---------------------------------------------------------------------------
# Structural features
# ---------------------------------------------------------------------------

def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)


def _char_entropy(text: str) -> float:
    """Shannon entropy of character distribution — high entropy can signal
    gibberish or code dumps."""
    if not text:
        return 0.0
    freq: Dict[str, int] = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    total = len(text)
    return -sum((c / total) * math.log2(c / total) for c in freq.values())


def structural_features(row: pd.Series) -> Dict[str, float]:
    body = row["Body"] if isinstance(row["Body"], str) else ""
    title = row["Title"] if isinstance(row["Title"], str) else ""
    cleaned_body = preprocessing.clean_text("", body)
    cleaned_title = preprocessing.clean_text(title, "")
    words_body = preprocessing.tokenize(cleaned_body)
    words_title = preprocessing.tokenize(cleaned_title)

    tags = row.get("Tags") or ""
    tag_list = [t for t in tags.strip("<> ").split("><") if t]
    creation_date = pd.to_datetime(row["CreationDate"])

    # ---- Code content measurement (fixed: measures actual code chars) ----
    code_matches = CODE_SNIPPET_PATTERN.findall(body)
    code_chars = sum(len(m) for m in code_matches)
    body_len = max(len(body), 1)

    # ---- Title–body overlap ----
    title_set = set(w.lower() for w in words_title)
    body_set = set(w.lower() for w in words_body)

    feat = {
        # --- Original features ---
        "title_token_count": len(words_title),
        "body_token_count": len(words_body),
        "body_char_count": len(body),
        "title_char_count": len(title),
        "num_urls": len(URL_PATTERN.findall(body)),
        "num_code_blocks": len(code_matches),
        "tag_count": len(tag_list),
        "is_multitag": 1 if len(tag_list) > 2 else 0,
        "title_question_mark": 1 if "?" in title else 0,

        # --- Fixed: actual code-character ratio ---
        "body_code_ratio": code_chars / body_len,

        # --- Temporal features (safe — no absolute timestamp) ---
        "creation_hour": creation_date.hour,
        "creation_dayofweek": creation_date.dayofweek,
        "creation_month": creation_date.month,
        "is_weekend": 1 if creation_date.dayofweek >= 5 else 0,
        # Cyclical encodings to let the model know 23→0 is close
        "hour_sin": math.sin(2 * math.pi * creation_date.hour / 24),
        "hour_cos": math.cos(2 * math.pi * creation_date.hour / 24),
        "dow_sin": math.sin(2 * math.pi * creation_date.dayofweek / 7),
        "dow_cos": math.cos(2 * math.pi * creation_date.dayofweek / 7),

        # --- NEW: structural quality signals ---
        "title_body_overlap": _jaccard(title_set, body_set),
        "body_to_title_len_ratio": len(body) / max(len(title), 1),
        "code_to_text_ratio": code_chars / max(body_len - code_chars, 1),
        "num_paragraphs": len(PARAGRAPH_PATTERN.findall(body)),
        "num_list_items": len(LIST_ITEM_PATTERN.findall(body)),
        "has_image": 1 if IMG_PATTERN.search(body) else 0,
        "num_external_links": len(URL_PATTERN.findall(body)),
        "politeness_score": len(POLITENESS_WORDS.findall(body + " " + title)),
        "body_entropy": _char_entropy(body),

        # Ratio features
        "title_upper_ratio": (
            sum(1 for c in title if c.isupper()) / max(len(title), 1)
        ),
        "body_upper_ratio": (
            sum(1 for c in body if c.isupper()) / max(len(body), 1)
        ),
    }
    feat.update(readability_metrics(cleaned_body))
    return feat


# ---------------------------------------------------------------------------
# Feature frame builder (data-leakage safe)
# ---------------------------------------------------------------------------

def build_feature_frame(
    df: pd.DataFrame,
    fill_values: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Return engineered feature matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data with at least Title, Body, Tags, CreationDate columns.
    fill_values : pd.Series, optional
        Column means computed on the **training** set.  When ``None`` the
        function computes means from *df* itself — which is fine only for
        the training split.  For validation/test, always pass the training
        fill values to prevent data leakage.
    """
    features = df.apply(structural_features, axis=1).tolist()
    feature_df = pd.DataFrame(features)
    if fill_values is None:
        fill_values = feature_df.mean()
    feature_df.fillna(fill_values, inplace=True)
    return feature_df, fill_values
