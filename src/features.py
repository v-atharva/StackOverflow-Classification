"""Handcrafted feature engineering utilities for baseline models."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from . import preprocessing

SENTENCE_SPLIT = re.compile(r"[.!?]+")
WORD_PATTERN = re.compile(r"[A-Za-z0-9_#\+\-]+")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
CODE_SNIPPET_PATTERN = re.compile(r"<code>.*?</code>", re.DOTALL | re.IGNORECASE)


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
    flesch_reading_ease = 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (
        syllables / num_words
    )
    flesch_kincaid_grade = 0.39 * (num_words / num_sentences) + 11.8 * (
        syllables / num_words
    ) - 15.59
    avg_word_length = sum(len(w) for w in words) / num_words
    return {
        "flesch_reading_ease": flesch_reading_ease,
        "flesch_kincaid_grade": flesch_kincaid_grade,
        "avg_sentence_length": num_words / num_sentences,
        "avg_word_length": avg_word_length,
    }


def structural_features(row: pd.Series) -> Dict[str, float]:
    body = row["Body"] or ""
    title = row["Title"] or ""
    cleaned_body = preprocessing.clean_text("", body)
    cleaned_title = preprocessing.clean_text(title, "")
    words_body = preprocessing.tokenize(cleaned_body)
    words_title = preprocessing.tokenize(cleaned_title)

    tags = row.get("Tags") or ""
    tag_list = [t for t in tags.strip("<> ").split("><") if t]
    creation_date = pd.to_datetime(row["CreationDate"])

    feat = {
        "title_token_count": len(words_title),
        "body_token_count": len(words_body),
        "body_char_count": len(body),
        "title_char_count": len(title),
        "num_urls": len(URL_PATTERN.findall(body)),
        "num_code_blocks": len(CODE_SNIPPET_PATTERN.findall(body)),
        "tag_count": len(tag_list),
        "is_multitag": 1 if len(tag_list) > 2 else 0,
        "body_code_ratio": body.count("<code>") / max(len(body), 1),
        "title_question_mark": 1 if "?" in title else 0,
        "creation_hour": creation_date.hour,
        "creation_dayofweek": creation_date.dayofweek,
        "creation_epoch_days": creation_date.timestamp() / (60 * 60 * 24),
    }
    feat.update(readability_metrics(cleaned_body))
    return feat


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return engineered feature matrix for the baseline model."""
    features = [structural_features(row) for _, row in df.iterrows()]
    feature_df = pd.DataFrame(features)
    feature_df.fillna(feature_df.mean(), inplace=True)
    return feature_df

