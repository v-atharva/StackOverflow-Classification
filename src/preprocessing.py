"""Utility functions for cleaning and tokenising Stack Overflow questions."""

from __future__ import annotations

import html
import re
from typing import Iterable, List


CODE_PATTERN = re.compile(r"<code>.*?</code>", re.DOTALL | re.IGNORECASE)
PRE_PATTERN = re.compile(r"<pre>.*?</pre>", re.DOTALL | re.IGNORECASE)
TAG_PATTERN = re.compile(r"<[^>]+>")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MARKDOWN_CODE_PATTERN = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE_PATTERN = re.compile(r"`[^`]+`")
WHITESPACE = re.compile(r"\s+")
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_#+\.\-]+|[^\s]")


def strip_code_blocks(text: str) -> str:
    """Replace code regions with a placeholder token."""
    placeholder = " CODETOKEN "
    text = CODE_PATTERN.sub(placeholder, text)
    text = PRE_PATTERN.sub(placeholder, text)
    text = MARKDOWN_CODE_PATTERN.sub(placeholder, text)
    text = INLINE_CODE_PATTERN.sub(placeholder, text)
    return text


def remove_html(text: str) -> str:
    """Remove HTML tags while keeping textual content."""
    return TAG_PATTERN.sub(" ", text)


def normalize_whitespace(text: str) -> str:
    return WHITESPACE.sub(" ", text).strip()


def clean_text(title: str, body: str, lowercase: bool = True) -> str:
    """Clean and join title/body content."""
    combined = f"{title or ''}. {body or ''}"
    combined = html.unescape(combined)
    combined = strip_code_blocks(combined)
    combined = URL_PATTERN.sub(" URLTOKEN ", combined)
    combined = remove_html(combined)
    combined = normalize_whitespace(combined)
    if lowercase:
        combined = combined.lower()
    return combined


def tokenize(text: str) -> List[str]:
    """Lightweight regex tokenizer preserving programming tokens."""
    return TOKEN_PATTERN.findall(text)


def preprocess(title: str, body: str) -> List[str]:
    """Full preprocessing pipeline returning tokens."""
    clean = clean_text(title, body)
    return tokenize(clean)


def detokenize(tokens: Iterable[str]) -> str:
    """Join tokens back into a display string."""
    return " ".join(tokens)

