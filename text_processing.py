from __future__ import annotations

import re
import string

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


STOP_WORDS = set(ENGLISH_STOP_WORDS)
PUNCT_TRANSLATION = str.maketrans("", "", string.punctuation)
EMOJI_REPLACEMENTS = {
    "😊": " smile ",
    "😀": " happy ",
    "😁": " happy ",
    "😂": " laugh ",
    "😍": " love ",
    "😘": " love ",
    "😢": " sad ",
    "😭": " cry ",
    "😡": " angry ",
    "😠": " angry ",
    "😨": " fear ",
    "😱": " surprise ",
    "😲": " surprise ",
    "❤️": " love ",
    "❤": " love ",
}


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def replace_urls_mentions_and_emojis(text: str) -> str:
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    for emoji_char, replacement in EMOJI_REPLACEMENTS.items():
        text = text.replace(emoji_char, replacement)
    return text


def tokenize_text(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text)


def simple_stem(token: str) -> str:
    """A lightweight stemmer to avoid external dependencies."""
    suffixes = ("ingly", "edly", "ingly", "edly", "ing", "edly", "edly", "ed", "ly", "ies", "ive", "es", "s")
    for suffix in suffixes:
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            if suffix == "ies":
                return token[:-3] + "y"
            return token[: -len(suffix)]
    return token


def preprocess_text(text: str) -> str:
    """Shared preprocessing for both training and inference."""
    text = text or ""
    text = text.lower()
    text = replace_urls_mentions_and_emojis(text)
    text = text.translate(PUNCT_TRANSLATION)
    text = normalize_whitespace(text)

    tokens = tokenize_text(text)
    cleaned_tokens = [
        simple_stem(token)
        for token in tokens
        if token not in STOP_WORDS and len(token) > 1
    ]
    return " ".join(cleaned_tokens)
