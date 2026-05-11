"""Tests for src.utils - text normalization and JSONL I/O.

These cover the pieces most likely to break dedup, validation, or
exporter joins if they regress.
"""
from __future__ import annotations

from pathlib import Path

from src.utils import (
    append_jsonl,
    clean_prompt_text,
    looks_like_arabic,
    normalize_arabic,
    read_jsonl,
    text_hash,
    word_count,
)


# Arabic detection
def test_looks_like_arabic_pure_arabic():
    assert looks_like_arabic("ازيك يا صديقي")


def test_looks_like_arabic_code_switched():
    # Egyptian-style code-switching should pass the 30% Arabic threshold.
    assert looks_like_arabic("ابعتلي الـ link على الواتساب")


def test_looks_like_arabic_pure_english_rejected():
    assert not looks_like_arabic("send me the link please")


def test_looks_like_arabic_empty_rejected():
    assert not looks_like_arabic("   ")


# Cleaning
def test_clean_strips_emoji():
    cleaned = clean_prompt_text("ازيك يا صحبي 😀🎉")
    assert "😀" not in cleaned
    assert "🎉" not in cleaned
    assert "ازيك يا صحبي" in cleaned


def test_clean_strips_quotes_and_collapses_whitespace():
    assert clean_prompt_text('  "ازيك   يا   صديقي"  ') == "ازيك يا صديقي"


def test_clean_strips_zero_width():
    s = "ازيك​يا‌صديقي"
    cleaned = clean_prompt_text(s)
    assert "​" not in cleaned
    assert "‌" not in cleaned


# Hashing / dedup
def test_hash_is_stable():
    assert text_hash("ازيك") == text_hash("ازيك")


def test_hash_ignores_whitespace_and_diacritics():
    # The same text with extra spaces and diacritics should hash identically
    # so we don't double-count near-duplicates.
    assert text_hash("ازيك") == text_hash("  ازيك  ")
    assert text_hash("اِزَيك") == text_hash("ازيك")


def test_hash_distinguishes_different_text():
    assert text_hash("ازيك") != text_hash("ازاي")


# Word count
def test_word_count():
    assert word_count("ازيك يا صديقي") == 3
    assert word_count("") == 0


# normalize_arabic
def test_normalize_strips_tatweel():
    assert "ـ" not in normalize_arabic("صبـــاح الخير")


def test_normalize_preserves_dialect_alef():
    # We deliberately do NOT collapse alef variants in normalize_arabic
    # because أ vs ا can be dialect-meaningful.
    assert "أ" in normalize_arabic("أنا")


# JSONL I/O
def test_jsonl_roundtrip(tmp_path: Path):
    p = tmp_path / "x.jsonl"
    append_jsonl(p, {"a": 1, "ar": "ازيك"})
    append_jsonl(p, {"a": 2})
    rows = read_jsonl(p)
    assert rows == [{"a": 1, "ar": "ازيك"}, {"a": 2}]


def test_jsonl_unicode_not_escaped(tmp_path: Path):
    """Arabic must round-trip cleanly without \\u escapes (so a human
    can grep / cat the manifest)."""
    p = tmp_path / "x.jsonl"
    append_jsonl(p, {"text": "ازيك"})
    raw = p.read_text(encoding="utf-8")
    assert "ازيك" in raw
    assert "\\u" not in raw
