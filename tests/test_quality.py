"""Tests for the QA-only Arabic normalization used in round-trip CER."""
from __future__ import annotations

from src.quality import normalize_for_qa


def test_alef_variants_folded():
    # All alef forms collapse to bare alef so Whisper isn't penalized
    # for picking a different variant than the prompt.
    for variant in ("أكلت", "إكلت", "آكلت", "اكلت"):
        assert normalize_for_qa(variant) == "اكلت"


def test_ta_marbuta_folded_to_ha():
    # Whisper commonly outputs ه where the source has ة.
    assert normalize_for_qa("مدرسة") == normalize_for_qa("مدرسه")


def test_alef_maqsura_folded_to_ya():
    assert normalize_for_qa("على") == normalize_for_qa("علي")


def test_arabic_indic_digits_normalized():
    assert normalize_for_qa("الساعة ٣") == normalize_for_qa("الساعه 3")


def test_diacritics_stripped():
    assert normalize_for_qa("اِزَّيَّك") == normalize_for_qa("ازيك")


def test_punctuation_stripped():
    assert normalize_for_qa("ازيك؟!") == normalize_for_qa("ازيك")
    assert normalize_for_qa("ازيك،") == normalize_for_qa("ازيك")


def test_whitespace_collapsed_and_lowercased():
    assert normalize_for_qa("  HELLO   World  ") == "hello world"
