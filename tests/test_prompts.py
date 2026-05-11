"""Tests for the prompt-stage validation and category-planning logic.

These don't make any LLM calls - they exercise the pure functions
that decide what to keep, what to discard, and what to ask for next.
"""
from __future__ import annotations

from collections import Counter

from src.prompts import _plan_remaining, _validate


# _validate
def test_validate_accepts_clean_arabic():
    ok, reason = _validate("ازيك يا صحبي", min_words=2, max_words=10, seen_hashes=set())
    assert ok, reason


def test_validate_rejects_too_short():
    ok, reason = _validate("ازيك", min_words=3, max_words=10, seen_hashes=set())
    assert not ok
    assert "too_short" in reason


def test_validate_rejects_too_long():
    text = " ".join(["كلمة"] * 30)
    ok, reason = _validate(text, min_words=3, max_words=10, seen_hashes=set())
    assert not ok
    assert "too_long" in reason


def test_validate_rejects_pure_english():
    ok, reason = _validate("send me the link please", min_words=3, max_words=10, seen_hashes=set())
    assert not ok
    assert reason == "not_arabic_enough"


def test_validate_rejects_duplicate():
    # Pre-seed the seen set with the hash that this text will produce.
    from src.utils import clean_prompt_text, text_hash
    text = "ازيك يا صحبي"
    pre = {text_hash(clean_prompt_text(text))}
    ok, reason = _validate(text, min_words=2, max_words=10, seen_hashes=pre)
    assert not ok
    assert reason == "duplicate"


def test_validate_accepts_code_switched():
    # Code-switched text is ~50% English, ~50% Arabic - must pass.
    ok, reason = _validate(
        "ابعتلي الـ link على الواتساب",
        min_words=3,
        max_words=15,
        seen_hashes=set(),
    )
    assert ok, reason


def test_validate_rejects_empty_after_cleaning():
    # Text that's only emoji collapses to empty after clean_prompt_text.
    ok, reason = _validate("😀🎉🌟", min_words=1, max_words=10, seen_hashes=set())
    assert not ok
    assert reason == "empty_after_cleaning"


# _plan_remaining
def _make_cfg(target: int, proportions: dict[str, float]) -> dict:
    return {"prompts": {"total_target": target, "categories": proportions}}


def test_plan_remaining_with_no_existing_prompts():
    cfg = _make_cfg(100, {"a": 0.5, "b": 0.5})
    remaining = _plan_remaining(cfg, existing=[])
    assert remaining == Counter({"a": 50, "b": 50})


def test_plan_remaining_subtracts_existing():
    cfg = _make_cfg(100, {"a": 0.5, "b": 0.5})
    existing = [{"category": "a"}] * 30 + [{"category": "b"}] * 10
    remaining = _plan_remaining(cfg, existing)
    assert remaining == Counter({"a": 20, "b": 40})


def test_plan_remaining_drops_satisfied_categories():
    cfg = _make_cfg(100, {"a": 0.5, "b": 0.5})
    existing = [{"category": "a"}] * 50  # category 'a' is at target
    remaining = _plan_remaining(cfg, existing)
    assert "a" not in remaining
    assert remaining == Counter({"b": 50})


def test_plan_remaining_does_not_go_negative():
    """If we somehow have MORE than the target, just exclude that category."""
    cfg = _make_cfg(10, {"a": 1.0})
    existing = [{"category": "a"}] * 15  # 5 over target
    remaining = _plan_remaining(cfg, existing)
    assert remaining == Counter()  # nothing positive remains
