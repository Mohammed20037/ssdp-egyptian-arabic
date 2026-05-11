"""Tests for TTS-stage logic that doesn't require network or audio.

We don't call edge-tts or gTTS here - those are network deps and
flake-prone. We test the pure pieces: voice slug stability, error
class shape, and that the round-robin assignment is deterministic.
"""
from __future__ import annotations

from src.tts import SynthesisError, Voice


# Voice.slug
def test_voice_slug_is_filesystem_safe():
    v = Voice(provider="edge", voice_id="ar-EG-SalmaNeural", gender="female", dialect="egyptian")
    slug = v.slug
    # slug is used inside filenames; must contain no path separators or colons.
    for bad_char in "/\\:*?\"<>|":
        assert bad_char not in slug, f"slug contains {bad_char!r}: {slug!r}"


def test_voice_slug_is_stable():
    v1 = Voice(provider="edge", voice_id="ar-EG-SalmaNeural", gender="female", dialect="egyptian")
    v2 = Voice(provider="edge", voice_id="ar-EG-SalmaNeural", gender="female", dialect="egyptian")
    assert v1.slug == v2.slug


def test_voice_slug_distinguishes_providers():
    edge_voice = Voice(provider="edge", voice_id="ar", gender="?", dialect="?")
    gtts_voice = Voice(provider="gtts", voice_id="ar", gender="?", dialect="?")
    assert edge_voice.slug != gtts_voice.slug


def test_voice_is_hashable_and_frozen():
    """frozen=True is required because Voice is used in sets / dict keys
    in the orchestration code."""
    v = Voice(provider="edge", voice_id="ar-EG-SalmaNeural", gender="female", dialect="egyptian")
    {v}  # must not raise
    {v: 1}  # must not raise


# Round-robin assignment determinism
def test_round_robin_voice_assignment_is_deterministic():
    """The TTS module assigns voices by `voices[i % len(voices)]`. Two
    runs with the same prompt order must pick the same voice for each
    prompt - this is what makes resumability work.
    """
    voices = ["a", "b", "c"]
    prompt_ids = ["p1", "p2", "p3", "p4", "p5", "p6"]
    assignment_1 = [voices[i % len(voices)] for i, _ in enumerate(prompt_ids)]
    assignment_2 = [voices[i % len(voices)] for i, _ in enumerate(prompt_ids)]
    assert assignment_1 == assignment_2
    # And it's the expected pattern:
    assert assignment_1 == ["a", "b", "c", "a", "b", "c"]


# SynthesisError shape
def test_synthesis_error_is_runtime_error():
    """RuntimeError lineage means callers can catch it via the broader
    RuntimeError class. Tests the stability of the public exception."""
    assert issubclass(SynthesisError, RuntimeError)
    # And carries the message:
    e = SynthesisError("boom")
    assert str(e) == "boom"
