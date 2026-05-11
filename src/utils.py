"""Shared helpers: JSONL I/O, dedup hashing, logging, text normalization."""
from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Iterator


# Logging
def setup_logging(level: str = "INFO") -> logging.Logger:
    # Force UTF-8 on stdout/stderr so Arabic text in log lines doesn't
    # crash on Windows' default cp1252 console.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # py3.7+
        except Exception:
            pass
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    return logging.getLogger("ssdp")


# JSONL
def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    """Atomic-enough append. Each record on one line, UTF-8, no ASCII escaping."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# IDs / dedup
def text_hash(text: str) -> str:
    """Stable short ID for dedup. Normalizes whitespace and Arabic forms."""
    normalized = normalize_arabic(text).strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]


# Arabic text handling
# Diacritics (tashkeel) - usually absent in real Egyptian text.
# Removing them from prompts gives more realistic STT training data.
_ARABIC_DIACRITICS = re.compile(r"[ً-ْٰـ]")
# Tatweel ـ (kashida) - purely decorative, never spoken
_TATWEEL = "ـ"
# Quranic / extended marks
_QURANIC_MARKS = re.compile(r"[ۖ-ۭ]")


def normalize_arabic(text: str) -> str:
    """Light normalization for dedup hashing only.
    Does NOT replace alefs / yas - that would alter dialect-meaningful text.
    """
    text = unicodedata.normalize("NFC", text)
    text = _ARABIC_DIACRITICS.sub("", text)
    text = _QURANIC_MARKS.sub("", text)
    text = text.replace(_TATWEEL, "")
    return text


def looks_like_arabic(text: str, min_ratio: float = 0.3) -> bool:
    """Sanity-check: at least `min_ratio` of non-space chars are Arabic.
    Lets through code-switched text (Arabic + English numerics/words),
    rejects pure-English or pure-emoji output.
    """
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return False
    arabic = sum(1 for c in chars if "؀" <= c <= "ۿ")
    return arabic / len(chars) >= min_ratio


def clean_prompt_text(text: str) -> str:
    """Normalize a prompt for TTS input.
    - Strip leading/trailing whitespace and quotes
    - Collapse internal whitespace
    - Remove zero-width chars and BOM
    - Strip emojis and pictographs (TTS reads them as awkward "emoji-name")
    """
    text = text.strip().strip('"""\'')
    text = text.replace("﻿", "").replace("​", "").replace("‌", "").replace("‍", "")
    text = re.sub(r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def word_count(text: str) -> int:
    return len(text.split())
