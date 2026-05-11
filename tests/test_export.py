"""Tests for the export-time join logic.

The exporter is the final gate on what enters the dataset, so the join
rules (reviewed-only, reject filtering, flag overrides) are worth
exercising against synthetic manifests.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.export import _join_records
from src.utils import write_jsonl


@pytest.fixture
def tiny_pipeline(tmp_path: Path):
    """Build a minimal manifest set in tmp_path and return a cfg dict
    that points to those files."""
    audio_dir = tmp_path / "data" / "audio"
    audio_dir.mkdir(parents=True)

    # Pretend audio files exist (the join doesn't actually read them).
    for sid in ("p1_v", "p2_v", "p3_v"):
        (audio_dir / f"{sid}.wav").write_bytes(b"\x00")

    audio_manifest = tmp_path / "audio.jsonl"
    write_jsonl(audio_manifest, [
        {"id": "p1_v", "prompt_id": "p1", "text": "T1", "category": "daily",
         "voice_provider": "edge", "voice_id": "ar-EG-SalmaNeural",
         "voice_gender": "female", "voice_dialect": "egyptian",
         "audio_path": "data/audio/p1_v.wav", "duration_sec": 1.5,
         "sample_rate": 16000, "status": "ok"},
        {"id": "p2_v", "prompt_id": "p2", "text": "T2", "category": "daily",
         "voice_provider": "edge", "voice_id": "ar-EG-ShakirNeural",
         "voice_gender": "male", "voice_dialect": "egyptian",
         "audio_path": "data/audio/p2_v.wav", "duration_sec": 2.0,
         "sample_rate": 16000, "status": "ok"},
        {"id": "p3_v", "prompt_id": "p3", "text": "T3", "category": "daily",
         "voice_provider": "gtts", "voice_id": "ar",
         "voice_gender": "unknown", "voice_dialect": "msa_leaning",
         "audio_path": "data/audio/p3_v.wav", "duration_sec": 1.8,
         "sample_rate": 16000, "status": "ok"},
    ])

    quality_manifest = tmp_path / "quality.jsonl"
    write_jsonl(quality_manifest, [
        {"id": "p1_v", "auto_status": "ok", "flags": [], "cer": 0.05, "wer": 0.10, "transcript": "t1"},
        {"id": "p2_v", "auto_status": "flagged", "flags": ["high_cer"], "cer": 0.50, "wer": 0.55, "transcript": "tt"},
        {"id": "p3_v", "auto_status": "flagged", "flags": ["too_short"], "cer": 0.10, "wer": 0.10, "transcript": "t3"},
    ])

    reviewed_manifest = tmp_path / "reviewed.jsonl"
    write_jsonl(reviewed_manifest, [
        {"id": "p1_v", "decision": "accept", "final_text": None, "note": None},
        {"id": "p2_v", "decision": "reject", "final_text": None, "note": "TTS skipped a word"},
        {"id": "p3_v", "decision": "edit", "final_text": "T3 corrected", "note": None},
    ])

    cfg = {
        "_root": str(tmp_path),
        "paths": {
            "audio_manifest": str(audio_manifest),
            "quality_manifest": str(quality_manifest),
            "reviewed_manifest": str(reviewed_manifest),
            "final_dir": str(tmp_path / "final"),
        },
        "export": {
            "include_rejected": False,
            "include_flagged": False,
            "dataset_name": "test_v1",
            "format": "jsonl_only",
        },
        "tts": {"output_sample_rate": 16000},
    }
    return cfg


def test_default_excludes_rejected(tiny_pipeline):
    rows = _join_records(tiny_pipeline)
    ids = {r["id"] for r in rows}
    # p2 was rejected -> excluded.
    assert "p2_v" not in ids


def test_human_accept_overrides_auto_flag(tiny_pipeline):
    rows = _join_records(tiny_pipeline)
    ids = {r["id"] for r in rows}
    # p3 was auto-flagged but the reviewer chose 'edit' -> keep.
    assert "p3_v" in ids


def test_edit_decision_uses_final_text(tiny_pipeline):
    rows = _join_records(tiny_pipeline)
    by_id = {r["id"]: r for r in rows}
    assert by_id["p3_v"]["text"] == "T3 corrected"
    assert by_id["p3_v"]["original_prompt"] == "T3"


def test_unreviewed_excluded(tiny_pipeline):
    # Add a 4th sample without a review entry; it should not appear.
    audio_path = Path(tiny_pipeline["paths"]["audio_manifest"])
    rows = []
    with open(audio_path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(line)
    import json as _json
    rows.append(_json.dumps({
        "id": "p4_v", "prompt_id": "p4", "text": "T4", "category": "daily",
        "voice_provider": "edge", "voice_id": "ar-EG-SalmaNeural",
        "voice_gender": "female", "voice_dialect": "egyptian",
        "audio_path": "data/audio/p4_v.wav", "duration_sec": 2.0,
        "sample_rate": 16000, "status": "ok",
    }, ensure_ascii=False) + "\n")
    audio_path.write_text("".join(rows), encoding="utf-8")

    out = _join_records(tiny_pipeline)
    assert "p4_v" not in {r["id"] for r in out}


def test_include_rejected_flag(tiny_pipeline):
    tiny_pipeline["export"]["include_rejected"] = True
    rows = _join_records(tiny_pipeline)
    assert "p2_v" in {r["id"] for r in rows}
