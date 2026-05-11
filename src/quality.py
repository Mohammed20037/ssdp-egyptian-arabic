"""Automated quality signals.

For each synthesized clip we compute three things:

  - round-trip ASR with faster-whisper, then CER/WER vs. the source
    prompt (after Arabic-aware normalization);
  - duration sanity (too short or too long);
  - speaking rate in chars/sec (too slow = silence padding,
    too fast = mumbling).

Samples are flagged, never auto-rejected. The flag just promotes them
to the top of the review queue. We use faster-whisper `small` int8 on
CPU because it's free, fast enough on a laptop, and reasonable on
Arabic. The Arabic normalization (alef variants, ta-marbuta, digits,
diacritics, punctuation) is QA-only and does not touch the dataset
text.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jiwer
import librosa
from faster_whisper import WhisperModel
from tqdm import tqdm

from .config import load_config
from .utils import append_jsonl, read_jsonl, setup_logging

log = setup_logging()
# Arabic normalization for fair text-to-text comparison
_DIACRITICS = re.compile(r"[ً-ْٰـ]")  # tashkeel + tatweel
# Strip anything that isn't a word char or whitespace. \w in Python 3 is
# unicode-aware so Arabic letters survive; Arabic punctuation (؟ ، ؛)
# does not.
_PUNCT = re.compile(r"[^\w\s]+")
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")       # Arabic-Indic -> ASCII


def normalize_for_qa(text: str) -> str:
    """Aggressive normalization used ONLY for round-trip CER/WER.

    Folds alef variants, ta marbuta, alef maqsura, removes diacritics
    and punctuation, normalizes Arabic-Indic digits to ASCII, collapses
    whitespace. This makes Whisper's output comparable to our prompt
    without penalizing dialect-irrelevant differences.
    """
    text = unicodedata.normalize("NFC", text)
    text = _DIACRITICS.sub("", text)
    text = text.translate(_DIGITS)
    # Alef family -> bare alef
    text = re.sub(r"[آأإ]", "ا", text)  # آ أ إ -> ا
    # Ta marbuta -> ha (common reader confusion)
    text = text.replace("ة", "ه")                 # ة -> ه
    # Alef maqsura -> ya
    text = text.replace("ى", "ي")                 # ى -> ي
    # Strip punctuation (incl. Arabic punctuation)
    text = _PUNCT.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


# Whisper model - loaded once, cached at module level
@dataclass
class WhisperConfig:
    model: str
    compute: str
    device: str
    language: str


_MODEL_CACHE: dict[tuple, WhisperModel] = {}


def _get_model(cfg: WhisperConfig) -> WhisperModel:
    key = (cfg.model, cfg.compute, cfg.device)
    if key not in _MODEL_CACHE:
        log.info("Loading Whisper model %s (compute=%s, device=%s) ...",
                 cfg.model, cfg.compute, cfg.device)
        _MODEL_CACHE[key] = WhisperModel(
            cfg.model,
            device=cfg.device,
            compute_type=cfg.compute,
        )
    return _MODEL_CACHE[key]


def _transcribe(model: WhisperModel, audio_path: Path, language: str) -> str:
    segments, _info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=1,           # greedy is fine for QA - we don't need top-quality decode
        vad_filter=False,      # the clips are already short and clean
        without_timestamps=True,
    )
    return " ".join(seg.text.strip() for seg in segments).strip()


# Per-sample quality computation
def _compute_signals(
    sample: dict[str, Any],
    project_root: Path,
    model: WhisperModel,
    qcfg: dict[str, Any],
    whisper_lang: str,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": sample["id"],
        "prompt_id": sample["prompt_id"],
        "audio_path": sample["audio_path"],
        "duration_sec": sample.get("duration_sec"),
        "transcript": None,
        "cer": None,
        "wer": None,
        "chars_per_sec": None,
        "flags": [],
        "auto_status": "ok",
        "error": None,
    }

    audio_path = project_root / sample["audio_path"]
    if not audio_path.exists():
        out["flags"].append("missing_audio")
        out["auto_status"] = "error"
        out["error"] = f"audio file not found: {audio_path}"
        return out

    # Duration check (recomputed from file rather than trusting manifest)
    try:
        duration = float(librosa.get_duration(path=str(audio_path)))
        out["duration_sec"] = round(duration, 3)
    except Exception as e:
        out["flags"].append("duration_read_error")
        out["auto_status"] = "error"
        out["error"] = f"librosa duration: {e}"
        return out

    if duration < qcfg["min_duration_sec"]:
        out["flags"].append("too_short")
    if duration > qcfg["max_duration_sec"]:
        out["flags"].append("too_long")

    # Speaking rate (chars / sec, on the source prompt)
    text = sample["text"]
    n_chars = len(text.replace(" ", ""))
    cps = n_chars / max(duration, 1e-6)
    out["chars_per_sec"] = round(cps, 2)
    if cps < qcfg["min_chars_per_second"]:
        out["flags"].append("too_slow")
    if cps > qcfg["max_chars_per_second"]:
        out["flags"].append("too_fast")

    # Round-trip ASR
    try:
        hyp = _transcribe(model, audio_path, whisper_lang)
        out["transcript"] = hyp
        ref_norm = normalize_for_qa(text)
        hyp_norm = normalize_for_qa(hyp)
        if ref_norm and hyp_norm:
            out["cer"] = round(float(jiwer.cer(ref_norm, hyp_norm)), 3)
            out["wer"] = round(float(jiwer.wer(ref_norm, hyp_norm)), 3)
            if out["cer"] > qcfg["max_cer"]:
                out["flags"].append("high_cer")
        elif not hyp_norm:
            out["flags"].append("empty_transcript")
    except Exception as e:
        out["flags"].append("asr_error")
        out["error"] = f"asr: {e}"
        log.warning("ASR failed for %s: %s", sample["id"], e)

    if out["flags"] and out["auto_status"] == "ok":
        out["auto_status"] = "flagged"

    return out


# Orchestration
def compute_quality(config_path: str | None = None) -> Path:
    cfg = load_config(config_path)
    qcfg = cfg["quality"].copy()
    paths = cfg["paths"]
    project_root = Path(cfg["_root"])

    # The original config used "max_wer" as the gate; we actually drive on CER
    # (more robust for Arabic). Map either key for backward compat.
    qcfg.setdefault("max_cer", qcfg.get("max_wer", 0.40))

    audio_records = [r for r in read_jsonl(paths["audio_manifest"]) if r.get("status") == "ok"]
    if not audio_records:
        raise RuntimeError("No successful audio records to evaluate.")

    out_path = Path(paths["quality_manifest"])
    existing = read_jsonl(out_path)
    done_ids: set[str] = {r["id"] for r in existing}
    pending = [r for r in audio_records if r["id"] not in done_ids]

    if not pending:
        log.info("Quality already computed for all %d samples.", len(audio_records))
        return out_path

    log.info("Computing quality for %d samples (skipped %d cached) ...",
             len(pending), len(done_ids))

    wcfg = WhisperConfig(
        model=qcfg["whisper_model"],
        compute=qcfg["whisper_compute"],
        device=qcfg["whisper_device"],
        language=qcfg["whisper_language"],
    )
    model = _get_model(wcfg)

    pbar = tqdm(pending, desc="Quality", unit="clip")
    n_flagged = 0
    n_errors = 0
    for sample in pbar:
        rec = _compute_signals(sample, project_root, model, qcfg, wcfg.language)
        append_jsonl(out_path, rec)
        if rec["auto_status"] == "flagged":
            n_flagged += 1
        elif rec["auto_status"] == "error":
            n_errors += 1
        pbar.set_postfix(flagged=n_flagged, errors=n_errors)
    pbar.close()

    log.info("Quality done. flagged=%d errors=%d ok=%d",
             n_flagged, n_errors, len(pending) - n_flagged - n_errors)
    return out_path


if __name__ == "__main__":
    compute_quality()
