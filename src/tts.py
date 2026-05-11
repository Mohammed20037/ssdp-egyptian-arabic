"""Stage 2: TTS synthesis.

Uses edge-tts (Salma, Shakir) and gTTS to synthesize each prompt as
16kHz mono PCM_16 WAV. One voice per prompt, picked round-robin by
prompt index so reruns are deterministic. Output manifest is
append-only JSONL, so an interrupted run resumes cleanly.

gTTS is included on purpose. It's MSA-leaning, but that voice
diversity helps the downstream model generalize away from a single
TTS engine's acoustic fingerprint.
"""
from __future__ import annotations

import asyncio
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import edge_tts
import librosa
import soundfile as sf
from gtts import gTTS
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

from .config import load_config
from .utils import append_jsonl, read_jsonl, setup_logging

log = setup_logging()


@dataclass(frozen=True)
class Voice:
    provider: str
    voice_id: str
    gender: str
    dialect: str

    @property
    def slug(self) -> str:
        # Filesystem-safe identifier used in audio filenames.
        return f"{self.provider}-{self.voice_id}".replace("/", "_").replace(":", "_")


class SynthesisError(RuntimeError):
    pass


# Provider implementations
@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=10),
    reraise=True,
)
async def _synth_edge(text: str, voice_id: str, mp3_out: Path, timeout: float) -> None:
    communicate = edge_tts.Communicate(text=text, voice=voice_id)
    try:
        await asyncio.wait_for(communicate.save(str(mp3_out)), timeout=timeout)
    except asyncio.TimeoutError as e:
        raise SynthesisError(f"edge-tts timeout after {timeout}s") from e
    if not mp3_out.exists() or mp3_out.stat().st_size < 256:
        raise SynthesisError("edge-tts produced empty/tiny file")


def _synth_gtts_sync(text: str, lang: str, mp3_out: Path) -> None:
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save(str(mp3_out))
    if not mp3_out.exists() or mp3_out.stat().st_size < 256:
        raise SynthesisError("gtts produced empty/tiny file")


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=10),
    reraise=True,
)
async def _synth_gtts(text: str, lang: str, mp3_out: Path, timeout: float) -> None:
    await asyncio.wait_for(
        asyncio.to_thread(_synth_gtts_sync, text, lang, mp3_out),
        timeout=timeout,
    )


# MP3 -> 16kHz mono WAV conversion
def _mp3_to_wav(mp3_path: Path, wav_path: Path, sample_rate: int) -> float:
    """Convert MP3 to mono PCM_16 WAV at the target sample rate.

    Returns duration in seconds. Uses librosa for decode (handles MP3
    via libsndfile or audioread+ffmpeg) and soundfile for encode.
    """
    y, _ = librosa.load(str(mp3_path), sr=sample_rate, mono=True)
    if len(y) == 0:
        raise SynthesisError("decoded audio is empty")
    sf.write(str(wav_path), y, sample_rate, subtype="PCM_16")
    return float(len(y) / sample_rate)


# Single-sample synthesis
async def _synthesize_one(
    sem: asyncio.Semaphore,
    prompt: dict[str, Any],
    voice: Voice,
    audio_dir: Path,
    project_root: Path,
    sample_rate: int,
    timeout: float,
) -> dict[str, Any]:
    async with sem:
        text = prompt["text"]
        prompt_id = prompt["id"]
        sample_id = f"{prompt_id}_{voice.slug}"
        wav_path = audio_dir / f"{sample_id}.wav"

        record: dict[str, Any] = {
            "id": sample_id,
            "prompt_id": prompt_id,
            "text": text,
            "category": prompt.get("category"),
            "voice_provider": voice.provider,
            "voice_id": voice.voice_id,
            "voice_gender": voice.gender,
            "voice_dialect": voice.dialect,
            "audio_path": None,
            "duration_sec": None,
            "sample_rate": sample_rate,
            "status": "failed",
            "error": None,
        }

        # Per-sample temp MP3 inside audio_dir so cleanup is local.
        # On Windows, NamedTemporaryFile holds an exclusive lock, so we
        # close the handle and reopen via path.
        fd, tmp_name = tempfile.mkstemp(suffix=".mp3", dir=str(audio_dir))
        os.close(fd)
        mp3_path = Path(tmp_name)

        try:
            if voice.provider == "edge":
                await _synth_edge(text, voice.voice_id, mp3_path, timeout)
            elif voice.provider == "gtts":
                await _synth_gtts(text, voice.voice_id, mp3_path, timeout)
            else:
                raise SynthesisError(f"unknown provider: {voice.provider!r}")

            duration = _mp3_to_wav(mp3_path, wav_path, sample_rate)
            record["audio_path"] = str(wav_path.relative_to(project_root)).replace("\\", "/")
            record["duration_sec"] = round(duration, 3)
            record["status"] = "ok"
        except Exception as e:
            record["error"] = f"{type(e).__name__}: {e}"
            log.warning("Synth failed [%s | %s]: %s", prompt_id, voice.slug, e)
        finally:
            try:
                mp3_path.unlink(missing_ok=True)
            except OSError:
                pass

        return record


# Orchestration
async def synthesize_async(config_path: str | None = None) -> Path:
    cfg = load_config(config_path)
    tts_cfg = cfg["tts"]
    paths = cfg["paths"]
    project_root = Path(cfg["_root"])

    prompts = read_jsonl(paths["prompts_manifest"])
    if not prompts:
        raise RuntimeError(
            "No prompts found at " + paths["prompts_manifest"] +
            ". Run prompt generation first (python -m src.prompts)."
        )

    audio_dir = Path(paths["audio_dir"])
    audio_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(paths["audio_manifest"])

    # Resumability: skip already-completed (prompt_id, voice) pairs.
    existing = read_jsonl(out_path)
    done_ids: set[str] = {r["id"] for r in existing if r.get("status") == "ok"}
    log.info("Loaded %d audio records (%d ok). Will skip those.",
             len(existing), len(done_ids))

    voices = [Voice(**v) for v in tts_cfg["voices"]]
    sample_rate = int(tts_cfg["output_sample_rate"])
    concurrency = int(tts_cfg["concurrency"])
    timeout = float(tts_cfg["timeout"])

    # Round-robin voice assignment by prompt index.
    # Stable across reruns because prompts.jsonl is append-only.
    sem = asyncio.Semaphore(concurrency)
    tasks: list[asyncio.Task] = []
    n_skipped = 0
    for i, prompt in enumerate(prompts):
        voice = voices[i % len(voices)]
        sample_id = f"{prompt['id']}_{voice.slug}"
        if sample_id in done_ids:
            n_skipped += 1
            continue
        tasks.append(asyncio.create_task(
            _synthesize_one(sem, prompt, voice, audio_dir, project_root, sample_rate, timeout)
        ))

    if not tasks:
        log.info("Nothing to do - all %d prompts already synthesized.", len(prompts))
        return out_path

    log.info("Synthesizing %d samples (skipped %d) | concurrency=%d",
             len(tasks), n_skipped, concurrency)

    n_ok = 0
    pbar = tqdm(total=len(tasks), desc="Synthesizing", unit="clip")
    for coro in asyncio.as_completed(tasks):
        record = await coro
        append_jsonl(out_path, record)
        if record["status"] == "ok":
            n_ok += 1
        pbar.update(1)
    pbar.close()

    log.info("Synthesis complete. %d/%d ok.", n_ok, len(tasks))
    return out_path


def synthesize(config_path: str | None = None) -> Path:
    return asyncio.run(synthesize_async(config_path))


if __name__ == "__main__":
    synthesize()
