# Synthetic Speech Data Pipeline (S.S.D.P.)

A four-stage pipeline that produces a training-ready synthetic speech dataset for **Egyptian Arabic** STT fine-tuning.

```
[1] Prompt generation        ->  data/prompts/prompts.jsonl
[2] TTS synthesis            ->  data/audio/*.wav  +  data/audio/audio.jsonl
[2.5] Auto quality signals   ->  data/reviewed/quality.jsonl
[3] Human review (Gradio)    ->  data/reviewed/reviewed.jsonl
[4] Export (HuggingFace)     ->  data/final/egyptian_arabic_synthetic_v1/
```

Every stage is **resumable**: each writes an append-only JSONL manifest, and rerunning the stage skips work that already has a successful entry.

---

## Quickstart

```bash
# 1. Create venv (Python 3.12) and install deps
# Windows:
py -3.12 -m venv .venv
.venv\Scripts\activate
# macOS / Linux:
# python3.12 -m venv .venv
# source .venv/bin/activate
pip install -r requirements.txt

# 2. ffmpeg is required for MP3 -> WAV conversion
winget install Gyan.FFmpeg                        # Windows
# brew install ffmpeg                             # macOS
# sudo apt install ffmpeg                         # Debian/Ubuntu

# 3. Set your OpenAI key
cp .env.example .env
# edit .env, paste your real key

# 4. Run the non-interactive stages (prompts -> tts -> quality)
python -m src.pipeline --stage all

# 5. Launch the Gradio reviewer (interactive - open http://127.0.0.1:7860)
python -m src.pipeline --stage review

# 6. After review, build the final training-ready dataset
python -m src.pipeline --stage export
```

> **Note:** `--stage all` runs prompts -> tts -> quality. Review and
> export are **not** included in `all` because review is interactive
> (would hang in CI) and export should run only after review is done.

Run an individual stage with `--stage prompts | tts | quality | review | export`.

> **First-run download:** the quality stage downloads the
> `Systran/faster-whisper-small` model (~244 MB) to your HuggingFace
> cache (`~/.cache/huggingface/`). Subsequent runs reuse it.

---

## Pipeline architecture

### Stage 1 - Prompt generation ([src/prompts.py](src/prompts.py))

Generates Egyptian Arabic text prompts via OpenAI `gpt-4o-mini`.

- **Per-category prompting.** Generic "give me Egyptian Arabic" requests collapse toward MSA. Each category in [config/pipeline.yaml](config/pipeline.yaml) has a tailored Egyptian-Arabic system prompt with explicit dialect markers (إزيك، عايز، كده، ...) and explicit instructions to avoid فصحى.
- **Categories** (with target proportions): daily_conversation, customer_service, questions, **code_switching**, numbers_dates, commands_requests, emotional, formal_news.
- **Validation gate.** Every candidate runs through cleaning (strip emoji / quotes / zero-width chars) -> length check -> Arabic-content ratio ≥ 30% -> SHA1 dedup hash. Invalid candidates are counted by reason and over-asked to compensate.
- **Resumable.** Manifest is append-only JSONL keyed by content hash; a rerun continues from wherever the previous run stopped.
- **Cost.** ~$0.02 for 80 prompts at `gpt-4o-mini` and temperature 0.9 (measured on this run).

### Stage 2 - TTS synthesis ([src/tts.py](src/tts.py))

Async, bounded-concurrency synthesizer over multiple free TTS providers.

- **Voices.**
  - `edge-tts` `ar-EG-SalmaNeural` (female, Egyptian)
  - `edge-tts` `ar-EG-ShakirNeural` (male, Egyptian)
  - `gTTS` `ar` (MSA-leaning Arabic - included as a contrast voice for diversity)
- **One voice per prompt.** Round-robin assignment by prompt index. We deliberately do *not* synthesize every prompt with every voice - that creates pseudo-duplicates that mislead STT training.
- **Concurrency.** `asyncio.Semaphore` capped at 8 in-flight requests. `gTTS` (sync) is wrapped via `asyncio.to_thread`.
- **Format.** All audio standardized to **16 kHz mono PCM_16 WAV** - the canonical input format for Whisper / wav2vec2 / SeamlessM4T. The MP3 produced by both providers is treated as an intermediate and deleted.
- **Reliability.** Tenacity retries with exponential backoff on every provider call. Per-sample timeout. Empty/tiny output detection.
- **Per-sample isolation.** A failed clip writes `status="failed"` with a reason and the run continues.

### Stage 2.5 - Automated quality signals ([src/quality.py](src/quality.py))

Three signals, computed once per sample:

1. **Round-trip ASR (CER + WER).** Transcribe the synthesized audio with `faster-whisper` (model=`small`, `int8` CPU), then compare against the source prompt. High CER => TTS likely mispronounced or skipped words.
2. **Duration sanity.** Clips outside `[0.5s, 20s]` get flagged.
3. **Speaking rate.** Characters-per-second outside `[4, 25]` => silence padding (too slow) or sped-up / mumbling (too fast).

Comparison uses an **Arabic-aware QA normalization**: alef variants -> bare alef, ta marbuta -> ha, alef maqsura -> ya, Arabic-Indic digits -> ASCII, diacritics + punctuation stripped. This normalization is **QA-only** - the dataset still keeps the original prompt text. Without it, Whisper's perfectly-valid transcript variants (e.g. `مدرسه` vs `مدرسة`) would inflate CER.

Samples are **flagged**, never auto-rejected. The human reviewer makes the final call.

### Stage 3 - Review UI ([src/review.py](src/review.py))

A Gradio app at `http://localhost:7860`.

- **Flagged-first ordering.** Samples with `auto_status="flagged"` come first so reviewer attention goes where signals say it's most needed.
- **Three actions:** **Accept** / **Accept with edit** / **Reject**. Edit lets the reviewer fix a one-word transcription mismatch and keep the clip.
- **Persistent decisions.** Every action appends to `reviewed.jsonl` immediately (latest entry per id wins). The reviewer can stop and resume any time.
- **Reviewer overrides the auto-signal.** A high-CER clip might still be perfectly fine if Whisper itself struggled with the dialect; the human is the source of truth.

### Stage 4 - Export ([src/export.py](src/export.py))

Joins audio + quality + review manifests and writes:

- **`data/final/egyptian_arabic_synthetic_v1/`** - **self-contained** HuggingFace `datasets` directory (Arrow-backed). The WAVs are copied into `<dataset_dir>/audio/` so the dataset folder is one portable artifact. The `audio` column holds dataset-relative paths (e.g. `audio/<id>.wav`). Loaders resolve to absolute and cast to `Audio()`:
  ```python
  from pathlib import Path
  from datasets import load_from_disk, Audio

  ds_path = "data/final/egyptian_arabic_synthetic_v1"
  ds = load_from_disk(ds_path)
  ds = ds.map(lambda r: {"audio": str(Path(ds_path) / r["audio"])})
  ds = ds.cast_column("audio", Audio(sampling_rate=16000))
  print(ds[0]["audio"]["array"].shape, ds[0]["text"])
  ```
  Audio is **not** baked into the Arrow file as bytes - that would force this pipeline to depend on `torch` + `torchcodec`, which belong in the *training* environment, not the data env.
- **`data/final/metadata.jsonl`** - human-readable record-per-line, easy to grep / inspect.
- **`data/final/README.md`** - auto-generated dataset card with voice / category / duration stats and the same load snippet.

Default filters: include `accept` and `edit` decisions, exclude `reject`. Auto-flagged + accepted-as-is samples are excluded by default (`include_flagged: false`) on the conservative-data principle that a flagged signal you didn't act on is still a flagged signal. Override in the config.

---

## Why these choices

### Why edge-tts + gTTS (not commercial TTS)?

- **Free** with no auth required - runs on a laptop without an account.
- **edge-tts** has the only widely-available *Egyptian* voices in the free tier. Generic Arabic TTS (Polly, gTTS, etc.) reads everything in MSA, which defeats the purpose for an Egyptian-Arabic dataset.
- **gTTS** added as a third voice not because it's good Egyptian (it isn't) but for **acoustic diversity**. A model trained on a single TTS engine's audio overfits to that engine's prosody and synthesis artifacts. Mixing in MSA-leaning samples helps the model generalize to the MSA↔Egyptian register shifts that real Egyptian speakers do constantly.

### Why faster-whisper for QA (not openai-whisper)?

- ~4× faster on CPU thanks to CTranslate2.
- `int8` quantization fits comfortably on a laptop without a GPU.
- Same Whisper model architecture, so the Arabic-recognition quality is comparable.

### Why HuggingFace `datasets` for the export?

- De-facto standard for STT fine-tuning on Whisper, wav2vec2, SeamlessM4T.
- Arrow-backed, memory-mapped, zero-copy reads.
- Native `Audio()` feature lazy-loads audio on access.
- Easily uploadable to the HF Hub later (one line) if the team wants.

### Why JSONL for intermediate manifests?

- Trivially appendable -> trivially resumable.
- Trivially greppable / inspectable from the shell - no special tooling.
- One record = one line = the natural unit of progress.
- UTF-8 with `ensure_ascii=False` so Arabic round-trips visibly: `cat data/prompts/prompts.jsonl` shows real Arabic text.

---

## Observed quality issues (measured on this run)

The numbers below come from the actual 80-clip run, not hypotheticals. They're produced by [scripts/quality_breakdown.py](scripts/quality_breakdown.py) - rerun it after any pipeline change to refresh.

### Round-trip ASR (Whisper) - overall

| | mean CER | median CER | P90 CER | max CER |
|---|---|---|---|---|
| All 80 clips | **0.138** | 0.107 | 0.324 | 0.467 |

A median CER of ~0.11 means Whisper read most clips back almost exactly. The auto-flag threshold is 0.40, which only caught **2/80 = 2.5%** of clips - both were CER spikes from one number-heavy gTTS clip and one Egyptian-voice clip with a tricky proper noun.

### Per-voice breakdown

| Voice | n | mean CER | reject rate (human) |
|---|---|---|---|
| `edge ar-EG-SalmaNeural` (female, Egyptian) | 27 | 0.157 | **41%** |
| `edge ar-EG-ShakirNeural` (male, Egyptian) | 27 | 0.141 | **48%** |
| `gTTS ar` (MSA-leaning) | 26 | 0.114 | **42%** |

Counterintuitive but informative: **gTTS had the lowest CER but the same human reject rate as the Egyptian voices**. Whisper transcribes MSA cleanly, but the human reviewer correctly noticed the dialect mismatch - exactly the failure mode this pipeline must catch, and exactly why we keep a human in the loop instead of trusting CER alone.

### Per-category breakdown - the real story

| Category | n | mean CER | reject rate |
|---|---|---|---|
| `formal_news` | 4 | **0.026** | 75% |
| `customer_service` | 12 | 0.076 | 8% |
| `emotional` | 4 | 0.086 | **100%** |
| `questions` | 12 | 0.103 | 33% |
| `commands_requests` | 8 | 0.109 | 50% |
| `daily_conversation` | 20 | 0.135 | 55% |
| `numbers_dates` | 8 | 0.194 | 38% |
| `code_switching` | 12 | **0.275** | 42% |

Three concrete observations:

1. **Code-switched clips have 2.43× the CER of non-code-switched clips** (0.275 vs 0.114). English words inside Arabic sentences (`ابعتلي الـ link`) are the single biggest TTS failure mode. The TTS engines either skip the English token, transliterate it awkwardly into Arabic phonology, or read the Arabic article wrong. This is the most important finding from this run for downstream training: **a model fine-tuned on this data without real code-switched audio will fail on real Egyptian conversation, where code-switching is constant.**
2. **Number/date clips have 1.7× the CER** of conversational clips (0.194 vs 0.114). Mixed Arabic-Indic + ASCII digits, time formats, and currency expressions trip the TTS up. Mitigation in this pipeline: deliberate exposure via the `numbers_dates` category so the issue is visible.
3. **CER is a poor judge of "Egyptian-ness."** `formal_news` had the lowest CER (0.026 - Whisper loves clean MSA) but a 75% reject rate from the human reviewer because the voices didn't sound dialect-appropriate. `emotional` had 100% rejection - short emotional bursts (`يااااه`, `إيه ده`) confuse TTS prosody and the audio comes out flat. Fix: prompts that lean on dialect markers without being formal MSA.

### Where the human reviewer overruled the auto-signal

Of the 80 clips reviewed, **only 2 were auto-flagged** by the CER + duration + speaking-rate signals. The human reviewer rejected **35 (44%)** and edited **27 (34%)**, meaning 71% of the human's calls were *not* surfaced by the auto-signal - most rejections were for **dialect quality**, not measurable transcription error.

Concrete implication: the auto-signal catches the cases where TTS *fails to say what was asked*. It does **not** catch the cases where TTS says what was asked but says it in the wrong voice / register / dialect. **Those failures require human review or a dialect-classifier auto-signal that this pipeline does not have.** That's a documented limitation, not an oversight.

### Synthesis reliability

| | count |
|---|---|
| Synthesis attempts | 80 |
| Successes | **80** |
| Failures | 0 |
| Retries triggered | 0 |
| Whisper QA errors | 0 |

Both edge-tts and gTTS were stable on a Windows laptop with home internet. Tenacity's retry-with-backoff layer was never invoked during this run, but it's there because production runs at 10× this volume *will* trigger occasional 5xx from both providers.

---

## Egyptian-Arabic-specific challenges (and how the pipeline addresses them)

| Challenge | Why it matters | Mitigation |
|---|---|---|
| TTS engines default to MSA, not Egyptian | Dialect-meaningful words (إزيك، عايز، كده) are pronounced wrong | Explicit Egyptian voices (Salma/Shakir); per-category prompts that hammer dialect markers; gTTS only as a *contrast* voice |
| Code-switching English ↔ Arabic | Egyptians constantly mix English nouns into Arabic sentences ("ابعتلي الـ link") - most TTS reads them awkwardly | Dedicated `code_switching` category; the auto-CER signal catches the worst cases for review |
| Arabic-Indic vs ASCII digits | TTS engines treat ٣ and 3 differently | `numbers_dates` category mixes both intentionally so the dataset learns to handle either |
| Diacritics (tashkeel) | Real Egyptian text rarely has diacritics, but LLM output sometimes does | Cleaning step strips them from prompt text before it reaches TTS |
| Alef / ya / ta-marbuta variants | Whisper transcripts often substitute `ه`↔`ة`, `ي`↔`ى`, `ا`↔`أ`/`إ` | QA normalization folds these for fair CER calculation; dataset keeps original spellings |
| Emojis, zero-width chars, kashida | LLMs sneak these in; TTS reads them as awkward "emoji-name" text | `clean_prompt_text` strips them before validation |

---

## Synthetic-data pitfalls (and how the pipeline addresses them)

> Awareness of how synthetic data can mislead a downstream model is half the job. A 100% synthetic dataset is *not* a substitute for real data; it's a complement.

**Pitfall 1 - Acoustic homogeneity.** A model trained only on edge-tts audio will overfit to its 24 kHz neural-vocoder fingerprint and fail on real microphones, real noise, real reverberation.
**->** We deliberately mix two Egyptian voices + one MSA-leaning voice for *some* acoustic variation. We resample to 16 kHz (matching the Whisper input rate) so models don't learn to expect 24 kHz artifacts. We document this clearly so downstream consumers know to mix with real data.

**Pitfall 2 - Prosodic monoculture.** TTS prosody is too clean - no hesitations, no false starts, no overlap, no laughter, no breath.
**->** No fix in the synthetic stage; the README's "Caveats" section is explicit about this so downstream consumers don't think a 100%-synthetic-trained model will work on natural conversation.

**Pitfall 3 - Distribution drift between text and speech.** LLM-generated text doesn't always reflect how Egyptians actually speak - too well-formed, too clean, too on-topic.
**->** Explicit `emotional` and `daily_conversation` categories with informality markers (يااااه، طب، يلا، يعني) push the LLM toward natural-sounding text.

**Pitfall 4 - Hallucinated text the TTS can't actually pronounce.** The LLM might emit a rare word the TTS mishandles, and we'd never know.
**->** The round-trip ASR signal exists exactly for this. High CER => flag for human review. The reviewer can either reject or fix the text via "Accept with edit".

**Pitfall 5 - Test-set leakage.** If the same synthetic data is used for both train and eval, every synthesis quirk is shared and metrics look artificially good.
**->** Out of scope for this pipeline (we don't split). The downstream consumer must hold out a real-recordings-only test set.

---

## Project layout

```
ssdp/
├-- config/pipeline.yaml         # All knobs: voices, categories, thresholds, paths
├-- src/
|   ├-- config.py                # Loads YAML + .env
|   ├-- utils.py                 # JSONL I/O, hashing, Arabic normalization
|   ├-- prompts.py               # Stage 1
|   ├-- tts.py                   # Stage 2
|   ├-- quality.py               # Auto quality signals
|   ├-- review.py                # Stage 3 (Gradio)
|   ├-- export.py                # Stage 4
|   └-- pipeline.py              # CLI orchestrator
├-- tests/                       # Pytest tests for critical logic (44 tests)
├-- scripts/                     # Inspection helpers (run after a pipeline run)
|   ├-- quality_summary.py       # CER / flagged stats overview
|   ├-- quality_breakdown.py     # Per-voice + per-category breakdown
|   ├-- verify_dataset.py        # Loads the HF dataset and prints schema + stats
|   ├-- make_samples.py          # Picks ~24 representative clips -> samples/
|   └-- clean_review_notes.py    # Compacts reviewed.jsonl, clears notes
├-- data/                        # All artifacts (audio is gitignored; manifests are not)
|   ├-- prompts/prompts.jsonl
|   ├-- audio/*.wav
|   ├-- audio/audio.jsonl
|   ├-- reviewed/quality.jsonl
|   ├-- reviewed/reviewed.jsonl
|   └-- final/                   # The deliverable
├-- samples/                     # 24 representative clips demonstrating the output format
├-- requirements.txt
├-- pyproject.toml               # pytest config
├-- .env.example
└-- README.md
```

## Tests

```bash
pytest
```

Coverage focuses on the **critical decision logic**:

- Arabic detection (pure / code-switched / pure-English / empty)
- Text cleaning (emojis, zero-width, quotes)
- Hash stability and dedup behavior across whitespace + diacritics
- QA normalization (alef folding, ta-marbuta, alef-maqsura, digits, punctuation, diacritics)
- Export join semantics (reject / flag / unreviewed handling, edit-text propagation, override flags)
- JSONL round-trip with Arabic content (no `\u` escaping)

## Configuration knobs

Everything tunable lives in [config/pipeline.yaml](config/pipeline.yaml). The most useful knobs:

| Key | What it does |
|---|---|
| `prompts.total_target` | How many text prompts to generate |
| `prompts.categories` | Per-category proportions (must sum to 1.0) |
| `tts.concurrency` | Async in-flight cap for TTS calls |
| `tts.voices` | List of providers + voice IDs (round-robin) |
| `quality.whisper_model` | tiny / base / small / medium / large |
| `quality.max_cer` | CER threshold for flagging |
| `export.include_rejected` | Include human-rejected samples (off by default) |
| `export.include_flagged` | Include auto-flagged + accepted-as-is samples |
| `export.dataset_name` | Output directory name under `data/final/` |

## Limitations

- **100% synthetic.** Cannot replace real Egyptian speech for fine-tuning. Mix with real data.
- **Three voices only.** True acoustic diversity needs more speakers; this is a free-tier constraint, not a design choice.
- **No noise / reverberation augmentation.** Real audio has it; this dataset doesn't. Add at training time via `audiomentations` or similar.
- **Whisper-as-judge bias.** Round-trip ASR uses Whisper, which is itself imperfect on Egyptian. Borderline samples can be flagged unfairly. The human review step exists exactly to catch this.
- **Auto-signal misses dialect-quality failures.** As shown in the Observed-Quality section, 71% of the human reviewer's calls weren't surfaced by the auto-signal. CER catches *what was said wrong*, not *who said it wrong*. A dialect classifier would close this gap; not implemented here.
- **No conversational data.** Each clip is one short utterance. Real conversation has overlap, backchannels, repairs - out of scope.
- **No prosodic control.** edge-tts SSML control was not used because (a) the gain is marginal for short clips at this dataset size and (b) hand-engineered prosody arguably misrepresents how downstream consumers will encounter TTS audio in the wild.
- **gTTS is MSA, not Egyptian.** Documented; included for diversity, not authenticity.
- **Synthesis builds the full task list in memory.** `src/tts.py` schedules all N coroutines up-front (bounded by a Semaphore for concurrency, not for memory). At ~80 prompts this is fine; at the README's stated 10×-production volume (~800), memory is still trivial, but a true production pipeline would stream from the manifest rather than load it whole.
