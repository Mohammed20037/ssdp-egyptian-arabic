"""Stage 3: Gradio review UI.

Lets a human listen to each (text, audio) pair and accept, edit, or
reject it. Auto-flagged clips are shown first so reviewer time goes
where it's most needed. Every action writes to reviewed.jsonl
immediately so you can stop and resume.

The human always wins over the auto-flag. A high-CER clip can still
be fine if Whisper itself struggled with the dialect.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import gradio as gr

from .config import load_config
from .utils import append_jsonl, read_jsonl, setup_logging

log = setup_logging()


# Data loading
def _load_review_queue(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Build the review queue and load any existing decisions.

    Returns (samples, decisions_by_id) where samples is the order the
    reviewer should see them in (flagged first, then ok).
    """
    paths = cfg["paths"]
    audio = [r for r in read_jsonl(paths["audio_manifest"]) if r.get("status") == "ok"]
    quality = {r["id"]: r for r in read_jsonl(paths["quality_manifest"])}
    decisions = {r["id"]: r for r in read_jsonl(paths["reviewed_manifest"])}

    enriched = []
    for r in audio:
        q = quality.get(r["id"], {})
        enriched.append({
            **r,
            "flags": q.get("flags", []),
            "auto_status": q.get("auto_status", "unknown"),
            "cer": q.get("cer"),
            "wer": q.get("wer"),
            "transcript": q.get("transcript"),
            "chars_per_sec": q.get("chars_per_sec"),
        })

    # Flagged first, then ok, then errors. Within each group: original order.
    priority = {"flagged": 0, "error": 1, "ok": 2, "unknown": 3}
    enriched.sort(key=lambda r: priority.get(r["auto_status"], 4))
    return enriched, decisions


# Decision recording
def _record_decision(
    decisions: dict[str, dict[str, Any]],
    reviewed_path: Path,
    sample_id: str,
    decision: str,
    final_text: str | None,
    note: str | None,
) -> None:
    """Append a decision to the manifest.

    The manifest is append-only; the latest entry per id wins (handled
    by readers via dict overwrite). Simpler than mutating a single file
    from a UI thread, and gives us an audit trail of overrides.
    """
    record = {
        "id": sample_id,
        "decision": decision,
        "final_text": final_text,
        "note": note,
    }
    append_jsonl(reviewed_path, record)
    decisions[sample_id] = record


# UI rendering
def _format_info(sample: dict[str, Any], decision: dict[str, Any] | None) -> str:
    flags = ", ".join(sample["flags"]) if sample["flags"] else "none"
    cer = f"{sample['cer']:.2f}" if sample["cer"] is not None else "-"
    wer = f"{sample['wer']:.2f}" if sample["wer"] is not None else "-"
    cps = f"{sample['chars_per_sec']:.1f}" if sample.get("chars_per_sec") is not None else "-"
    # Use `is not None` rather than truthy so duration=0.0 (a valid but
    # broken clip) renders as "0.00s" instead of vanishing into "-".
    dur = f"{sample['duration_sec']:.2f}s" if sample.get("duration_sec") is not None else "-"
    transcript = sample.get("transcript") or ""
    lines = [
        f"**Voice:** `{sample['voice_provider']}/{sample['voice_id']}` ({sample['voice_gender']}, {sample['voice_dialect']})",
        f"**Category:** `{sample.get('category', '?')}`",
        f"**Duration:** {dur}  -  **CPS:** {cps}  -  **CER:** {cer}  -  **WER:** {wer}",
        f"**Auto-flags:** {flags}",
    ]
    if transcript:
        lines.append(f"**Whisper transcript:** {transcript}")
    if decision:
        lines.append(f"**Previous decision:** `{decision['decision']}`" +
                     (f" - note: {decision['note']}" if decision.get('note') else ""))
    return "\n\n".join(lines)


def build_app(config_path: str | None = None) -> gr.Blocks:
    cfg = load_config(config_path)
    paths = cfg["paths"]
    project_root = Path(cfg["_root"])
    reviewed_path = Path(paths["reviewed_manifest"])
    reviewed_path.parent.mkdir(parents=True, exist_ok=True)

    samples, decisions = _load_review_queue(cfg)
    if not samples:
        raise RuntimeError("No samples to review. Run synthesis + quality first.")

    # State helpers -----
    def _stats() -> str:
        c = Counter(d["decision"] for d in decisions.values())
        return (
            f"**Reviewed:** {len(decisions)} / {len(samples)}  -  "
            f"✅ accept: {c.get('accept', 0)}  -  "
            f"✏️ edit: {c.get('edit', 0)}  -  "
            f"❌ reject: {c.get('reject', 0)}"
        )

    def _render(idx: int):
        idx = max(0, min(idx, len(samples) - 1))
        s = samples[idx]
        d = decisions.get(s["id"])
        audio_path = str(project_root / s["audio_path"])
        return (
            idx,
            f"### Sample {idx + 1} / {len(samples)} - `{s['id']}`",
            s["text"],                                # the source prompt
            audio_path,                                # audio file
            _format_info(s, d),                        # info markdown
            d["final_text"] if d and d.get("final_text") else s["text"],  # editable text
            _stats(),
        )

    # Decision handlers -----
    def _decide(idx: int, decision: str, edited_text: str, note: str):
        s = samples[idx]
        final_text = edited_text.strip() if decision == "edit" else (s["text"] if decision == "accept" else None)
        _record_decision(decisions, reviewed_path, s["id"], decision, final_text, note.strip() or None)
        next_idx = min(idx + 1, len(samples) - 1)
        return _render(next_idx)

    def _go(idx: int, delta: int):
        return _render(idx + delta)

    def _jump_to_next_unreviewed(idx: int):
        for j in range(idx + 1, len(samples)):
            if samples[j]["id"] not in decisions:
                return _render(j)
        # Wrap around
        for j in range(0, idx):
            if samples[j]["id"] not in decisions:
                return _render(j)
        return _render(idx)  # all reviewed

    # UI -----
    with gr.Blocks(title="SSDP Review", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🎙️ Synthetic Speech Data - Review")
        gr.Markdown(
            "Listen, compare to the prompt, then **accept** / **edit** / **reject**. "
            "Flagged samples appear first."
        )

        with gr.Row():
            stats_md = gr.Markdown(_stats())

        with gr.Row():
            header = gr.Markdown()

        with gr.Row():
            with gr.Column(scale=1):
                prompt_box = gr.Textbox(label="Prompt text (source)", interactive=False, lines=3)
                audio_player = gr.Audio(label="Synthesized audio", interactive=False)
                info_box = gr.Markdown()
            with gr.Column(scale=1):
                edit_box = gr.Textbox(
                    label="Edit this if you 'Accept with edit' (e.g., correct a transcription mismatch)",
                    lines=3,
                )
                note_box = gr.Textbox(label="Reviewer note (optional)", lines=1)
                with gr.Row():
                    accept_btn = gr.Button("✅ Accept", variant="primary")
                    edit_btn = gr.Button("✏️ Accept with edit")
                    reject_btn = gr.Button("❌ Reject")
                with gr.Row():
                    prev_btn = gr.Button("<- Prev")
                    next_btn = gr.Button("Next ->")
                    skip_btn = gr.Button("Skip to next unreviewed")

        idx_state = gr.State(value=0)

        outputs = [idx_state, header, prompt_box, audio_player, info_box, edit_box, stats_md]

        # Wire actions
        accept_btn.click(lambda i, e, n: _decide(i, "accept", e, n),
                         inputs=[idx_state, edit_box, note_box], outputs=outputs)
        edit_btn.click(lambda i, e, n: _decide(i, "edit", e, n),
                       inputs=[idx_state, edit_box, note_box], outputs=outputs)
        reject_btn.click(lambda i, e, n: _decide(i, "reject", e, n),
                         inputs=[idx_state, edit_box, note_box], outputs=outputs)
        prev_btn.click(lambda i: _go(i, -1), inputs=[idx_state], outputs=outputs)
        next_btn.click(lambda i: _go(i, +1), inputs=[idx_state], outputs=outputs)
        skip_btn.click(_jump_to_next_unreviewed, inputs=[idx_state], outputs=outputs)

        app.load(lambda: _render(0), outputs=outputs)

    return app


def launch(config_path: str | None = None) -> None:
    cfg = load_config(config_path)
    rcfg = cfg["review"]
    app = build_app(config_path)
    app.launch(server_port=int(rcfg["port"]), share=bool(rcfg["share"]), inbrowser=True)


if __name__ == "__main__":
    launch()
