"""Generate README charts from the actual run data.

Three charts:
  - assets/cer_by_category.png       CER per category (the key finding)
  - assets/voice_breakdown.png       CER + reject rate per voice
  - assets/cer_distribution.png      Overall CER histogram
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)


def load_rows():
    audio = {r["id"]: r for r in (json.loads(l) for l in (ROOT / "data/audio/audio.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()) if r.get("status") == "ok"}
    quality = [json.loads(l) for l in (ROOT / "data/reviewed/quality.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    reviews = {}
    for line in (ROOT / "data/reviewed/reviewed.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            reviews[r["id"]] = r
    rows = []
    for q in quality:
        a = audio.get(q["id"])
        if a:
            rows.append({**a, **q, "decision": reviews.get(q["id"], {}).get("decision")})
    return rows


def chart_cer_by_category(rows):
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r["cer"])

    cats = sorted(by_cat.keys(), key=lambda c: statistics.mean(by_cat[c]))
    means = [statistics.mean(by_cat[c]) for c in cats]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(cats, means, color="#4a7eb3", edgecolor="#2c4e6c")

    # Highlight code-switching in red since it's the key finding
    for bar, cat in zip(bars, cats):
        if cat == "code_switching":
            bar.set_color("#c44e52")
            bar.set_edgecolor("#7a2a2e")

    ax.set_xlabel("Mean round-trip CER (lower is better)")
    ax.set_title("CER by prompt category (n=80)")
    ax.axvline(0.40, color="gray", linestyle="--", linewidth=0.8, label="auto-flag threshold (0.40)")
    ax.legend(loc="lower right", fontsize=9)

    for bar, v in zip(bars, means):
        ax.text(v + 0.005, bar.get_y() + bar.get_height() / 2, f"{v:.3f}",
                va="center", fontsize=9)

    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(ASSETS / "cer_by_category.png", dpi=140)
    plt.close()
    print("wrote cer_by_category.png")


def chart_voice_breakdown(rows):
    by_voice = defaultdict(list)
    for r in rows:
        by_voice[r["voice_id"]].append(r)

    voices = sorted(by_voice.keys())
    short_names = {
        "ar-EG-SalmaNeural": "edge / Salma (F, EG)",
        "ar-EG-ShakirNeural": "edge / Shakir (M, EG)",
        "ar": "gTTS (MSA-leaning)",
    }
    labels = [short_names.get(v, v) for v in voices]

    cers = [statistics.mean(r["cer"] for r in by_voice[v]) for v in voices]
    rejects = [
        100 * sum(1 for r in by_voice[v] if r["decision"] == "reject") / len(by_voice[v])
        for v in voices
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.bar(labels, cers, color="#4a7eb3", edgecolor="#2c4e6c")
    ax1.set_ylabel("Mean CER")
    ax1.set_title("Round-trip CER")
    for i, v in enumerate(cers):
        ax1.text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=9)
    ax1.set_ylim(0, max(cers) * 1.2)
    ax1.tick_params(axis="x", rotation=15, labelsize=9)
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_axisbelow(True)

    ax2.bar(labels, rejects, color="#c4884e", edgecolor="#7a542a")
    ax2.set_ylabel("Reject rate (%)")
    ax2.set_title("Human reject rate")
    for i, v in enumerate(rejects):
        ax2.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=9)
    ax2.set_ylim(0, max(rejects) * 1.2)
    ax2.tick_params(axis="x", rotation=15, labelsize=9)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_axisbelow(True)

    fig.suptitle("Per-voice quality: ASR agrees but humans don't", fontsize=11)
    plt.tight_layout()
    plt.savefig(ASSETS / "voice_breakdown.png", dpi=140)
    plt.close()
    print("wrote voice_breakdown.png")


def chart_cer_distribution(rows):
    cers = [r["cer"] for r in rows if r.get("cer") is not None]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(cers, bins=18, color="#4a7eb3", edgecolor="#2c4e6c")
    ax.axvline(statistics.median(cers), color="green", linestyle="-",
               linewidth=1.5, label=f"median = {statistics.median(cers):.3f}")
    ax.axvline(0.40, color="red", linestyle="--",
               linewidth=1.2, label="flag threshold = 0.40")
    ax.set_xlabel("Round-trip CER")
    ax.set_ylabel("Clip count")
    ax.set_title(f"CER distribution across all 80 synthesized clips")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(ASSETS / "cer_distribution.png", dpi=140)
    plt.close()
    print("wrote cer_distribution.png")


def main():
    rows = load_rows()
    print(f"loaded {len(rows)} rows")
    chart_cer_by_category(rows)
    chart_voice_breakdown(rows)
    chart_cer_distribution(rows)


if __name__ == "__main__":
    main()
