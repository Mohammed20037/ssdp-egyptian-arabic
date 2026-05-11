"""Compute per-voice and per-category quality stats from this run.

Used to populate the README's 'Observed quality issues' section with
real numbers, not hand-waving.
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    audio = {r["id"]: r for r in (json.loads(l) for l in (root / "data/audio/audio.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()) if r.get("status") == "ok"}
    quality = [json.loads(l) for l in (root / "data/reviewed/quality.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    reviews = {}
    for line in (root / "data/reviewed/reviewed.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            reviews[r["id"]] = r  # latest wins

    # Join
    rows = []
    for q in quality:
        a = audio.get(q["id"])
        if not a:
            continue
        rows.append({**a, **q, "decision": reviews.get(q["id"], {}).get("decision")})

    # Stats by voice
    by_voice = defaultdict(list)
    for r in rows:
        by_voice[r["voice_id"]].append(r)

    print("=" * 70)
    print("PER-VOICE BREAKDOWN")
    print("=" * 70)
    print(f"{'voice':30s} {'n':>4s}  {'mean_cer':>9s} {'median_cer':>11s} {'p90_cer':>9s} {'flagged':>9s} {'rejected':>9s}")
    for voice, items in sorted(by_voice.items()):
        cers = [r["cer"] for r in items if r.get("cer") is not None]
        cers_sorted = sorted(cers)
        flagged = sum(1 for r in items if r["auto_status"] == "flagged")
        rejected = sum(1 for r in items if r["decision"] == "reject")
        print(f"{voice:30s} {len(items):>4d}  {statistics.mean(cers):>9.3f} {statistics.median(cers):>11.3f} {cers_sorted[int(len(cers)*0.9)]:>9.3f} {flagged:>4d} ({100*flagged/len(items):>3.0f}%) {rejected:>4d} ({100*rejected/len(items):>3.0f}%)")

    # Stats by category
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)

    print()
    print("=" * 70)
    print("PER-CATEGORY BREAKDOWN")
    print("=" * 70)
    print(f"{'category':25s} {'n':>4s}  {'mean_cer':>9s} {'flagged':>9s} {'rejected':>9s}")
    for cat, items in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
        cers = [r["cer"] for r in items if r.get("cer") is not None]
        flagged = sum(1 for r in items if r["auto_status"] == "flagged")
        rejected = sum(1 for r in items if r["decision"] == "reject")
        print(f"{cat:25s} {len(items):>4d}  {statistics.mean(cers):>9.3f} {flagged:>4d} ({100*flagged/len(items):>3.0f}%) {rejected:>4d} ({100*rejected/len(items):>3.0f}%)")

    # Code-switching specifically
    cs_rows = [r for r in rows if r["category"] == "code_switching"]
    cs_cers = [r["cer"] for r in cs_rows if r.get("cer") is not None]
    non_cs = [r for r in rows if r["category"] != "code_switching"]
    non_cs_cers = [r["cer"] for r in non_cs if r.get("cer") is not None]
    print()
    print("=" * 70)
    print("CODE-SWITCHING vs OTHER")
    print("=" * 70)
    print(f"  code_switching:  n={len(cs_rows):2d}  mean CER = {statistics.mean(cs_cers):.3f}")
    print(f"  other:           n={len(non_cs):2d}  mean CER = {statistics.mean(non_cs_cers):.3f}")
    print(f"  ratio:           code-switched CER is {statistics.mean(cs_cers) / statistics.mean(non_cs_cers):.2f}x non-code-switched")

    # Numbers/dates specifically
    num_rows = [r for r in rows if r["category"] == "numbers_dates"]
    num_cers = [r["cer"] for r in num_rows if r.get("cer") is not None]
    print()
    print(f"  numbers_dates:   n={len(num_rows):2d}  mean CER = {statistics.mean(num_cers):.3f}")

    # Reject rate by voice - does any voice get rejected more?
    print()
    print("=" * 70)
    print("REJECTION RATE BY VOICE (your reviewer decisions)")
    print("=" * 70)
    for voice, items in sorted(by_voice.items()):
        reviewed = [r for r in items if r["decision"] is not None]
        rejected = sum(1 for r in reviewed if r["decision"] == "reject")
        accepted = sum(1 for r in reviewed if r["decision"] == "accept")
        edited = sum(1 for r in reviewed if r["decision"] == "edit")
        n = len(reviewed)
        print(f"  {voice:30s}  n={n:>2d}  reject={rejected:>2d} ({100*rejected/max(n,1):>3.0f}%)  accept={accepted:>2d}  edit={edited:>2d}")


if __name__ == "__main__":
    main()
