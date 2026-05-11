"""Print a quick summary of the quality signals pass."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    q = [
        json.loads(line)
        for line in (root / "data" / "reviewed" / "quality.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cers = [r["cer"] for r in q if r.get("cer") is not None]
    cers.sort()
    flagged = [r for r in q if r["auto_status"] == "flagged"]
    flag_reasons = Counter(f for r in q for f in r["flags"])

    print(f"Total samples: {len(q)}")
    print(f"Mean CER:   {sum(cers)/len(cers):.3f}")
    print(f"Median CER: {cers[len(cers)//2]:.3f}")
    print(f"P90 CER:    {cers[int(len(cers)*0.9)]:.3f}")
    print(f"Max CER:    {max(cers):.3f}")
    print(f"Flagged:    {len(flagged)} / {len(q)} ({100*len(flagged)/len(q):.1f}%)")
    print(f"Flag reasons: {dict(flag_reasons)}")
    print()
    print("--- Flagged samples ---")
    for r in flagged:
        print(f"  {r['id'][:30]:30s}  cer={r['cer']}  flags={r['flags']}")
        print(f"    transcript: {r['transcript']}")


if __name__ == "__main__":
    main()
