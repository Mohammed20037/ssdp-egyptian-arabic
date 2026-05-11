"""Pick a representative subset (default: 24 samples) from the final
dataset and copy them into samples/ as a demonstration of the output
format.

Selection: round-robin across (voice, category) buckets to balance
voices and categories, drawing only from accepted / edited rows.
"""
from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    metadata = root / "data" / "final" / "metadata.jsonl"
    samples_dir = root / "samples"
    samples_dir.mkdir(exist_ok=True)
    audio_out = samples_dir / "audio"
    audio_out.mkdir(exist_ok=True)

    rows = [json.loads(l) for l in metadata.read_text(encoding="utf-8").splitlines() if l.strip()]

    # Group by (voice, category) and round-robin so the sample set covers all combinations.
    by_bucket: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_bucket[(r["voice_id"], r["category"])].append(r)

    selected: list[dict] = []
    seen_ids: set[str] = set()
    target = 24

    # Round-robin: take one from each bucket until we hit the target.
    while len(selected) < target:
        added_this_round = 0
        for bucket, items in list(by_bucket.items()):
            if not items:
                continue
            r = items.pop(0)
            if r["id"] in seen_ids:
                continue
            selected.append(r)
            seen_ids.add(r["id"])
            added_this_round += 1
            if len(selected) >= target:
                break
        if added_this_round == 0:
            break

    # Copy WAVs and write a sample-only metadata file with rewritten paths.
    sample_meta = []
    for r in selected:
        src = root / r["audio_relpath"]
        dst = audio_out / src.name
        shutil.copy2(src, dst)
        new_row = dict(r)
        new_row["audio_relpath"] = f"audio/{src.name}"
        sample_meta.append(new_row)

    out = samples_dir / "metadata.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in sample_meta:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(sample_meta)} samples to {samples_dir}")
    print(f"  audio: {audio_out}")
    print(f"  metadata: {out}")
    print()
    print("Selected samples:")
    for r in sample_meta:
        print(f"  [{r['voice_id']:25s} | {r['category']:18s}] {r['text']}")


if __name__ == "__main__":
    main()
