"""Verify the final exported dataset loads cleanly and print stats."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from datasets import load_from_disk


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    ds_path = root / "data" / "final" / "egyptian_arabic_synthetic_v1"
    ds = load_from_disk(str(ds_path))

    print(f"Loaded dataset from: {ds_path}")
    print(f"Rows: {len(ds)}")
    print(f"Schema:")
    for name, feat in ds.features.items():
        print(f"  - {name}: {feat}")
    print()
    print("Voice distribution:")
    for v, n in Counter(ds["voice_id"]).most_common():
        print(f"  {v:30s}  {n}")
    print()
    print("Category distribution:")
    for c, n in Counter(ds["category"]).most_common():
        print(f"  {c:20s}  {n}")
    print()
    print("Decision distribution:")
    for d, n in Counter(ds["review_decision"]).most_common():
        print(f"  {d:10s}  {n}")
    print()
    durations = [d for d in ds["duration_sec"] if d is not None]
    print(f"Total duration: {sum(durations):.1f} s ({sum(durations)/60:.1f} min)")
    print(f"Mean duration: {sum(durations)/len(durations):.2f} s")
    print()
    cers = [c for c in ds["asr_cer"] if c is not None]
    print(f"Round-trip CER - mean: {sum(cers)/len(cers):.3f}, max: {max(cers):.3f}")
    print()
    print("Sample row #0:")
    r = ds[0]
    for k, v in r.items():
        s = str(v)
        if len(s) > 80:
            s = s[:77] + "..."
        print(f"  {k}: {s}")


if __name__ == "__main__":
    main()
