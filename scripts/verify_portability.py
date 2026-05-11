"""Verify the exported HF dataset is portable.

We check:
  1. The `audio` column is a dataset-relative path, NOT absolute.
  2. The relative path actually resolves to a file inside the dataset.
  3. The audio file is readable by librosa (smoke test).
  4. The metadata.jsonl has no leaked absolute paths.
"""
from __future__ import annotations

import json
from pathlib import Path

import librosa
from datasets import load_from_disk


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    ds_path = root / "data" / "final" / "egyptian_arabic_synthetic_v1"
    ds = load_from_disk(str(ds_path))

    failed = 0

    # Check 1: audio column is relative
    sample = ds[0]
    audio_col = sample["audio"]
    print(f"Sample audio column: {audio_col!r}")
    if Path(audio_col).is_absolute():
        print(f"  FAIL: audio column is absolute, expected dataset-relative")
        failed += 1
    elif not audio_col.startswith("audio/"):
        print(f"  FAIL: audio column doesn't start with 'audio/'")
        failed += 1
    else:
        print(f"  OK: dataset-relative")

    # Check 2: resolves to a real file
    full_path = ds_path / audio_col
    if not full_path.exists():
        print(f"  FAIL: resolved path doesn't exist: {full_path}")
        failed += 1
    else:
        print(f"  OK: resolves to {full_path}")

    # Check 3: librosa can decode it
    try:
        y, sr = librosa.load(str(full_path), sr=None, mono=True)
        print(f"  OK: librosa decoded {len(y)} samples at {sr} Hz")
    except Exception as e:
        print(f"  FAIL: librosa decode error: {e}")
        failed += 1

    # Check 4: metadata.jsonl has no abs paths
    metadata_path = root / "data" / "final" / "metadata.jsonl"
    rows = [json.loads(l) for l in metadata_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    abs_count = 0
    for r in rows:
        for k, v in r.items():
            if isinstance(v, str) and (v.startswith("C:") or v.startswith("c:") or v.startswith("/Users/")):
                abs_count += 1
                if abs_count <= 3:
                    print(f"  FAIL: row {r['id']} field {k!r} has absolute path: {v!r}")
    if abs_count == 0:
        print(f"  OK: metadata.jsonl has no absolute paths")
    else:
        print(f"  FAIL: {abs_count} absolute paths in metadata.jsonl")
        failed += 1

    # Check 5: dataset_info.json declares audio as Value("string"), matching the README claim
    info = json.loads((ds_path / "dataset_info.json").read_text(encoding="utf-8"))
    audio_type = info.get("features", {}).get("audio", {})
    print(f"Schema check: audio feature is {audio_type}")
    if audio_type.get("dtype") != "string":
        print(f"  FAIL: expected dtype=string for `audio`, got {audio_type}")
        failed += 1
    else:
        print(f"  OK: schema matches README claim")

    # Check 6: load snippet from README actually works end-to-end
    try:
        ds2 = ds.map(lambda r: {"audio": str(ds_path / r["audio"])})
        # Don't actually cast to Audio() since that needs torchcodec. Just verify the path step.
        first = ds2[0]
        if not Path(first["audio"]).exists():
            print(f"  FAIL: README load snippet produces non-existent path: {first['audio']}")
            failed += 1
        else:
            print(f"  OK: README load snippet resolves correctly")
    except Exception as e:
        print(f"  FAIL: README snippet errored: {e}")
        failed += 1

    print()
    if failed:
        print(f"=== {failed} CHECK(S) FAILED ===")
        raise SystemExit(1)
    print("=== ALL CHECKS PASSED ===")


if __name__ == "__main__":
    main()
