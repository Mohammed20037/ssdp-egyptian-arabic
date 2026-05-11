"""Compact reviewed.jsonl to one record per sample id, with notes set
to None.

Why this exists
-----
The append-only manifest pattern means reviewed.jsonl can have multiple
entries per id (re-reviews) and accumulates whatever the reviewer typed
in the note box during a session. In this run, the reviewer used the
note box to write the same generic "voice should sound more Egyptian"
comment for every clip - useful as session feedback, but noise inside
the per-record dataset metadata.

This script:
  - keeps only the LATEST entry per id (matching how export reads it)
  - sets review_note to None
  - rewrites the file as a clean snapshot

The decisions themselves (accept/edit/reject + final_text) are
preserved unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    p = root / "data" / "reviewed" / "reviewed.jsonl"

    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"Read {len(rows)} entries from {p}")

    # Latest entry per id wins (this matches export.py's join behavior)
    by_id: dict[str, dict] = {}
    for r in rows:
        by_id[r["id"]] = r

    # Wipe notes
    n_changed = 0
    for r in by_id.values():
        if r.get("note"):
            n_changed += 1
        r["note"] = None

    out = list(by_id.values())
    print(f"Compacted to {len(out)} unique samples, cleared notes from {n_changed} rows")

    p.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote clean snapshot to {p}")


if __name__ == "__main__":
    main()
