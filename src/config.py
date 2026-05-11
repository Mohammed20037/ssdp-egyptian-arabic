"""Config loader. Reads pipeline.yaml and exposes a typed-ish dict."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "config" / "pipeline.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    load_dotenv(ROOT / ".env")
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Resolve relative paths against project root.
    for key, value in cfg.get("paths", {}).items():
        cfg["paths"][key] = str(ROOT / value)
    cfg["_root"] = str(ROOT)

    # Validate category proportions sum to ~1.0
    cats = cfg["prompts"]["categories"]
    total = sum(cats.values())
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"Prompt category proportions sum to {total:.3f}, expected 1.0")

    return cfg


def get_openai_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key or key.startswith("sk-your-key"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return key
