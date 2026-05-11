"""End-to-end pipeline runner.

Orchestrates the four stages and exposes a CLI:

    python -m src.pipeline --stage all       # run everything except review UI
    python -m src.pipeline --stage prompts   # generate text prompts
    python -m src.pipeline --stage tts       # synthesize audio
    python -m src.pipeline --stage quality   # auto-quality signals
    python -m src.pipeline --stage review    # launch Gradio reviewer
    python -m src.pipeline --stage export    # build training-ready dataset

Each stage is independently resumable: rerunning a stage skips work
that's already been done by reading its manifest.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config
from .utils import setup_logging

log = setup_logging()


STAGES = ("prompts", "tts", "quality", "review", "export", "all")


def _run_prompts(config_path: str | None) -> None:
    from .prompts import generate_prompts
    out = generate_prompts(config_path)
    log.info("[OK] prompts -> %s", out)


def _run_tts(config_path: str | None) -> None:
    from .tts import synthesize
    out = synthesize(config_path)
    log.info("[OK] tts -> %s", out)


def _run_quality(config_path: str | None) -> None:
    from .quality import compute_quality
    out = compute_quality(config_path)
    log.info("[OK] quality -> %s", out)


def _run_review(config_path: str | None) -> None:
    from .review import launch
    log.info("Launching review UI ... (Ctrl+C to exit when done reviewing)")
    launch(config_path)


def _run_export(config_path: str | None) -> None:
    from .export import export_dataset
    out = export_dataset(config_path)
    log.info("[OK] export -> %s", out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic Speech Data Pipeline (S.S.D.P.)")
    parser.add_argument(
        "--stage",
        choices=STAGES,
        default="all",
        help="Which stage to run. 'all' runs prompts -> tts -> quality "
             "(review and export are interactive / final and must be invoked explicitly).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to pipeline.yaml (defaults to config/pipeline.yaml).",
    )
    args = parser.parse_args()

    # Validate config eagerly so we fail fast with a useful error.
    cfg = load_config(args.config)
    log.info("SSDP starting. Project root: %s", cfg["_root"])

    runners = {
        "prompts": _run_prompts,
        "tts": _run_tts,
        "quality": _run_quality,
        "review": _run_review,
        "export": _run_export,
    }

    if args.stage == "all":
        # Note: review is intentionally not in 'all'. It's interactive
        # and must be triggered explicitly so a CI run doesn't hang.
        for name in ("prompts", "tts", "quality"):
            log.info("--- stage: %s ---", name)
            runners[name](args.config)
        log.info(
            "Pipeline finished through `quality`. Now run:\n"
            "  python -m src.pipeline --stage review   (launches Gradio UI)\n"
            "  python -m src.pipeline --stage export   (after reviewing)"
        )
    else:
        runners[args.stage](args.config)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.warning("Interrupted by user.")
        sys.exit(130)
