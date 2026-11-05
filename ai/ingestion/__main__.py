"""CLI entrypoint for Phase 2 ingestion.

Run from backend root:
    uv run python -m ai.ingestion
    uv run python -m ai.ingestion --validate-only
"""

from __future__ import annotations

import argparse
import logging
import sys

from ai.ingestion.pipeline import run_ingestion
from ai.ingestion.validate import validate_ingestion
from config import get_settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 2 SEC Markdown ingestion")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Skip ingest and only run retrieval validation against an existing report/index.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    settings = get_settings()
    try:
        if args.validate_only:
            validation = validate_ingestion(settings=settings)
        else:
            report = run_ingestion(settings)
            validation = validate_ingestion(report=report, settings=settings)
    except Exception as exc:
        logging.exception("Ingestion failed: %s", exc)
        return 1

    ok = bool(validation.get("all_namespaces_present") and validation.get("retrieval_ok"))
    print(
        "Phase 2 validation:",
        "PASS" if ok else "FAIL",
        f"namespaces_missing={validation.get('missing_namespaces')}",
        f"retrieval_ok={validation.get('retrieval_ok')}",
    )
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
