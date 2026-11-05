"""Self-check for markdown cleanup helpers used by the converter.

Run:
    uv run data/check_markdown_cleanup.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_sec_filings_to_markdown import build_frontmatter, clean_markdown


def main() -> None:
    dirty = """\
false 0000320193 us-gaap:CommonStockMember 2020-12-29 xbrli:shares

UNITED STATES

|  |  |  |
| --- | --- | --- |
|  |  |  |
|  | | |

| California | 94-2404110 |
| --- | --- |
| Cupertino | 95014 |

![logo](g29637g56j44.jpg)

[Item 1A.](#anchor) Risk Factors toc

| Item 5.02 | Departure of Directors or Certain Officers. |
| --- | --- |
|  |  |

Item 7. Management Discussion.

Apple Inc. | 2024 Form 10-K | 2

Real sentence about the business.
"""
    cleaned = clean_markdown(dirty)
    assert "us-gaap" not in cleaned, cleaned
    assert "xbrli" not in cleaned, cleaned
    assert "![logo]" not in cleaned, cleaned
    assert "Form 10-K | 2" not in cleaned, cleaned
    assert "California" in cleaned and "95014" in cleaned, cleaned
    # Empty spacer table removed; real address table kept.
    assert cleaned.count("| --- |") == 1, cleaned
    assert "## Item 5.02 Departure of Directors or Certain Officers." in cleaned, cleaned
    assert "## Item 7. Management Discussion." in cleaned, cleaned
    assert "Item 1A. Risk Factors toc" in cleaned, cleaned
    assert "Real sentence about the business." in cleaned, cleaned

    front = build_frontmatter(
        {
            "ticker": "AAPL",
            "form": "8-K",
            "filing_date": "2021-01-05",
            "accession_number": "0001193125-21-001982",
            "company_name": "Apple Inc.",
        }
    )
    assert front.startswith("---\n")
    assert "ticker: AAPL" in front
    assert "form: 8-K" in front
    assert front.endswith("---\n\n")

    print("check_markdown_cleanup: ok")


if __name__ == "__main__":
    main()
