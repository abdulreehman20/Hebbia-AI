"""Convert downloaded SEC filing primary documents to Markdown.

Reads downloads/manifest.json and writes one Markdown file per supported source
file plus markdown/manifest.json. Output is cleaned for chunking/RAG:
XBRL noise removed, empty layout tables dropped, Item lines promoted to
headings, and YAML frontmatter attached from filing metadata.

Install dependencies with:
    uv add beautifulsoup4 markdownify pypdf

Run with:
    uv run data/convert_sec_filings_to_markdown.py
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from bs4 import BeautifulSoup
from markdownify import markdownify

try:
    from pypdf import PdfReader
except ImportError:  # PDF support is optional; HTML/TXT conversion still works.
    PdfReader = None  # type: ignore[assignment,misc]


BASE_DIR = Path(__file__).resolve().parent
DOWNLOADS_DIR = BASE_DIR / "downloads"
MARKDOWN_DIR = BASE_DIR / "markdown"
DOWNLOAD_MANIFEST = DOWNLOADS_DIR / "manifest.json"
MARKDOWN_MANIFEST = MARKDOWN_DIR / "manifest.json"

SUPPORTED_EXTENSIONS = {".htm", ".html", ".txt", ".pdf", ".xml"}
OVERWRITE_MARKDOWN_FILES = True

# Hidden iXBRL / layout noise that markdownify otherwise dumps as text.
XBRL_CONTAINER_NAMES = {
    "ix:header",
    "ix:hidden",
    "ix:references",
    "ix:resources",
}
DISPLAY_NONE_RE = re.compile(r"display\s*:\s*none", re.IGNORECASE)
XBRL_LINE_RE = re.compile(
    r"^(?:false|true)?.*(?:us-gaap:|xbrli:|iso4217:|dei:|http://fasb\.org).*$",
    re.IGNORECASE | re.MULTILINE,
)
PAGE_FOOTER_RE = re.compile(
    r"^[A-Za-z0-9 .,&'\-]+\|\s*\d{4}\s+Form\s+10-[KQ]\s*\|\s*\d+\s*$",
    re.MULTILINE,
)
BROKEN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
TOC_ANCHOR_RE = re.compile(r"\[([^\]]+)\]\(#[^)]+\)")
SEPARATOR_ROW_RE = re.compile(r"^\|(?:\s*:?-{3,}:?\s*\|)+\s*$")
TABLE_ROW_RE = re.compile(r"^\|.*\|\s*$")
# 10-K/Q Item 1 / Item 1A / 8-K Item 5.02, etc.
ITEM_HEADING_RE = re.compile(
    r"(?m)^(Item\s+\d{1,2}[A-C]?(?:\.\d{2})?\.?\s*.+)$"
)
ITEM_LABEL_RE = re.compile(
    r"^Item\s+\d{1,2}[A-C]?(?:\.\d{2})?\.?\s*$",
    re.IGNORECASE,
)
HORIZONTAL_RULE_NOISE_RE = re.compile(r"(?:\n---\n){2,}")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def normalize_markdown(content: str) -> str:
    """Normalize line endings and excessive blank lines."""
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    content = re.sub(r"\n{3,}", "\n\n", content)
    return content.strip() + "\n"


def _cell_texts(row: str) -> list[str]:
    return [cell.strip() for cell in row.strip().strip("|").split("|")]


def _row_has_substance(row: str) -> bool:
    ignore = {"", "☐", "☒", "—", "-", "&#9744;", "&#9745;"}
    return any(cell not in ignore for cell in _cell_texts(row))


def _item_heading_from_table_rows(rows: list[str]) -> str | None:
    """Turn a 2-column Item/title table into a single ## heading when possible."""
    data_rows = [
        row
        for row in rows
        if not SEPARATOR_ROW_RE.match(row) and _row_has_substance(row)
    ]
    if len(data_rows) != 1:
        return None
    cells = [cell for cell in _cell_texts(data_rows[0]) if cell]
    if not cells:
        return None
    if len(cells) == 1 and ITEM_HEADING_RE.match(cells[0]):
        return f"## {cells[0]}"
    if len(cells) >= 2 and ITEM_LABEL_RE.match(cells[0]):
        return f"## {cells[0].rstrip('.')} {cells[1]}".strip()
    return None


def _filter_markdown_tables(content: str) -> str:
    """Drop empty layout tables; keep real ones. One table at a time (line scan)."""
    lines = content.split("\n")
    output: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if (
            TABLE_ROW_RE.match(line)
            and index + 1 < len(lines)
            and SEPARATOR_ROW_RE.match(lines[index + 1])
        ):
            start = index
            index += 2
            while index < len(lines) and TABLE_ROW_RE.match(lines[index]):
                # Next table starts when a row is followed by a separator.
                if (
                    index + 1 < len(lines)
                    and SEPARATOR_ROW_RE.match(lines[index + 1])
                    and index > start + 1
                ):
                    break
                index += 1
            block = lines[start:index]
            item_heading = _item_heading_from_table_rows(block)
            if item_heading:
                output.append(item_heading)
            elif any(
                _row_has_substance(row)
                for row in block
                if not SEPARATOR_ROW_RE.match(row)
            ):
                output.extend(block)
            # else: drop empty layout table
            continue
        output.append(line)
        index += 1
    return "\n".join(output)


def clean_markdown(content: str) -> str:
    """Strip conversion artifacts so files are ready for Item-based chunking."""
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    content = XBRL_LINE_RE.sub("", content)
    content = BROKEN_IMAGE_RE.sub("", content)
    content = TOC_ANCHOR_RE.sub(r"\1", content)
    content = PAGE_FOOTER_RE.sub("", content)
    content = _filter_markdown_tables(content)
    content = ITEM_HEADING_RE.sub(r"## \1", content)
    # Avoid double-promoting already-cleaned files on re-run.
    content = re.sub(r"(?m)^(?:##\s+){2,}(Item\s+)", r"## \1", content)
    content = HORIZONTAL_RULE_NOISE_RE.sub("\n\n---\n\n", content)
    content = re.sub(r"[ \t]+\n", "\n", content)
    content = re.sub(r"\n{3,}", "\n\n", content)
    return content.strip() + "\n"


def build_frontmatter(filing: dict[str, Any]) -> str:
    """YAML frontmatter for parsers/chunkers (metadata lives with the text)."""
    fields = {
        "ticker": filing.get("ticker"),
        "company_name": filing.get("company_name"),
        "cik": filing.get("cik"),
        "form": filing.get("form"),
        "filing_date": filing.get("filing_date"),
        "report_date": filing.get("report_date"),
        "accession_number": filing.get("accession_number"),
        "source_url": filing.get("source_url"),
    }
    lines = ["---"]
    for key, value in fields.items():
        if value is None or value == "":
            continue
        text = str(value).replace('"', '\\"')
        if any(ch in text for ch in (":", "#", "{", "}", "[", "]", ",", "&", "*")):
            lines.append(f'{key}: "{text}"')
        else:
            lines.append(f"{key}: {text}")
    lines.append("---")
    return "\n".join(lines) + "\n\n"


def _decompose_hidden_and_xbrl(soup: BeautifulSoup) -> None:
    for tag in soup(["script", "style", "noscript", "img"]):
        tag.decompose()

    # Drop structural iXBRL containers (they dump us-gaap noise into markdown).
    for tag in list(soup.find_all(True)):
        name = (tag.name or "").lower()
        if name in XBRL_CONTAINER_NAMES:
            tag.decompose()

    # Unwrap remaining ix:* value tags so human-readable text is kept.
    for tag in list(soup.find_all(True)):
        name = (tag.name or "").lower()
        if name.startswith("ix:"):
            tag.unwrap()

    for tag in list(soup.find_all(True)):
        # Some malformed SEC tags have attrs=None; tag.get then crashes.
        attrs = tag.attrs if isinstance(tag.attrs, dict) else {}
        style = attrs.get("style")
        if isinstance(style, str) and DISPLAY_NONE_RE.search(style):
            tag.decompose()
            continue

        classes = attrs.get("class") or []
        if isinstance(classes, list) and "hidden" in {str(c).lower() for c in classes}:
            tag.decompose()


def html_to_markdown(file_path: Path) -> str:
    html_content = file_path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html_content, "html.parser")
    _decompose_hidden_and_xbrl(soup)

    # Keep the body when present, but preserve the whole document if it is absent.
    content_node = soup.body if soup.body is not None else soup
    markdown = markdownify(
        str(content_node),
        heading_style="ATX",
        bullets="-",
        strip=["script", "style", "noscript", "img"],
    )
    return clean_markdown(normalize_markdown(markdown))


def text_to_markdown(file_path: Path) -> str:
    text = file_path.read_text(encoding="utf-8", errors="replace")
    return clean_markdown(normalize_markdown(text))


def pdf_to_markdown(file_path: Path) -> str:
    if PdfReader is None:
        raise RuntimeError("PDF conversion requires pypdf. Install it with: uv add pypdf")

    reader = PdfReader(str(file_path))
    pages: list[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = (page.extract_text() or "").strip()
        pages.append(
            f"<!-- source_page: {page_number} -->\n\n"
            f"## Page {page_number}\n\n{page_text}"
        )
    return clean_markdown(normalize_markdown("\n\n---\n\n".join(pages)))


def convert_file(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix in {".htm", ".html", ".xml"}:
        return html_to_markdown(file_path)
    if suffix == ".txt":
        return text_to_markdown(file_path)
    if suffix == ".pdf":
        return pdf_to_markdown(file_path)
    raise ValueError(f"Unsupported source file extension: {suffix}")


def resolve_source_path(relative_path: str) -> Path:
    """Resolve a manifest path while preventing paths outside downloads/."""
    normalized = relative_path.replace("\\", "/")
    source_path = (DOWNLOADS_DIR / Path(*PurePosixPath(normalized).parts)).resolve()
    downloads_root = DOWNLOADS_DIR.resolve()
    if not source_path.is_relative_to(downloads_root):
        raise ValueError(f"Manifest path escapes downloads directory: {relative_path}")
    return source_path


def write_markdown_manifest(entries: list[dict[str, Any]], source_manifest: dict[str, Any]) -> None:
    statuses = [entry.get("conversion_status") for entry in entries]
    output_manifest = {
        "schema_version": 1,
        "source": "SEC EDGAR",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "derived_from_manifest": "downloads/manifest.json",
        "chunking_hints": {
            "split_on": "## Item",
            "frontmatter": True,
            "notes": "Item headings are ATX ##; metadata is YAML frontmatter.",
        },
        "source_filing_count": len(entries),
        "converted_count": statuses.count("converted"),
        "skipped_count": sum(str(status).startswith("skipped") for status in statuses),
        "failed_count": statuses.count("failed"),
        "companies": source_manifest.get("companies", []),
        "forms_requested": source_manifest.get("forms_requested", []),
        "filings": entries,
    }
    MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)
    temporary_path = MARKDOWN_MANIFEST.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(output_manifest, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(MARKDOWN_MANIFEST)


def convert_all_filings() -> dict[str, Any]:
    if not DOWNLOAD_MANIFEST.is_file():
        raise FileNotFoundError(
            f"Download manifest not found: {DOWNLOAD_MANIFEST}. Run the download script first."
        )

    with DOWNLOAD_MANIFEST.open(encoding="utf-8") as file:
        source_manifest = json.load(file)

    source_filings = source_manifest.get("filings", [])
    if not isinstance(source_filings, list):
        raise ValueError("downloads/manifest.json must contain a 'filings' list")

    output_entries: list[dict[str, Any]] = []
    MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)

    for filing in source_filings:
        if not isinstance(filing, dict):
            continue

        entry = dict(filing)
        # Prefer the original HTM path if this entry was already converted once.
        original_local_path = entry.get("source_local_path") or entry.get("local_path")
        entry["source_local_path"] = original_local_path
        entry["markdown_local_path"] = None
        entry["local_path"] = None  # In this manifest, local_path points to Markdown.

        if entry.get("download_status") == "failed":
            entry["conversion_status"] = "skipped_download_failed"
            output_entries.append(entry)
            continue

        if not isinstance(original_local_path, str) or not original_local_path:
            entry["conversion_status"] = "skipped_missing_local_path"
            output_entries.append(entry)
            continue

        # Guard against feeding a .md path back into the HTML converter on re-run
        # if someone pointed this at markdown/manifest.json by mistake.
        if original_local_path.lower().endswith(".md"):
            entry["conversion_status"] = "skipped_markdown_source"
            entry["conversion_error"] = "source_local_path points to markdown, not a download"
            output_entries.append(entry)
            continue

        try:
            source_path = resolve_source_path(original_local_path)
            if not source_path.is_file():
                raise FileNotFoundError(f"Downloaded source file not found: {source_path}")

            if source_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                entry["conversion_status"] = "skipped_unsupported_format"
                entry["conversion_error"] = f"Unsupported extension: {source_path.suffix}"
                output_entries.append(entry)
                continue

            relative_source = Path(*PurePosixPath(original_local_path.replace("\\", "/")).parts)
            relative_markdown = relative_source.with_suffix(".md")
            markdown_path = MARKDOWN_DIR / relative_markdown
            markdown_path.parent.mkdir(parents=True, exist_ok=True)

            if markdown_path.exists() and not OVERWRITE_MARKDOWN_FILES:
                entry["conversion_status"] = "already_present"
            else:
                markdown_content = build_frontmatter(entry) + convert_file(source_path)
                temporary_path = markdown_path.with_name(markdown_path.name + ".part")
                temporary_path.write_text(markdown_content, encoding="utf-8")
                temporary_path.replace(markdown_path)
                entry["conversion_status"] = "converted"
                entry["markdown_size_bytes"] = markdown_path.stat().st_size

            entry["markdown_local_path"] = relative_markdown.as_posix()
            entry["local_path"] = relative_markdown.as_posix()
            entry["source_format"] = source_path.suffix.lower().lstrip(".")
            logger.info("Converted %s -> %s", original_local_path, relative_markdown.as_posix())
        except Exception as exc:
            entry["conversion_status"] = "failed"
            entry["conversion_error"] = str(exc)
            logger.exception("Failed converting %s", original_local_path)

        output_entries.append(entry)

    write_markdown_manifest(output_entries, source_manifest)
    summary = {
        "source_filing_count": len(output_entries),
        "converted_count": sum(e.get("conversion_status") == "converted" for e in output_entries),
        "skipped_count": sum(str(e.get("conversion_status", "")).startswith("skipped") for e in output_entries),
        "failed_count": sum(e.get("conversion_status") == "failed" for e in output_entries),
        "manifest_path": MARKDOWN_MANIFEST,
    }
    logger.info("Markdown conversion finished: %s", summary)
    return summary


if __name__ == "__main__":
    convert_all_filings()
