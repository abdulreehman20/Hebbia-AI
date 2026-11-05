"""Download SEC 10-K, 10-Q, and 8-K primary filing documents.

Run from any directory with:
    uv run data/download_sec_filings.py

Set SEC_USER_AGENT to identify your application and provide a real contact email.
The script downloads all filing history exposed by each company's SEC submissions
JSON, including historical submission files listed by the SEC.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


# ---------------------------------------------------------------------------
# Configuration: edit these values before running the script.
# ---------------------------------------------------------------------------
SEC_USER_AGENT = "FinanceDocumentCopilot abdul008844@gmail.com"  # Use a real email.

COMPANIES: dict[str, dict[str, str]] = {
    "AAPL": {"name": "Apple Inc.", "cik": "0000320193"},
    "MSFT": {"name": "Microsoft Corporation", "cik": "0000789019"},
    "GOOGL": {"name": "Alphabet Inc.", "cik": "0001652044"},
    "AMZN": {"name": "Amazon.com, Inc.", "cik": "0001018724"},
    "NVDA": {"name": "NVIDIA Corporation", "cik": "0001045810"},
    "META": {"name": "Meta Platforms, Inc.", "cik": "0001326801"},
    "WMT": {"name": "Walmart Inc.", "cik": "0000104169"},
    "ORCL": {"name": "Oracle Corporation", "cik": "0001341439"},
}

BASE_FORMS = {"10-K", "10-Q", "8-K"}
# Historical SEC 8-K form variants that may appear in older EDGAR records.
LEGACY_8K_FORMS = {"8-K12B", "8-K12G3", "8-K15D5"}
INCLUDE_AMENDMENTS = True  # Also include each selected form's /A version.

# None means no date restriction (all filing history returned by SEC).
# Dates apply to filing_date, not the fiscal/reporting period.
START_DATE: str | None = "2021-01-01"
END_DATE: str | None = "2026-12-31"

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "downloads"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
CLEAR_OUTPUT_DIR = False  # True deletes the entire downloads directory first.
SKIP_EXISTING_FILES = True

# SEC currently sets a maximum access rate of 10 requests/second. Stay well below it.
REQUEST_INTERVAL_SECONDS = 0.15
MAX_RETRIES = 5
REQUEST_TIMEOUT_SECONDS = 60

SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions"
SEC_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)
_last_request_at = 0.0


# ---------------------------------------------------------------------------
# HTTP helpers: throttle every request and retry temporary SEC/network errors.
# ---------------------------------------------------------------------------
def _throttle_request() -> None:
    global _last_request_at
    elapsed = time.monotonic() - _last_request_at
    wait_for = REQUEST_INTERVAL_SECONDS - elapsed
    if wait_for > 0:
        time.sleep(wait_for)
    _last_request_at = time.monotonic()


def get_bytes(url: str, accept: str = "*/*") -> bytes:
    # Do not request compressed responses here: urllib does not automatically
    # decompress gzip/deflate payloads. The SEC can return the default identity body.
    headers = {
        "User-Agent": SEC_USER_AGENT,
        "Accept": accept,
    }

    for attempt in range(MAX_RETRIES + 1):
        _throttle_request()
        req = Request(url, headers=headers)
        try:
            with urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                return response.read()
        except HTTPError as exc:
            retryable = exc.code in {408, 429, 500, 502, 503, 504}
            if not retryable or attempt >= MAX_RETRIES:
                raise
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            try:
                delay = float(retry_after) if retry_after else 2**attempt
            except ValueError:
                delay = 2**attempt
            logger.warning("HTTP %s for %s; retrying in %.1fs", exc.code, url, delay)
            time.sleep(min(delay, 60.0))
        except URLError:
            if attempt >= MAX_RETRIES:
                raise
            delay = min(2**attempt, 60)
            logger.warning("Network error for %s; retrying in %ss", url, delay)
            time.sleep(delay)

    raise RuntimeError(f"Could not retrieve URL after retries: {url}")


def get_json(url: str) -> dict[str, Any]:
    raw = get_bytes(url, accept="application/json")
    parsed = json.loads(raw.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a JSON object from {url}")
    return parsed


# ---------------------------------------------------------------------------
# SEC submissions parsing.
# ---------------------------------------------------------------------------
def filing_rows(submission_payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert SEC's column-oriented submissions data into row dictionaries."""
    filings_section = submission_payload.get("filings")
    if isinstance(filings_section, dict) and isinstance(filings_section.get("recent"), dict):
        recent = filings_section["recent"]
    else:
        # Historical submissions files contain the column arrays at the root.
        recent = submission_payload

    required_fields = ("form", "accessionNumber", "filingDate", "primaryDocument")
    optional_fields = (
        "reportDate",
        "acceptanceDateTime",
        "items",
        "isXBRL",
        "isInlineXBRL",
    )
    arrays = {field: recent.get(field, []) for field in required_fields + optional_fields}

    # The required columns define the number of filing rows. Optional columns
    # are allowed to be absent or shorter without dropping every filing row.
    required_lengths = [
        len(arrays[field])
        for field in required_fields
        if isinstance(arrays[field], list)
    ]
    if len(required_lengths) != len(required_fields) or not required_lengths:
        return []
    row_count = min(required_lengths)

    rows: list[dict[str, Any]] = []
    for index in range(row_count):
        row = {
            field: values[index]
            if isinstance(values, list) and index < len(values)
            else None
            for field, values in arrays.items()
        }
        rows.append(row)
    return rows


def allowed_forms() -> set[str]:
    forms = BASE_FORMS | LEGACY_8K_FORMS
    if not INCLUDE_AMENDMENTS:
        return forms
    return forms | {f"{form}/A" for form in forms}


def within_date_range(filing_date: str) -> bool:
    if START_DATE and filing_date < START_DATE:
        return False
    if END_DATE and filing_date > END_DATE:
        return False
    return True


def get_company_filings(ticker: str, company: dict[str, str]) -> list[dict[str, Any]]:
    cik = company["cik"].zfill(10)
    submissions_url = f"{SEC_SUBMISSIONS_URL}/CIK{cik}.json"
    submission = get_json(submissions_url)

    payloads = [submission]
    historical_files = submission.get("filings", {}).get("files", [])

    # The main submissions file contains the most recent filings. Older records
    # are referenced by separate JSON files under filings.files.
    for historical_file in historical_files:
        name = historical_file.get("name")
        if not name:
            continue
        historical_url = f"{SEC_SUBMISSIONS_URL}/{quote(name, safe="-_.")}" 
        payloads.append(get_json(historical_url))

    found: dict[str, dict[str, Any]] = {}
    forms = allowed_forms()

    for payload in payloads:
        for row in filing_rows(payload):
            form = row.get("form")
            accession = row.get("accessionNumber")
            filing_date = row.get("filingDate")
            primary_document = row.get("primaryDocument")

            if form not in forms or not accession or not filing_date or not primary_document:
                continue
            if not within_date_range(filing_date):
                continue

            # Accession numbers uniquely identify filings for a given registrant.
            found[accession] = {
                "ticker": ticker,
                "company_name": company["name"],
                "cik": cik,
                "form": form,
                "filing_date": filing_date,
                "report_date": row.get("reportDate") or None,
                "accession_number": accession,
                "primary_document": primary_document,
                "acceptance_datetime": row.get("acceptanceDateTime") or None,
                "items": row.get("items") or None,
                "is_xbrl": row.get("isXBRL"),
                "is_inline_xbrl": row.get("isInlineXBRL"),
            }

    return sorted(found.values(), key=lambda item: item["filing_date"], reverse=True)


def build_document_paths(filing: dict[str, Any]) -> tuple[str, Path]:
    accession_without_dashes = filing["accession_number"].replace("-", "")
    cik_number = str(int(filing["cik"]))
    primary_document = PurePosixPath(filing["primary_document"])
    suffix = Path(primary_document.name).suffix or ".html"
    form_directory = filing["form"].replace("/", "-")
    year = filing["filing_date"][:4]

    filename = (
        f"{filing['ticker']}_{form_directory}_{filing['filing_date']}_"
        f"{accession_without_dashes}{suffix.lower()}"
    )
    relative_path = Path(filing["ticker"]) / form_directory / year / filename

    encoded_document = quote(primary_document.as_posix(), safe="/._-")
    source_url = (
        f"{SEC_ARCHIVES_URL}/{cik_number}/"
        f"{accession_without_dashes}/{encoded_document}"
    )
    return source_url, relative_path


def write_manifest(
    entries: list[dict[str, Any]], company_errors: list[dict[str, str]], generated_at: str
) -> None:
    statuses = [entry.get("download_status") for entry in entries]
    manifest = {
        "schema_version": 1,
        "source": "SEC EDGAR",
        "generated_at_utc": generated_at,
        "updated_at_utc": datetime.now(UTC).isoformat(),
        "companies": [
            {"ticker": ticker, **company} for ticker, company in COMPANIES.items()
        ],
        "forms_requested": sorted(allowed_forms()),
        "include_amendments": INCLUDE_AMENDMENTS,
        "filing_date_filter": {"start": START_DATE, "end": END_DATE},
        "discovered_count": len(entries),
        "downloaded_this_run_count": statuses.count("downloaded"),
        "already_present_count": statuses.count("already_present"),
        "available_count": statuses.count("downloaded") + statuses.count("already_present"),
        "failed_count": statuses.count("failed"),
        "company_errors": company_errors,
        "filings": entries,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temporary_path = MANIFEST_PATH.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(MANIFEST_PATH)


def download_all_filings() -> dict[str, Any]:
    if not SEC_USER_AGENT or "example.com" in SEC_USER_AGENT:
        raise ValueError("Set SEC_USER_AGENT to your application name and real contact email.")

    if START_DATE and END_DATE and START_DATE > END_DATE:
        raise ValueError("START_DATE must be earlier than or equal to END_DATE.")

    if CLEAR_OUTPUT_DIR and OUTPUT_DIR.exists():
        logger.warning("Deleting existing output directory: %s", OUTPUT_DIR)
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now(UTC).isoformat()
    entries: list[dict[str, Any]] = []
    company_errors: list[dict[str, str]] = []

    for ticker, company in COMPANIES.items():
        logger.info("Discovering %s filings (%s)...", ticker, company["name"])
        try:
            filings = get_company_filings(ticker, company)
        except Exception as exc:
            logger.exception("Could not retrieve submissions for %s", ticker)
            company_errors.append({"ticker": ticker, "error": str(exc)})
            write_manifest(entries, company_errors, generated_at)
            continue

        logger.info("Found %s matching filing(s) for %s", len(filings), ticker)
        for filing in filings:
            source_url, relative_path = build_document_paths(filing)
            local_path = OUTPUT_DIR / relative_path
            local_path.parent.mkdir(parents=True, exist_ok=True)

            entry = {
                **filing,
                "source_url": source_url,
                "local_path": relative_path.as_posix(),
            }

            if SKIP_EXISTING_FILES and local_path.is_file() and local_path.stat().st_size > 0:
                entry["download_status"] = "already_present"
                entries.append(entry)
                continue

            try:
                content = get_bytes(source_url)
                if not content:
                    raise ValueError("SEC returned an empty document")
                temporary_file = local_path.with_name(local_path.name + ".part")
                temporary_file.write_bytes(content)
                temporary_file.replace(local_path)
                entry["download_status"] = "downloaded"
                entry["file_size_bytes"] = len(content)
                logger.info("Downloaded %s %s (%s)", ticker, filing["form"], filing["filing_date"])
            except Exception as exc:
                entry["download_status"] = "failed"
                entry["error"] = str(exc)
                logger.exception("Failed downloading %s", source_url)

            entries.append(entry)

        # Save after each company, so a long run still leaves a useful manifest.
        write_manifest(entries, company_errors, generated_at)

    write_manifest(entries, company_errors, generated_at)
    return {
        "discovered_count": len(entries),
        "downloaded_count": sum(e.get("download_status") == "downloaded" for e in entries),
        "already_present_count": sum(e.get("download_status") == "already_present" for e in entries),
        "failed_count": sum(e.get("download_status") == "failed" for e in entries),
        "manifest_path": MANIFEST_PATH,
    }


if __name__ == "__main__":
    summary = download_all_filings()
    logger.info("Download run finished: %s", summary)
