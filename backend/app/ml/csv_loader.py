"""
Production-Ready CSV Loader.

Centralised, resilient CSV ingestion for the Unified ML Platform.
Handles large files, encoding detection, delimiter sniffing, corrupted
rows, and memory-aware chunked reading — replacing every ad-hoc
``pd.read_csv()`` call in the project.

Design goals:
  1. Never crash the server on malformed input.
  2. Return actionable error messages the frontend can display.
  3. Log every significant decision for debugging.
  4. Stay under a configurable memory ceiling.
"""

from __future__ import annotations

import csv
import io
import os
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from ..logger import setup_logger

logger = setup_logger(__name__)

# ── Configurable limits ───────────────────────────────────────
MAX_FILE_SIZE_MB: int = 500  # Hard reject above this
CHUNK_THRESHOLD_MB: int = 50  # Use chunked reading above this
MAX_COLUMNS: int = 2000  # Reject absurdly wide files
SAMPLE_BYTES: int = 32_768  # Bytes to sniff for delimiter / encoding
CHUNK_ROWS: int = 50_000  # Rows per chunk when reading large files

ENCODINGS_TO_TRY = ("utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1")
DELIMITERS_TO_TRY = (",", ";", "\t", "|")


# ── Result container ──────────────────────────────────────────
@dataclass
class CSVLoadResult:
    """Returned by every public loader function."""

    df: Optional[pd.DataFrame] = None
    success: bool = False
    error: Optional[str] = None
    warnings: list[str] = field(default_factory=list)

    # Diagnostics
    file_size_mb: float = 0.0
    detected_encoding: Optional[str] = None
    detected_delimiter: Optional[str] = None
    row_count: int = 0
    col_count: int = 0
    skipped_rows: int = 0
    memory_usage_mb: float = 0.0


# ── Encoding detection ───────────────────────────────────────
def _detect_encoding(raw_bytes: bytes) -> str:
    """Try each candidate encoding against *raw_bytes*; return the first
    one that decodes without error.  Falls back to ``latin-1`` (which
    never raises)."""
    for enc in ENCODINGS_TO_TRY:
        try:
            raw_bytes.decode(enc)
            return enc
        except (UnicodeDecodeError, LookupError):
            continue
    return "latin-1"


# ── Delimiter detection ───────────────────────────────────────
def _detect_delimiter(sample_text: str) -> str:
    """Use Python's ``csv.Sniffer`` and fall back to frequency counting."""
    # 1. Try the stdlib Sniffer first
    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters="".join(DELIMITERS_TO_TRY))
        if dialect.delimiter in DELIMITERS_TO_TRY:
            return dialect.delimiter
    except csv.Error:
        pass

    # 2. Frequency heuristic — count first non-header line
    lines = sample_text.splitlines()
    data_line = lines[1] if len(lines) > 1 else lines[0] if lines else ""
    best, best_count = ",", 0
    for d in DELIMITERS_TO_TRY:
        c = data_line.count(d)
        if c > best_count:
            best, best_count = d, c
    return best


# ── Pre-flight validation ─────────────────────────────────────
def _validate_file(filepath: str) -> Optional[str]:
    """Return an error string if the file is unsuitable, else ``None``."""
    if not os.path.isfile(filepath):
        return f"File not found: {filepath}"

    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return (
            f"File is too large ({size_mb:.1f} MB). "
            f"Maximum allowed size is {MAX_FILE_SIZE_MB} MB."
        )

    if size_mb == 0:
        return "The uploaded file is empty (0 bytes)."

    return None


# ── Core loader ───────────────────────────────────────────────
def load_csv(filepath: str) -> CSVLoadResult:
    """
    Production-ready CSV loader.

    Steps:
      1. Validate file size & existence.
      2. Read a small sample to detect encoding + delimiter.
      3. Decide between single-pass vs. chunked reading.
      4. Read with ``on_bad_lines='warn'`` to skip corrupt rows.
      5. Enforce column-count limits.
      6. Log diagnostics.

    Returns a :class:`CSVLoadResult` that always has ``.success``
    and ``.error`` set — callers should never need to catch exceptions.
    """
    result = CSVLoadResult()

    # ── 1. Validate ──────────────────────────────────────────
    err = _validate_file(filepath)
    if err:
        result.error = err
        return result

    file_size = os.path.getsize(filepath)
    result.file_size_mb = round(file_size / (1024 * 1024), 2)
    logger.info(f"CSV load started — {filepath} ({result.file_size_mb} MB)")

    # ── 2. Sniff encoding + delimiter ────────────────────────
    try:
        with open(filepath, "rb") as f:
            raw_sample = f.read(SAMPLE_BYTES)
    except OSError as exc:
        result.error = f"Could not read file: {exc}"
        return result

    encoding = _detect_encoding(raw_sample)
    result.detected_encoding = encoding
    logger.info(f"Detected encoding: {encoding}")

    try:
        text_sample = raw_sample.decode(encoding, errors="replace")
    except Exception:
        text_sample = raw_sample.decode("latin-1", errors="replace")

    delimiter = _detect_delimiter(text_sample)
    result.detected_delimiter = delimiter
    logger.info(f"Detected delimiter: {repr(delimiter)}")

    # ── 3. Build common read_csv kwargs ──────────────────────
    read_kwargs = dict(
        filepath_or_buffer=filepath,
        encoding=encoding,
        sep=delimiter,
        on_bad_lines="warn",  # skip corrupt rows, log a warning
        low_memory=True,  # parse in chunks internally
        engine="c",  # fastest parser
        na_values=["", "NA", "N/A", "null", "NULL", "NaN", "nan", "None", "none", "-"],
    )

    # ── 4. Read — chunked if file is large ───────────────────
    use_chunks = result.file_size_mb > CHUNK_THRESHOLD_MB
    bad_row_count = 0

    try:
        if use_chunks:
            logger.info(
                f"File exceeds {CHUNK_THRESHOLD_MB} MB — using chunked reading "
                f"({CHUNK_ROWS} rows/chunk)"
            )
            chunks: list[pd.DataFrame] = []
            reader = pd.read_csv(**read_kwargs, chunksize=CHUNK_ROWS)
            for i, chunk in enumerate(reader):
                chunks.append(chunk)
                # Safety valve: estimate total memory
                mem_so_far = sum(c.memory_usage(deep=True).sum() for c in chunks)
                if mem_so_far > MAX_FILE_SIZE_MB * 1024 * 1024:
                    result.warnings.append(
                        f"Stopped reading at chunk {i + 1} — estimated memory "
                        f"({mem_so_far / (1024**2):.0f} MB) exceeds limit."
                    )
                    logger.warning(result.warnings[-1])
                    break

            if not chunks:
                result.error = "CSV file contains no readable data."
                return result

            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_csv(**read_kwargs)

    except pd.errors.EmptyDataError:
        result.error = "The CSV file is empty or has no parseable columns."
        return result
    except pd.errors.ParserError as exc:
        result.error = (
            f"CSV parsing failed: {exc}. "
            "The file may have inconsistent column counts or be corrupted."
        )
        return result
    except UnicodeDecodeError:
        # Very rare after our encoding detection, but handle gracefully
        result.error = (
            "Failed to decode the file even after automatic encoding detection. "
            "Please re-save the file as UTF-8."
        )
        return result
    except MemoryError:
        result.error = (
            f"The file ({result.file_size_mb} MB) is too large to fit in memory. "
            "Try reducing the number of rows or columns."
        )
        return result
    except Exception as exc:
        result.error = f"Unexpected error while reading CSV: {exc}"
        logger.exception("Unexpected CSV read error")
        return result

    # ── 5. Post-read validation ──────────────────────────────
    if df.empty:
        result.error = "The CSV file produced an empty DataFrame (0 rows after parsing)."
        return result

    if df.shape[1] > MAX_COLUMNS:
        result.error = (
            f"The dataset has {df.shape[1]} columns, exceeding the maximum "
            f"of {MAX_COLUMNS}. Please reduce the number of columns."
        )
        return result

    # Drop fully-empty columns that sometimes appear from trailing delimiters
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        df = df.drop(columns=empty_cols)
        result.warnings.append(
            f"Dropped {len(empty_cols)} fully-empty columns (likely trailing delimiters)."
        )

    # Drop fully-empty rows
    empty_before = len(df)
    df = df.dropna(how="all")
    dropped_empty = empty_before - len(df)
    if dropped_empty:
        result.warnings.append(f"Dropped {dropped_empty} completely empty rows.")

    # ── 6. Diagnostics ───────────────────────────────────────
    result.df = df
    result.success = True
    result.row_count = len(df)
    result.col_count = df.shape[1]
    result.skipped_rows = bad_row_count
    result.memory_usage_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)

    logger.info(
        f"CSV loaded — {result.row_count:,} rows × {result.col_count} cols, "
        f"{result.memory_usage_mb} MB in memory"
    )
    if result.warnings:
        for w in result.warnings:
            logger.warning(f"CSV warning: {w}")

    return result


# ── Buffer-based loader (for batch prediction) ────────────────
def load_csv_from_bytes(content: bytes, filename: str = "<upload>") -> CSVLoadResult:
    """
    Load a CSV from an in-memory byte buffer.

    Used by the batch-prediction endpoint which receives the file body
    directly without saving to disk first.
    """
    result = CSVLoadResult()
    result.file_size_mb = round(len(content) / (1024 * 1024), 2)

    if not content or len(content) == 0:
        result.error = "The uploaded file is empty."
        return result

    if result.file_size_mb > MAX_FILE_SIZE_MB:
        result.error = (
            f"File is too large ({result.file_size_mb:.1f} MB). "
            f"Maximum allowed size is {MAX_FILE_SIZE_MB} MB."
        )
        return result

    logger.info(f"CSV buffer load — {filename} ({result.file_size_mb} MB)")

    # Detect encoding + delimiter from the buffer
    sample = content[:SAMPLE_BYTES]
    encoding = _detect_encoding(sample)
    result.detected_encoding = encoding

    try:
        text_sample = sample.decode(encoding, errors="replace")
    except Exception:
        text_sample = sample.decode("latin-1", errors="replace")
    delimiter = _detect_delimiter(text_sample)
    result.detected_delimiter = delimiter

    try:
        df = pd.read_csv(
            io.BytesIO(content),
            encoding=encoding,
            sep=delimiter,
            on_bad_lines="warn",
            low_memory=True,
            engine="c",
            na_values=["", "NA", "N/A", "null", "NULL", "NaN", "nan", "None", "none", "-"],
        )
    except pd.errors.EmptyDataError:
        result.error = "The CSV file is empty or has no parseable columns."
        return result
    except pd.errors.ParserError as exc:
        result.error = f"CSV parsing failed: {exc}"
        return result
    except Exception as exc:
        result.error = f"Error reading CSV: {exc}"
        return result

    if df.empty:
        result.error = "The CSV file produced an empty DataFrame."
        return result

    # Drop ghost columns / rows
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        df = df.drop(columns=empty_cols)
    df = df.dropna(how="all")

    result.df = df
    result.success = True
    result.row_count = len(df)
    result.col_count = df.shape[1]
    result.memory_usage_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)

    logger.info(f"Buffer CSV loaded — {result.row_count:,} rows × {result.col_count} cols")
    return result
