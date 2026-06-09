"""
Production-Ready CSV Loader — Tiered Large Dataset Support.

Centralised, resilient CSV ingestion for the Unified ML Platform.
Handles files up to 10 GB through adaptive, tiered processing:

  Tier 1 (< 500 MB)  — Standard pandas load, full dataset.
  Tier 2 (500 MB – 2 GB) — Chunked ingestion + memory optimisation.
  Tier 3 (2 GB – 10 GB)  — Chunked + feature-aware + sampling.
  Tier 4 (> 10 GB)       — Rejected with clear message.

Design goals:
  1. Never crash the server on malformed input.
  2. Return actionable error messages the frontend can display.
  3. Log every significant decision for debugging.
  4. Stay under a configurable memory ceiling.
  5. Produce a single in-memory DataFrame that downstream
     preprocessing and training pipelines can consume unchanged.
"""

from __future__ import annotations

import csv
import io
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from ..logger import setup_logger

logger = setup_logger(__name__)

# ── Tier boundaries (MB) ─────────────────────────────────────
TIER1_MAX_MB: int = 500  # Standard pandas load
TIER2_MAX_MB: int = 2_048  # Chunked + memory optimisation
TIER3_MAX_MB: int = 10_240  # Chunked + sampling (10 GB)
# Above TIER3 → rejected

MAX_COLUMNS: int = 2_000  # Reject absurdly wide files
SAMPLE_BYTES: int = 32_768  # Bytes to sniff for delimiter / encoding
CHUNK_ROWS: int = 100_000  # Rows per chunk when reading large files

# Sampling defaults
DEFAULT_SAMPLE_ROWS: int = 1_000_000  # Max rows for Tier 3 training sample
CATEGORY_CARDINALITY_THRESHOLD: int = 50  # Convert to category dtype below this

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

    # Tier and dataset stats (new)
    tier: int = 0
    total_rows_in_file: int = 0  # Before sampling
    memory_before_optimization_mb: float = 0.0
    memory_after_optimization_mb: float = 0.0
    memory_savings_pct: float = 0.0
    sampling_applied: bool = False
    sampling_ratio: float = 1.0
    processing_time_seconds: float = 0.0
    columns_loaded: int = 0  # vs. total columns in file


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
    if size_mb > TIER3_MAX_MB:
        return (
            f"File is too large ({size_mb:,.1f} MB / {size_mb / 1024:.1f} GB). "
            f"Maximum allowed size is {TIER3_MAX_MB / 1024:.0f} GB. "
            "Please reduce the file size before uploading."
        )

    if size_mb == 0:
        return "The uploaded file is empty (0 bytes)."

    return None


def _determine_tier(file_size_mb: float) -> int:
    """Classify the file into a processing tier."""
    if file_size_mb <= TIER1_MAX_MB:
        return 1
    elif file_size_mb <= TIER2_MAX_MB:
        return 2
    elif file_size_mb <= TIER3_MAX_MB:
        return 3
    else:
        return 4  # Should be caught by _validate_file


# ── Memory optimisation ──────────────────────────────────────
def optimize_memory(df: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    """
    Downcast numeric dtypes and convert low-cardinality strings to
    ``category`` to reduce memory footprint.

    Returns:
        (optimised_df, memory_before_mb, memory_after_mb)
    """
    mem_before = df.memory_usage(deep=True).sum() / (1024 * 1024)

    for col in df.columns:
        col_dtype = df[col].dtype

        # Integer downcasting
        if pd.api.types.is_integer_dtype(col_dtype):
            df[col] = pd.to_numeric(df[col], downcast="integer")

        # Float downcasting
        elif pd.api.types.is_float_dtype(col_dtype):
            df[col] = pd.to_numeric(df[col], downcast="float")

        # Low-cardinality string → category
        elif col_dtype == "object":
            n_unique = df[col].nunique()
            if n_unique <= CATEGORY_CARDINALITY_THRESHOLD:
                df[col] = df[col].astype("category")

    mem_after = df.memory_usage(deep=True).sum() / (1024 * 1024)
    return df, round(mem_before, 2), round(mem_after, 2)


# ── Sampling ─────────────────────────────────────────────────
def _sample_dataframe(
    df: pd.DataFrame,
    target_column: Optional[str],
    problem_type: Optional[str],
    max_rows: int = DEFAULT_SAMPLE_ROWS,
) -> tuple[pd.DataFrame, float]:
    """
    Create a representative sample from a large DataFrame.

    - Classification → stratified sampling (preserves class distribution).
    - Regression     → random sampling.

    Returns:
        (sampled_df, sampling_ratio)
    """
    if len(df) <= max_rows:
        return df, 1.0

    ratio = max_rows / len(df)

    if target_column and problem_type in ("binary", "multiclass"):
        # Stratified sampling
        try:
            sampled = df.groupby(target_column, group_keys=False).apply(
                lambda grp: grp.sample(
                    n=max(1, int(len(grp) * ratio)),
                    random_state=42,
                ),
                include_groups=True,
            )
            # Trim to exact max_rows if groupby rounding overshot
            if len(sampled) > max_rows:
                sampled = sampled.sample(n=max_rows, random_state=42)
            logger.info(
                f"Stratified sampling: {len(df):,} → {len(sampled):,} rows "
                f"(ratio {ratio:.4f}), class distribution preserved"
            )
            return sampled.reset_index(drop=True), round(ratio, 6)
        except Exception as e:
            logger.warning(f"Stratified sampling failed ({e}), falling back to random.")

    # Random sampling (regression or fallback)
    sampled = df.sample(n=max_rows, random_state=42)
    logger.info(f"Random sampling: {len(df):,} → {len(sampled):,} rows (ratio {ratio:.4f})")
    return sampled.reset_index(drop=True), round(ratio, 6)


# ── Sniff helpers ─────────────────────────────────────────────
def _sniff_file(filepath: str) -> tuple[str, str]:
    """Read a small sample and return (encoding, delimiter)."""
    with open(filepath, "rb") as f:
        raw_sample = f.read(SAMPLE_BYTES)

    encoding = _detect_encoding(raw_sample)
    try:
        text_sample = raw_sample.decode(encoding, errors="replace")
    except Exception:
        text_sample = raw_sample.decode("latin-1", errors="replace")

    delimiter = _detect_delimiter(text_sample)
    return encoding, delimiter


def _base_read_kwargs(filepath: str, encoding: str, delimiter: str) -> dict:
    """Common kwargs shared by all pd.read_csv calls."""
    return dict(
        filepath_or_buffer=filepath,
        encoding=encoding,
        sep=delimiter,
        on_bad_lines="warn",
        low_memory=True,
        engine="c",
        na_values=["", "NA", "N/A", "null", "NULL", "NaN", "nan", "None", "none", "-"],
    )


def _count_total_columns(filepath: str, encoding: str, delimiter: str) -> int:
    """Read only the header row to determine total column count."""
    try:
        header = pd.read_csv(filepath, encoding=encoding, sep=delimiter, nrows=0)
        return len(header.columns)
    except Exception:
        return 0


# ── Post-read cleanup ────────────────────────────────────────
def _post_read_cleanup(df: pd.DataFrame, result: CSVLoadResult) -> pd.DataFrame:
    """Drop fully-empty columns/rows, enforce column limits."""
    # Drop fully-empty columns (trailing delimiters)
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

    # Enforce column limit
    if df.shape[1] > MAX_COLUMNS:
        result.error = (
            f"The dataset has {df.shape[1]} columns, exceeding the maximum "
            f"of {MAX_COLUMNS}. Please reduce the number of columns."
        )
        return df

    return df


# ══════════════════════════════════════════════════════════════
#  TIER 1: Standard Load (< 500 MB)
# ══════════════════════════════════════════════════════════════
def _load_tier1(filepath: str, kwargs: dict, result: CSVLoadResult) -> pd.DataFrame:
    """Simple single-pass pd.read_csv for small files."""
    logger.info("Tier 1 — standard single-pass read")
    return pd.read_csv(**kwargs)


# ══════════════════════════════════════════════════════════════
#  TIER 2: Chunked + Memory Optimised (500 MB – 2 GB)
# ══════════════════════════════════════════════════════════════
def _load_tier2(filepath: str, kwargs: dict, result: CSVLoadResult) -> pd.DataFrame:
    """Chunked reading with per-chunk memory optimisation."""
    logger.info(
        f"Tier 2 — chunked ingestion ({CHUNK_ROWS:,} rows/chunk) "
        "with memory optimisation"
    )
    chunks: list[pd.DataFrame] = []
    total_rows = 0

    reader = pd.read_csv(**kwargs, chunksize=CHUNK_ROWS)
    for i, chunk in enumerate(reader):
        # Optimise each chunk immediately to keep peak RAM low
        chunk, _, _ = optimize_memory(chunk)
        chunks.append(chunk)
        total_rows += len(chunk)

        if (i + 1) % 10 == 0:
            logger.info(f"  chunk {i + 1}: {total_rows:,} rows ingested so far")

    if not chunks:
        result.error = "CSV file contains no readable data."
        return pd.DataFrame()

    result.total_rows_in_file = total_rows
    df = pd.concat(chunks, ignore_index=True)
    logger.info(f"Tier 2 complete — {total_rows:,} rows assembled")
    return df


# ══════════════════════════════════════════════════════════════
#  TIER 3: Chunked + Feature-Aware + Sampling (2 GB – 10 GB)
# ══════════════════════════════════════════════════════════════
def _load_tier3(
    filepath: str,
    kwargs: dict,
    result: CSVLoadResult,
    usecols: Optional[list[str]] = None,
    target_column: Optional[str] = None,
    problem_type: Optional[str] = None,
    max_sample_rows: int = DEFAULT_SAMPLE_ROWS,
) -> pd.DataFrame:
    """
    Chunked reading that:
      1. Loads only ``usecols`` columns (if specified) via pd.read_csv(usecols=…).
      2. Applies per-chunk memory optimisation.
      3. Reservoirs-samples down to ``max_sample_rows`` across all chunks.

    Because reservoir sampling across chunks is complex, we instead:
      - Read all chunks (column-filtered), concatenate.
      - Then sample the assembled DataFrame.
    This is safe because column-filtering alone cuts memory by a huge factor
    (e.g. 50 columns → 5 columns = ~10× reduction before sampling).
    """
    if usecols:
        kwargs["usecols"] = usecols
        logger.info(
            f"Tier 3 — feature-aware loading: {len(usecols)} of "
            f"{_count_total_columns(filepath, kwargs.get('encoding', 'utf-8'), kwargs.get('sep', ','))} columns"
        )
    else:
        logger.info("Tier 3 — chunked ingestion (no column filter)")

    chunks: list[pd.DataFrame] = []
    total_rows = 0

    reader = pd.read_csv(**kwargs, chunksize=CHUNK_ROWS)
    for i, chunk in enumerate(reader):
        chunk, _, _ = optimize_memory(chunk)
        chunks.append(chunk)
        total_rows += len(chunk)

        if (i + 1) % 20 == 0:
            logger.info(f"  chunk {i + 1}: {total_rows:,} rows ingested so far")

    if not chunks:
        result.error = "CSV file contains no readable data."
        return pd.DataFrame()

    result.total_rows_in_file = total_rows
    df = pd.concat(chunks, ignore_index=True)
    del chunks  # Free memory immediately

    logger.info(f"Tier 3 assembled {total_rows:,} rows, starting sampling phase")

    # Sampling
    df, ratio = _sample_dataframe(df, target_column, problem_type, max_sample_rows)
    result.sampling_applied = ratio < 1.0
    result.sampling_ratio = ratio

    if result.sampling_applied:
        result.warnings.append(
            f"Dataset sampled from {total_rows:,} to {len(df):,} rows "
            f"(ratio {ratio:.4f}) to fit in memory for training. "
            f"{'Stratified' if problem_type in ('binary', 'multiclass') else 'Random'} "
            "sampling was used."
        )

    logger.info(f"Tier 3 complete — {len(df):,} rows ready for training")
    return df


# ══════════════════════════════════════════════════════════════
#  PUBLIC API — Primary Loader
# ══════════════════════════════════════════════════════════════
def load_csv(
    filepath: str,
    usecols: Optional[list[str]] = None,
    target_column: Optional[str] = None,
    problem_type: Optional[str] = None,
    max_sample_rows: int = DEFAULT_SAMPLE_ROWS,
) -> CSVLoadResult:
    """
    Production-ready, tiered CSV loader.

    Args:
        filepath: Path to the CSV file on disk.
        usecols: Optional list of column names to load (feature-aware loading).
                 If provided, only these columns are read from the CSV,
                 dramatically reducing memory for wide datasets.
        target_column: The name of the target column (used for stratified
                       sampling in Tier 3).
        problem_type: "binary", "multiclass", or "regression" (used for
                      sampling strategy in Tier 3).
        max_sample_rows: Maximum rows to retain for Tier 3 sampling.

    Returns:
        CSVLoadResult with ``.success``, ``.df``, diagnostics, and
        ``dataset_stats`` fields populated.
    """
    result = CSVLoadResult()
    start_time = time.time()

    # ── 1. Validate ──────────────────────────────────────────
    err = _validate_file(filepath)
    if err:
        result.error = err
        return result

    file_size = os.path.getsize(filepath)
    result.file_size_mb = round(file_size / (1024 * 1024), 2)
    tier = _determine_tier(result.file_size_mb)
    result.tier = tier
    logger.info(
        f"CSV load started — {filepath} ({result.file_size_mb:,.1f} MB, Tier {tier})"
    )

    # ── 2. Sniff encoding + delimiter ────────────────────────
    try:
        encoding, delimiter = _sniff_file(filepath)
    except OSError as exc:
        result.error = f"Could not read file: {exc}"
        return result

    result.detected_encoding = encoding
    result.detected_delimiter = delimiter
    logger.info(f"Detected encoding={encoding}, delimiter={repr(delimiter)}")

    # ── 3. Build read_csv kwargs ─────────────────────────────
    kwargs = _base_read_kwargs(filepath, encoding, delimiter)

    # Record total columns in file before any column filtering
    result.columns_loaded = _count_total_columns(filepath, encoding, delimiter)

    # ── 4. Tiered loading ────────────────────────────────────
    try:
        if tier == 1:
            # Small file — optionally apply usecols even here for consistency
            if usecols:
                kwargs["usecols"] = usecols
            df = _load_tier1(filepath, kwargs, result)
            result.total_rows_in_file = len(df)

        elif tier == 2:
            if usecols:
                kwargs["usecols"] = usecols
            df = _load_tier2(filepath, kwargs, result)

        else:  # tier == 3
            df = _load_tier3(
                filepath,
                kwargs,
                result,
                usecols=usecols,
                target_column=target_column,
                problem_type=problem_type,
                max_sample_rows=max_sample_rows,
            )

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
        result.error = (
            "Failed to decode the file even after automatic encoding detection. "
            "Please re-save the file as UTF-8."
        )
        return result
    except MemoryError:
        result.error = (
            f"The file ({result.file_size_mb:,.0f} MB) exhausted available RAM. "
            "Try selecting fewer features or reducing the file size."
        )
        return result
    except Exception as exc:
        result.error = f"Unexpected error while reading CSV: {exc}"
        logger.exception("Unexpected CSV read error")
        return result

    # ── 5. Post-read cleanup ─────────────────────────────────
    if df.empty:
        result.error = "The CSV file produced an empty DataFrame (0 rows after parsing)."
        return result

    df = _post_read_cleanup(df, result)
    if result.error:
        return result

    # ── 6. Memory optimisation (Tier 2+ always, Tier 1 skip) ─
    if tier >= 2 and not result.sampling_applied:
        # Tier 3 already optimised per-chunk, but run again on the
        # final concatenated frame for maximum savings
        df, mem_before, mem_after = optimize_memory(df)
        result.memory_before_optimization_mb = mem_before
        result.memory_after_optimization_mb = mem_after
    elif tier == 1:
        # Still record memory, but skip heavy optimisation for speed
        mem_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
        result.memory_before_optimization_mb = mem_mb
        result.memory_after_optimization_mb = mem_mb
    else:
        # Tier 3 sampled — memory was optimised during chunk read
        mem_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
        result.memory_before_optimization_mb = mem_mb
        result.memory_after_optimization_mb = mem_mb

    if result.memory_before_optimization_mb > 0:
        result.memory_savings_pct = round(
            (1 - result.memory_after_optimization_mb / result.memory_before_optimization_mb) * 100,
            1,
        )

    # ── 7. Finalise result ───────────────────────────────────
    result.df = df
    result.success = True
    result.row_count = len(df)
    result.col_count = df.shape[1]
    result.memory_usage_mb = round(
        df.memory_usage(deep=True).sum() / (1024 * 1024), 2
    )
    result.processing_time_seconds = round(time.time() - start_time, 2)
    if usecols:
        result.columns_loaded = len(usecols)

    if result.total_rows_in_file == 0:
        result.total_rows_in_file = result.row_count

    logger.info(
        f"CSV loaded (Tier {tier}) — "
        f"{result.row_count:,} rows × {result.col_count} cols, "
        f"{result.memory_usage_mb} MB in memory, "
        f"took {result.processing_time_seconds}s"
    )
    if result.memory_savings_pct > 0:
        logger.info(
            f"Memory optimisation: {result.memory_before_optimization_mb} MB → "
            f"{result.memory_after_optimization_mb} MB "
            f"({result.memory_savings_pct}% reduction)"
        )
    if result.warnings:
        for w in result.warnings:
            logger.warning(f"CSV warning: {w}")

    return result


def get_dataset_stats(result: CSVLoadResult) -> dict:
    """
    Build a ``dataset_stats`` dict suitable for inclusion in API responses.
    """
    return {
        "tier": result.tier,
        "file_size_mb": result.file_size_mb,
        "total_rows_in_file": result.total_rows_in_file,
        "rows_loaded": result.row_count,
        "columns_loaded": result.columns_loaded,
        "memory_before_mb": result.memory_before_optimization_mb,
        "memory_after_mb": result.memory_after_optimization_mb,
        "memory_savings_pct": result.memory_savings_pct,
        "sampling_applied": result.sampling_applied,
        "sampling_ratio": result.sampling_ratio,
        "processing_time_seconds": result.processing_time_seconds,
        "encoding": result.detected_encoding,
        "delimiter": repr(result.detected_delimiter),
    }


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

    if result.file_size_mb > TIER3_MAX_MB:
        result.error = (
            f"File is too large ({result.file_size_mb:.1f} MB). "
            f"Maximum allowed size is {TIER3_MAX_MB / 1024:.0f} GB."
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
