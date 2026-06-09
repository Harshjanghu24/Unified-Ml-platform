import os
import tempfile

import pandas as pd
import pytest

from app.ml.csv_loader import (
    _detect_delimiter,
    _detect_encoding,
    load_csv,
    load_csv_from_bytes,
)

# ── Helpers for creating temp files ─────────────────────────────


@pytest.fixture
def temp_csv():
    """Create a temporary file and return its path. Cleans up after."""
    fd, path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)


def write_to_path(path: str, content: str, encoding="utf-8"):
    with open(path, "w", encoding=encoding) as f:
        f.write(content)


def write_bytes_to_path(path: str, content: bytes):
    with open(path, "wb") as f:
        f.write(content)


# ── 1. Encoding & Delimiter Detection ──────────────────────────


def test_detect_encoding():
    # UTF-8
    assert _detect_encoding(b"Hello,World\n1,2") in ("utf-8", "utf-8-sig")

    # UTF-8-sig (BOM)
    bom_content = b"\xef\xbb\xbfHello,World\n"
    assert _detect_encoding(bom_content) in ("utf-8", "utf-8-sig")

    # latin-1 (cannot be decoded as utf-8)
    latin_content = b"Caf\xe9,Preis\nEspresso,2"
    assert _detect_encoding(latin_content) in ("latin-1", "iso-8859-1", "cp1252")


def test_detect_delimiter():
    assert _detect_delimiter("a,b,c\n1,2,3") == ","
    assert _detect_delimiter("a;b;c\n1;2;3") == ";"
    assert _detect_delimiter("a\tb\tc\n1\t2\t3") == "\t"
    assert _detect_delimiter("a|b|c\n1|2|3") == "|"


# ── 2. Empty & Invalid Files ──────────────────────────────────


def test_load_empty_file(temp_csv):
    write_to_path(temp_csv, "")
    result = load_csv(temp_csv)
    assert not result.success
    assert "empty" in result.error.lower()

    result_bytes = load_csv_from_bytes(b"")
    assert not result_bytes.success
    assert "empty" in result_bytes.error.lower()


def test_load_file_no_data(temp_csv):
    # Only headers, no data
    write_to_path(temp_csv, "col1,col2,col3\n")
    result = load_csv(temp_csv)
    # pandas read_csv handles headers-only by returning an empty dataframe
    # our loader requires at least one row, so it should be an error
    assert not result.success
    assert "empty" in result.error.lower() or "0 rows" in result.error.lower()


def test_load_file_too_large(temp_csv, monkeypatch):
    import app.ml.csv_loader

    # Monkeypatch the max size limit to -1 for testing so even 0.0 MB files trigger it
    monkeypatch.setattr(app.ml.csv_loader, "MAX_FILE_SIZE_MB", -1)

    write_to_path(temp_csv, "a,b\n1,2\n")
    result = load_csv(temp_csv)
    assert not result.success
    assert "too large" in result.error.lower()

    result_bytes = load_csv_from_bytes(b"a,b\n1,2\n")
    assert not result_bytes.success
    assert "too large" in result_bytes.error.lower()


# ── 3. Malformed Rows & Corrupted Files ───────────────────────


def test_load_malformed_rows(temp_csv):
    # The first row has 3 columns. The second row has 4. The third row has 3.
    # on_bad_lines='warn' should drop the bad row and read the rest.
    content = "a,b,c\n1,2,3,4\n5,6,7\n"
    write_to_path(temp_csv, content)

    result = load_csv(temp_csv)
    assert result.success
    assert result.row_count > 0


def test_load_inconsistent_delimiters(temp_csv):
    # Header is comma, but body is semicolon
    content = "a,b,c\n1;2;3\n4;5;6\n"
    write_to_path(temp_csv, content)

    # The sniffer might detect comma from the header,
    # so the body will be read as a single column '1;2;3'.
    # Our robust loader shouldn't crash, it just parses it weirdly, which is expected.
    result = load_csv(temp_csv)
    assert result.success


# ── 4. Mixed Types & Missing Values ───────────────────────────


def test_load_missing_values(temp_csv):
    content = "a,b,c\n1,,3\nNA,5,None\n"
    write_to_path(temp_csv, content)

    result = load_csv(temp_csv)
    assert result.success
    assert result.row_count == 2

    df = result.df
    assert pd.isna(df.iloc[0]["b"])
    assert pd.isna(df.iloc[1]["a"])
    assert pd.isna(df.iloc[1]["c"])


# ── 5. Large Files (Chunking) ─────────────────────────────────


def test_load_large_file_chunking(temp_csv, monkeypatch):
    import app.ml.csv_loader

    # Monkeypatch the chunk threshold to 0 to force chunked reading
    monkeypatch.setattr(app.ml.csv_loader, "CHUNK_THRESHOLD_MB", 0)
    monkeypatch.setattr(app.ml.csv_loader, "CHUNK_ROWS", 2)

    # 5 rows total
    content = "col1,col2\n1,A\n2,B\n3,C\n4,D\n5,E\n"
    write_to_path(temp_csv, content)

    result = load_csv(temp_csv)
    assert result.success
    assert result.row_count == 5
    assert result.col_count == 2


# ── 6. Bytes Loader ───────────────────────────────────────────


def test_load_csv_from_bytes_success():
    content = b"a,b,c\n1,2,3\n4,5,6\n"
    result = load_csv_from_bytes(content)

    assert result.success
    assert result.row_count == 2
    assert result.col_count == 3
    assert result.detected_delimiter == ","
    assert result.df.iloc[0]["a"] == 1


def test_load_csv_from_bytes_different_delimiter():
    content = b"a|b|c\n1|2|3\n4|5|6\n"
    result = load_csv_from_bytes(content)

    assert result.success
    assert result.row_count == 2
    assert result.col_count == 3
    assert result.detected_delimiter == "|"
