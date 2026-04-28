"""Integration tests driven by local PDF fixtures in tests/fixtures/.

These tests exercise the full landrecords-card-reader pipeline against a handful
of real Virginia property cards. For each PDF we:

  1. Load it from disk (no network required for the PDF itself)
  2. Run download_pdf -> extract_property_photos -> extract_pdf_text -> extract_data
  3. Assert the extracted property_data contains the facts we know to be true
     for that card (flexible matching: substrings for text, exact for ints,
     tolerance for floats).

The tests require a reachable Ollama endpoint running the configured
CARD_READER_EXTRACTION_MODEL. They skip automatically if Ollama is unreachable.

Run:
    pytest src/utils/property_card_image_reader/tests/test_fixtures.py -v -s
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import httpx
import pytest

from ..config import CARD_READER_OLLAMA_HOST
from ..nodes import (
    download_pdf,
    extract_data,
    extract_pdf_text,
    extract_property_photos,
    fill_from_photo,
)

logger = logging.getLogger(__name__)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
EXTRACTED_IMAGES_DIR = Path(__file__).parent / "extracted_images"


def _save_property_photos(pdf_name: str, photos: list[dict]) -> int:
    """Write extracted property photos to disk as JPG files.

    Files land in ``tests/extracted_images/<pdf_name>/photo_<idx>.jpg``.
    Previous outputs for this PDF are cleared first so each test run starts
    with a clean directory.
    """
    import io
    from PIL import Image

    out_dir = EXTRACTED_IMAGES_DIR / pdf_name
    if out_dir.exists():
        for f in out_dir.iterdir():
            if f.is_file():
                f.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for i, photo in enumerate(photos):
        try:
            img = Image.open(io.BytesIO(photo["bytes"]))
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            out_path = out_dir / f"photo_{i:03d}.jpg"
            img.save(out_path, "JPEG", quality=90)
            saved += 1
        except Exception as e:
            logger.warning("Failed to save photo %d: %s", i, e)

    logger.info("Wrote %d property photo(s) to %s", saved, out_dir)
    return saved


# ---------------------------------------------------------------------------
# Expected facts per fixture
# ---------------------------------------------------------------------------
# Matchers:
#   ("substr", "text")  -> str(value).upper() contains text.upper()
#   ("int", 12345)      -> int(value) == expected
#   ("float", 1.094)    -> abs(float(value) - expected) <= 0.01
#   ("any_of", [...])   -> value in list (case-insensitive for strings)
#
# Keep these to high-confidence facts readable directly from the PDF text —
# avoid wording-sensitive fields like usedesc/zoningdesc where the LLM may
# paraphrase.
# ---------------------------------------------------------------------------

FIXTURES: dict[str, dict[str, Any]] = {
    "carroll_va_0000028365.pdf": {
        "taxacctnum": ("substr", "21597"),
        "ownername": ("substr", "HICKS"),
        "ownercity": ("substr", "HILLSVILLE"),
        "ownerstate": ("any_of", ["VA"]),
        "ownerzip": ("substr", "24343"),
        "parceladdr": ("substr", "GRAYSON"),
        "yearbuilt": ("int", 1950),
        "bldgsqft": ("int", 864),
        "bedrooms": ("int", 2),
        "landvalue": ("int", 15000),
        "imprvalue": ("int", 74600),
        "totalvalue": ("int", 89600),
    },
    "henry_va_173760004.pdf": {
        "parcelstate": ("any_of", ["VA", 'VIRGINIA']),
        #"taxacctnum": ("substr", "173760004"),
        #"ownername": ("substr", "LEE STREET APARTMENTS"),
        "parceladdr": ("substr", "SPRUCE ST"),
        "landvalue": ("int", 35000),
        "imprvalue": ("int", 545800),
        "totalvalue": ("int", 580800),
        #"yearbuilt": ("int", 2001),
        "taxacres": ("float", 1.094),
    },
    "henry_va_216430002.pdf": {
        "taxacctnum": ("substr", "216430002"),
        "parceladdr": ("substr", "ELIJAH CIR"),
        "yearbuilt": ("int", 1963),
        "taxacres": ("float", 2.034),
        "landvalue": ("int", 11900),
        "imprvalue": ("int", 40600),
        "totalvalue": ("int", 52500),
    },
    "henry_va_139290002.pdf": {
        "taxacctnum": ("substr", "139290002"),
        "ownername": ("substr", "KOGER"),
        "ownercity": ("substr", "BASSETT"),
        "ownerstate": ("any_of", ["VA", "VIRGINIA"]),
        "ownerzip": ("substr", "24055"),
        "parceladdr": ("substr", "SANVILLE SCHOOL"),
        "yearbuilt": ("int", 2000),
        "bldgsqft": ("int", 1539),
        "landvalue": ("int", 27000),
        "imprvalue": ("int", 67700),
        "totalvalue": ("int", 94700),
        "taxacres": ("float", 13.477),
        "heatfuel": ("any_of", ["ELECTRIC"]),
    },
    "henry_va_059800000.pdf": {
        "ownername": ("substr", "BRICKEY"),
        "ownercity": ("substr", "BASSETT"),
        "ownerstate": ("any_of", ["VA", "VIRGINIA"]),
        "parceladdr": ("substr", "ELF"),
        "yearbuilt": ("int", 1985),
        "landvalue": ("int", 14400),
        "imprvalue": ("int", 288000),
        "totalvalue": ("int", 302400),
        "extwall": ("substr", "BRICK"),
        "roofcover": ("substr", "SHINGLES"),
        "heatfuel": ("any_of", ["ELECTRIC"]),
    },
    "henry_va_116650000.pdf": {
        "ownername": ("substr", "PUGH"),
        "ownercity": ("substr", "COLLINSVILLE"),
        "ownerstate": ("any_of", ["VA", "VIRGINIA"]),
        "ownerzip": ("substr", "24078"),
        "parceladdr": ("substr", "MILES"),
        "yearbuilt": ("int", 1956),
        "bldgsqft": ("int", 988),
        "landvalue": ("int", 11500),
        "imprvalue": ("int", 81700),
        "totalvalue": ("int", 93200),
        "extwall": ("substr", "VINYL"),
        "roofcover": ("substr", "SHINGLES"),
        "heatfuel": ("any_of", ["ELECTRIC"]),
    },
    "henry_va_143010001.pdf": {
        # taxacctnum was omitted historically: the printed account number
        # ("143010001") is rasterised on the card and the previous OCR
        # backend (Tesseract) reliably misread it as "1438010001". Worth
        # re-checking now that docTR is in use — if it reads correctly,
        # add `"taxacctnum": ("substr", "143010001")` here.
        "ownername": ("substr", "PEREZ BUENO"),
        "ownercity": ("substr", "MARTINSVILLE"),
        "ownerstate": ("any_of", ["VA", "VIRGINIA"]),
        "parceladdr": ("substr", "LAFAYETTE"),
        "yearbuilt": ("int", 1951),
        "bldgsqft": ("int", 912),
        "landvalue": ("int", 7800),
        "imprvalue": ("int", 58300),
        "totalvalue": ("int", 66100),
        "extwall": ("substr", "ALUMINUM"),
        "roofcover": ("substr", "SHINGLES"),
        "heatfuel": ("any_of", ["OIL"]),
    },
    "patrick_va_5012120A.pdf": {
        "ownername": ("substr", "GRIFFIN"),
        "owneraddr": ("substr", "BROOK"),
        "ownercity": ("substr", "PATRICK SPRINGS"),
        "ownerstate": ("any_of", ["VA"]),
        "ownerzip": ("substr", "24133"),
        "parceladdr": ("substr", "BROOK"),
        "landvalue": ("int", 12000),
        "totalvalue": ("int", 85000),
        "yearbuilt": ("int", 2000),
        "taxacres": ("float", 1.665),
    },
    "russell_va_230506.pdf": {
        "parcelid": ("substr", "230506"),
        "ownername": ("substr", "HUFFMAN"),
        "parceladdr": ("substr", "SUNNY POINT"),
        "parcelcity": ("substr", "CASTLEWOOD"),
        "parcelstate": ("any_of", ["VA", 'VIRGINIA']),
        "parcelzip": ("substr", "24224"),
        #"saleamt": ("int", 130000),
    },
    "wythe_va_13436.pdf": {
        "taxacctnum": ("substr", "14662"),
        "parceladdr": ("substr", "WHIPPOORWILL"),
        "parcelstate": ("any_of", ["VA", 'VIRGINIA']),
    },
    "wythe_va_4328.pdf": {
        "ownername": ("substr", "SMITH"),
        "ownercity": ("substr", "WYTHEVILLE"),
        "ownerstate": ("any_of", ["VA", "VIRGINIA"]),
        "ownerzip": ("substr", "24382"),
        "parceladdr": ("substr", "PEPPERS FERRY"),
        "yearbuilt": ("int", 1986),
        "bldgsqft": ("int", 2225),
        "bedrooms": ("int", 3),
        "fullbaths": ("int", 2),
        "fireplaces": ("int", 1),
        "landvalue": ("int", 54300),
        "imprvalue": ("int", 244400),
        "totalvalue": ("int", 298700),
        "extwall": ("substr", "BRICK"),
        "roofcover": ("substr", "SHINGLE"),
        "roofstyle": ("substr", "GABLE"),
        "heatfuel": ("any_of", ["ELECTRIC"]),
        "heating": ("substr", "HEAT PUMP"),
    },
    "travis_tx_372462.pdf": {
        "parcelid": ("substr", "372462"),
        "ownername": ("substr", "PARR"),
        "owneraddr": ("substr", "HITCHER"),
        "ownercity": ("substr", "AUSTIN"),
        "ownerstate": ("any_of", ["TX", 'TEXAS']),
        "ownerzip": ("substr", "78749"),
        "parceladdr": ("substr", "HITCHER"),
        "parcelstate": ("any_of", ["TX", 'TEXAS']),
        "parcelzip": ("substr", "78749"),
        "zoningcode": ("substr", "SF-2"),
        "legaldesc": ("substr", "WESTERN OAKS"),
        "yearbuilt": ("int", 1999),
        "bldgsqft": ("int", 1851),
        "landvalue": ("int", 250000),
        "imprvalue": ("int", 278783),
        "totalvalue": ("int", 528783),
        "taxacres": ("float", 0.1778),
    },
    "manassas_va_33894.pdf": {
        "taxacctnum": ("substr", "33894"),
        "ownername": ("substr", "JACQUES"),
        "parceladdr": ("substr", "9112 MAIN ST"),
        "parcelcity": ("substr", "MANASSAS"),
        "parcelstate": ("any_of", ["VA", "VIRGINIA"]),
        "parcelzip": ("substr", "20110"),
        "zoningcode": ("substr", "R1"),
        "yearbuilt": ("int", 1900),
        "landvalue": ("int", 183500),
        "imprvalue": ("int", 664500),
        "totalvalue": ("int", 848000),
        "saleamt": ("int", 695000),
        "bedrooms": ("int", 4),
    },
    "williamsburg_va_160.pdf": {
        "parcelid": ("any_of", ["3208.794.862", "372-06-07-608"]),
        "ownercity": ("substr", "NORFOLK"),
        "ownerstate": ("any_of", ["VA", "VIRGINIA"]),
        "ownerzip": ("substr", "23507"),
        "parceladdr": ("substr", "SETTLEMENT"),
        "parcelstate": ("any_of", ["VA", "VIRGINIA"]),
        "zoningcode": ("substr", "RM-2"),
        #"yearbuilt": ("int", 2001),
        "landvalue": ("int", 11200),
        "imprvalue": ("int", 226000),
        "totalvalue": ("int", 237200),
        #"saleamt": ("int", 170000),
        "bedrooms": ("int", 2),
        "fullbaths": ("int", 2),
        "halfbaths": ("int", 1),
    },
    "beacon_steuben_in.pdf": {
        "parcelid": ("substr", "76-06-11-420-151.000-012"),
        "ownername": ("substr", "TIMMONS"),
        "owneraddr": ("substr", "DUNCASTLE"),
        "ownercity": ("substr", "ANGOLA"),
        "ownerstate": ("any_of", ["IN", "INDIANA"]),
        "ownerzip": ("substr", "46703"),
        "parceladdr": ("substr", "DUNCASTLE"),
        "legaldesc": ("substr", "GLENDARIN HILLS"),
        "yearbuilt": ("int", 2014),
        "bldgsqft": ("int", 1724),
        "bedrooms": ("int", 3),
        "fullbaths": ("int", 2),
        #"halfbaths": ("int", 2),
        "landvalue": ("int", 63800),
        "imprvalue": ("int", 256900),
        "totalvalue": ("int", 320700),
        #"saleamt": ("int", 200000),
        "taxacres": ("float", 0.28),
    },
}

# Expected number of property photos per fixture. These are confirmed by
# manual inspection — Henry has two pictures of the structure on page 1, and
# Wythe has two on page 2. Patrick and Russell have no embedded photos.
EXPECTED_PHOTO_COUNTS: dict[str, int] = {
    "carroll_va_0000028365.pdf": 1,
    "henry_va_173760004.pdf": 1,
    "henry_va_216430002.pdf": 1,
    "henry_va_059800000.pdf": 1,
    "henry_va_116650000.pdf": 1,
    "henry_va_139290002.pdf": 1,
    "henry_va_143010001.pdf": 1,
    "manassas_va_33894.pdf": 1,
    "patrick_va_5012120A.pdf": 0,
    "russell_va_230506.pdf": 0,
    "travis_tx_372462.pdf": 0,
    "williamsburg_va_160.pdf": 2,
    "wythe_va_13436.pdf": 1,
    "wythe_va_4328.pdf": 1,
    "beacon_steuben_in.pdf": 0,
}


# ---------------------------------------------------------------------------
# Ollama availability check — skip all tests in this file if unreachable
# ---------------------------------------------------------------------------

def _ollama_reachable() -> bool:
    try:
        with httpx.Client(timeout=3.0) as client:
            r = client.get(f"{CARD_READER_OLLAMA_HOST}/api/tags")
            return r.status_code == 200
    except Exception as e:
        logger.info("Ollama not reachable at %s: %s", CARD_READER_OLLAMA_HOST, e)
        return False


pytestmark = pytest.mark.skipif(
    not _ollama_reachable(),
    reason=f"Ollama not reachable at {CARD_READER_OLLAMA_HOST}",
)


# ---------------------------------------------------------------------------
# Fixture: discover PDFs and run the full pipeline once per file
# ---------------------------------------------------------------------------

def _discovered_pdfs() -> list[str]:
    files = sorted(p.name for p in FIXTURES_DIR.glob("*.pdf"))
    # Sanity check: every file we discovered on disk has an EXPECTED entry.
    # If a new PDF is dropped into fixtures/ without expected values, fail
    # loudly rather than silently skipping assertions.
    unknown = [f for f in files if f not in FIXTURES]
    if unknown:
        raise AssertionError(
            f"PDFs in fixtures/ without expected values in FIXTURES dict: {unknown}"
        )
    return files


@pytest.fixture(scope="module", params=_discovered_pdfs())
def pipeline_result(request) -> dict:
    """Run the end-to-end extraction pipeline against one fixture PDF."""
    pdf_name: str = request.param
    pdf_path = FIXTURES_DIR / pdf_name
    pdf_bytes = pdf_path.read_bytes()

    state: dict = {
        "pdf_url": f"file://{pdf_path}",
        "pdf_bytes": pdf_bytes,
        "pdf_content": b"",
        "pdf_text": "",
        "property_photos": [],
        "property_data": {},
        "result": "",
    }

    logger.info("=== Pipeline for %s (%d bytes) ===", pdf_name, len(pdf_bytes))

    from concurrent.futures import ThreadPoolExecutor

    state.update(download_pdf(state))
    assert state["pdf_content"], "download_pdf should populate pdf_content"

    # Photo extraction and text extraction (incl. docTR OCR) run in parallel.
    with ThreadPoolExecutor(max_workers=2) as pool:
        photos_future = pool.submit(extract_property_photos, state)
        text_future = pool.submit(extract_pdf_text, state)
        state.update(photos_future.result())
        state.update(text_future.result())

    # Write property photos to disk for inspection.
    _save_property_photos(pdf_name, state.get("property_photos", []))
    state.update(extract_data(state))

    # Analyze property photo to fill missing fields.
    photos = state.get("property_photos", [])
    if photos:
        state["property_data"] = fill_from_photo(
            state["property_data"], photos[0]["bytes"],
        )

    # Tag for downstream inspection/debugging
    state["_pdf_name"] = pdf_name
    return state


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------

def _assert_match(field: str, actual: Any, matcher: tuple) -> None:
    kind, expected = matcher
    if kind == "substr":
        assert actual is not None, f"{field}: expected substring {expected!r}, got None"
        assert expected.upper() in str(actual).upper(), (
            f"{field}: expected to contain {expected!r}, got {actual!r}"
        )
    elif kind == "int":
        assert actual is not None, f"{field}: expected {expected!r}, got None"
        assert int(actual) == int(expected), (
            f"{field}: expected {expected!r}, got {actual!r}"
        )
    elif kind == "float":
        assert actual is not None, f"{field}: expected ~{expected!r}, got None"
        assert abs(float(actual) - float(expected)) <= 0.01, (
            f"{field}: expected ~{expected!r}, got {actual!r}"
        )
    elif kind == "any_of":
        assert actual is not None, f"{field}: expected one of {expected!r}, got None"
        vals = {str(v).upper() for v in expected}
        assert str(actual).upper() in vals, (
            f"{field}: expected one of {expected!r}, got {actual!r}"
        )
    else:
        raise AssertionError(f"Unknown matcher kind {kind!r} for field {field}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPipeline:
    def test_discovered_fixtures_not_empty(self):
        assert _discovered_pdfs(), "No PDF fixtures found"

    def test_pdf_content_populated(self, pipeline_result):
        assert len(pipeline_result["pdf_content"]) > 0

    def test_text_has_content(self, pipeline_result):
        assert len(pipeline_result["pdf_text"]) > 50, (
            f"{pipeline_result['_pdf_name']}: text extraction produced <50 chars "
            f"(len={len(pipeline_result['pdf_text'])})"
        )

    def test_expected_fields(self, pipeline_result):
        """Check every declared expected fact for this fixture."""
        pdf_name = pipeline_result["_pdf_name"]
        data = pipeline_result["property_data"]
        expected = FIXTURES[pdf_name]

        failures: list[str] = []
        for field, matcher in expected.items():
            try:
                _assert_match(field, data.get(field), matcher)
            except AssertionError as e:
                failures.append(str(e))

        if failures:
            # Include the full extracted data to make debugging easier
            import json
            pretty = json.dumps(data, indent=2, default=str)
            raise AssertionError(
                f"{pdf_name}: {len(failures)}/{len(expected)} expected facts failed:\n"
                + "\n".join(f"  - {f}" for f in failures)
                + f"\n\nExtracted data:\n{pretty}"
            )

    def test_property_photos_count(self, pipeline_result):
        pdf_name = pipeline_result["_pdf_name"]
        photos = pipeline_result["property_photos"]
        expected = EXPECTED_PHOTO_COUNTS[pdf_name]
        assert len(photos) == expected, (
            f"{pdf_name}: expected {expected} property photo(s), got {len(photos)}: "
            f"{[(p['page'], p['width'], p['height'], p['ext']) for p in photos]}"
        )

    def test_property_photos_have_valid_bytes(self, pipeline_result):
        """Each extracted photo must have non-empty image bytes with a recognizable header."""
        # Magic-byte prefixes for the formats pymupdf is likely to return.
        magic = {
            "jpeg": b"\xff\xd8",
            "jpg": b"\xff\xd8",
            "png": b"\x89PNG\r\n\x1a\n",
            "gif": b"GIF8",
            "bmp": b"BM",
            "tiff": (b"II*\x00", b"MM\x00*"),
            "tif": (b"II*\x00", b"MM\x00*"),
        }
        for i, p in enumerate(pipeline_result["property_photos"]):
            data = p["bytes"]
            assert isinstance(data, (bytes, bytearray)) and len(data) > 0, (
                f"photo[{i}] has empty bytes"
            )
            ext = (p.get("ext") or "").lower()
            expected_magic = magic.get(ext)
            if expected_magic is None:
                # Unknown format — at least confirm it's not empty.
                continue
            if isinstance(expected_magic, tuple):
                assert any(data.startswith(m) for m in expected_magic), (
                    f"photo[{i}] ext={ext!r} did not match any expected magic bytes"
                )
            else:
                assert data.startswith(expected_magic), (
                    f"photo[{i}] ext={ext!r} did not start with expected magic bytes"
                )

    def test_print_extracted_data(self, pipeline_result, capsys):
        """Dump extracted data for each fixture — handy with `pytest -s`."""
        import json
        pdf_name = pipeline_result["_pdf_name"]
        with capsys.disabled():
            print(f"\n=== {pdf_name} ===")
            print(json.dumps(pipeline_result["property_data"], indent=2, default=str))
