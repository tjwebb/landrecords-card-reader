"""Integration test for the HTML-to-PDF property card extraction path.

Requires:
- Network access to fetch the HTML property card
- wkhtmltopdf installed (system dependency for pdfkit)
- Ollama running with the configured CARD_READER_EXTRACTION_MODEL

Run:
    pytest src/tests/test_html.py -v -s
"""

from __future__ import annotations

import json
import logging

import httpx
import pytest

from ..config import CARD_READER_OLLAMA_HOST

logger = logging.getLogger(__name__)

PULASKI_URL = "https://www.webgis.net/LinkedFiles/va/pulaski/pc/cards/PC17759.htm"


def _ollama_reachable() -> bool:
    try:
        with httpx.Client(timeout=3.0) as client:
            r = client.get(f"{CARD_READER_OLLAMA_HOST}/api/tags")
            return r.status_code == 200
    except Exception:
        return False


def _pdfkit_available() -> bool:
    try:
        import pdfkit
        pdfkit.configuration()
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.skipif(not _ollama_reachable(), reason="Ollama not reachable"),
    pytest.mark.skipif(not _pdfkit_available(), reason="pdfkit/wkhtmltopdf not available"),
]


@pytest.fixture(scope="module")
def pulaski_result():
    """Run the full pipeline against the Pulaski County HTML property card.

    Calls each pipeline step directly (rather than the high-level
    ``read_property_card``) so the raw OCR text can be logged for
    inspection — this card has historically caused glyph-merging issues
    in the HTML→PDF→OCR path (e.g. "511E BROWNTOWNF RD") and the OCR
    output is the single most useful artifact for debugging them.
    """
    from ..nodes import (
        download_pdf,
        extract_data,
        extract_pdf_and_photos,
        fill_from_photo,
    )

    state: dict = {
        "pdf_url": PULASKI_URL,
        "pdf_bytes": None,
        "pdf_content": b"",
        "pdf_text": "",
        "property_photos": [],
        "property_data": {},
        "result": "",
        "context": None,
    }

    state.update(download_pdf(state))
    state.update(extract_pdf_and_photos(state))

    logger.info(
        "=== Pulaski OCR text (%d chars) ===\n%s\n=== end Pulaski OCR text ===",
        len(state["pdf_text"]), state["pdf_text"],
    )

    state.update(extract_data(state))

    photos = state.get("property_photos", [])
    photo = photos[0]["bytes"] if photos else None
    data = state["property_data"]
    if photo:
        data = fill_from_photo(data, photo)

    logger.info("Extracted data:\n%s", json.dumps(data, indent=2, default=str))
    return data, photo


class TestHtmlPropertyCard:
    def test_returns_data(self, pulaski_result):
        data, _ = pulaski_result
        assert isinstance(data, dict)
        assert len(data) > 0, "Should extract at least some fields"

    def test_parcelid(self, pulaski_result):
        data, _ = pulaski_result
        pid = str(data.get("parcelid", "")).upper()
        assert "072-051-0027-008A" in pid or "072" in pid, (
            f"parcelid: expected to contain '072-051-0027-008A', got {pid!r}"
        )

    def test_ownername(self, pulaski_result):
        data, _ = pulaski_result
        owner = str(data.get("ownername", "")).upper()
        assert "ROSSI" in owner, f"ownername: expected to contain 'ROSSI', got {owner!r}"

    def test_parceladdr(self, pulaski_result):
        data, _ = pulaski_result
        addr = str(data.get("parceladdr", "")).upper()
        assert "7TH" in addr, f"parceladdr: expected to contain '7TH', got {addr!r}"

    def test_parcelstate(self, pulaski_result):
        data, _ = pulaski_result
        state = str(data.get("parcelstate", "")).upper()
        assert state in ("VA", "VIRGINIA"), f"parcelstate: expected VA, got {state!r}"

    def test_landvalue(self, pulaski_result):
        data, _ = pulaski_result
        land = data.get("landvalue")
        assert land == 17000, f"landvalue: expected 17000, got {land!r}"

    def test_imprvalue(self, pulaski_result):
        data, _ = pulaski_result
        impr = data.get("imprvalue")
        assert impr == 96600, f"imprvalue: expected 96600, got {impr!r}"

    def test_totalvalue(self, pulaski_result):
        data, _ = pulaski_result
        total = data.get("totalvalue")
        assert total == 113600, f"totalvalue: expected 113600, got {total!r}"

    def test_value_consistency(self, pulaski_result):
        data, _ = pulaski_result
        land = data.get("landvalue")
        impr = data.get("imprvalue")
        total = data.get("totalvalue")
        if isinstance(land, int) and isinstance(impr, int) and isinstance(total, int):
            assert land + impr == total, (
                f"land({land}) + impr({impr}) = {land + impr} != total({total})"
            )

    def test_zoningcode(self, pulaski_result):
        data, _ = pulaski_result
        zoning = str(data.get("zoningcode", "")).upper()
        assert "R2" in zoning, f"zoningcode: expected to contain 'R2', got {zoning!r}"

    def test_print_data(self, pulaski_result, capsys):
        data, photo = pulaski_result
        with capsys.disabled():
            print(f"\n=== Pulaski HTML card ===")
            print(json.dumps(data, indent=2, default=str))
            print(f"Photo: {'yes (' + str(len(photo)) + ' bytes)' if photo else 'none'}")
