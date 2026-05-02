"""Integration test for the Arlington County (VA) property search.

Card: parcel 12-027-002 — N Arlington Mill Dr, Dominion Hills Park
(County-owned vacant park land, no improvements).

Exercises:
- Browser-UA HTTP fetching (Arlington blocks the default httpx UA)
- HTML→PDF conversion via wkhtmltopdf
- Sparse-page extraction (the Improvements tab has minimal content
  for parcels with no improvements)

Requires:
- Network access to fetch the HTML property card
- wkhtmltopdf installed (system dependency for pdfkit)
- Ollama running with the configured CARD_READER_EXTRACTION_MODEL

Run:
    pytest src/tests/test_arlington_va.py -v -s
"""

from __future__ import annotations

import json
import logging

import httpx
import pytest

from ..config import CARD_READER_OLLAMA_HOST

logger = logging.getLogger(__name__)

ARLINGTON_URL = "https://propertysearch.arlingtonva.us/Home/Improvements?lrsn=18059"


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
def arlington_result():
    from .. import read_property_card

    data, photo = read_property_card(ARLINGTON_URL)
    logger.info("Arlington extracted:\n%s", json.dumps(data, indent=2, default=str))
    return data, photo


class TestArlingtonVa:
    """Card lrsn=18059 — vacant Dominion Hills park parcel.

    The Improvements tab shows only the parcel header
    ('12-027-002 N ARLINGTON MILL DR ARLINGTON VA 22205') and the
    notice 'No Improvement data available'. So we only assert the
    location facts that are actually present on this page.
    """

    def test_parcelid(self, arlington_result):
        data, _ = arlington_result
        pid = str(data.get("parcelid", "")).upper()
        assert "12-027-002" in pid or "12-027" in pid, (
            f"parcelid: expected to contain '12-027-002', got {pid!r}"
        )

    def test_parceladdr(self, arlington_result):
        data, _ = arlington_result
        addr = str(data.get("parceladdr", "")).upper()
        assert "ARLINGTON MILL" in addr, (
            f"parceladdr: expected to contain 'ARLINGTON MILL', got {addr!r}"
        )

    def test_parcelcity(self, arlington_result):
        data, _ = arlington_result
        city = str(data.get("parcelcity", "")).upper()
        assert "ARLINGTON" in city, f"parcelcity: got {city!r}"

    def test_parcelstate(self, arlington_result):
        data, _ = arlington_result
        state = str(data.get("parcelstate", "")).upper()
        assert state in ("VA", "VIRGINIA"), f"parcelstate: got {state!r}"

    def test_parcelzip(self, arlington_result):
        data, _ = arlington_result
        zip_code = str(data.get("parcelzip", ""))
        assert "22205" in zip_code, f"parcelzip: got {zip_code!r}"

    def test_print_data(self, arlington_result, capsys):
        data, photo = arlington_result
        with capsys.disabled():
            print("\n=== Arlington VA lrsn=18059 ===")
            print(json.dumps(data, indent=2, default=str))
            print(f"Photo: {'yes (' + str(len(photo)) + ' bytes)' if photo else 'none'}")
