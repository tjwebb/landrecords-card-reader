"""Integration test for the Arlington County (VA) property search.

Card: parcel 12-027-002 — N Arlington Mill Dr, Dominion Hills Park.
County-owned vacant park land, no improvements; tax-exempt.

The page is the General Information tab, which exposes:
  - parcel ID, physical address (N Arlington Mill Dr, Arlington, VA 22205)
  - owner ("COUNTY BOARD OF ARLINGTON")
  - legal description ("LOTS 120, 121, SEC 1 DOMINION HILLS 10,412 SQ FT")
  - zoning S-3A, property class "200-GenCom VacLand-no siteplan"
  - explicit "Year Built N/A" — no improvements on this lot

Exercises:
- Browser-UA HTTP fetching (Arlington blocks the default httpx UA
  with a 403 from its Azure App Gateway)
- HTML→PDF conversion via wkhtmltopdf
- Tax-exempt / no-improvement extraction (no yearbuilt, no bldgsqft,
  no values to assert against)

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

ARLINGTON_URL = "https://propertysearch.arlingtonva.us/Home/GeneralInformation?lrsn=18059"


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
    """Card lrsn=18059 — Dominion Hills Park, owned by Arlington County."""

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

    def test_ownername(self, arlington_result):
        """Owner: 'COUNTY BOARD OF ARLINGTON'."""
        data, _ = arlington_result
        owner = str(data.get("ownername", "")).upper()
        assert "ARLINGTON" in owner and (
            "COUNTY BOARD" in owner or "COUNTY" in owner
        ), f"ownername: expected 'COUNTY BOARD OF ARLINGTON', got {owner!r}"

    def test_zoningcode(self, arlington_result):
        """Zoning: S-3A."""
        data, _ = arlington_result
        zoning = str(data.get("zoningcode", "")).upper()
        assert "S-3A" in zoning or "S3A" in zoning, f"zoningcode: got {zoning!r}"

    def test_legaldesc(self, arlington_result):
        """Legal Description: 'LOTS 120, 121, SEC 1 DOMINION HILLS 10,412 SQ FT'."""
        data, _ = arlington_result
        legal = str(data.get("legaldesc", "")).upper()
        assert "DOMINION HILLS" in legal or ("120" in legal and "121" in legal), (
            f"legaldesc: expected to mention DOMINION HILLS or lots 120/121, "
            f"got {legal!r}"
        )

    def test_yearbuilt_absent(self, arlington_result):
        """Card explicitly says 'Year Built N/A'. Per the extraction rules,
        N/A placeholders must be OMITTED from the output — never substituted
        with 0, 'NA', or any sentinel value. This is the regression target:
        the LLM must not invent a year for a no-improvement parcel."""
        data, _ = arlington_result
        yb = data.get("yearbuilt")
        assert yb is None, (
            f"yearbuilt: card shows 'Year Built N/A' for this vacant parcel; "
            f"expected absent, got {yb!r}"
        )

    def test_print_data(self, arlington_result, capsys):
        data, photo = arlington_result
        with capsys.disabled():
            print("\n=== Arlington VA lrsn=18059 (Dominion Hills Park) ===")
            print(json.dumps(data, indent=2, default=str))
            print(f"Photo: {'yes (' + str(len(photo)) + ' bytes)' if photo else 'none'}")
