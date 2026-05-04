"""Integration test for Richmond County VA property card record 878 (PRC 837).

Card: 518 Menokin Rd, Warsaw — single-family dwelling on 1.3 ac in the
Marshall district. Built 1940 (effective 1980), 880 sqft, 2 br / 5 rm,
crawl/concrete foundation, gable comp-shingle roof, drywall interior,
electric C-Heat + C-Air. One 16x20 storage shed.

Source page is HTML wrapping two JPEG scans of the paper card —
``download_pdf`` renders the HTML to PDF (wkhtmltopdf) and the OCR path
in ``extract_pdf_and_photos`` reads the embedded scans. Assertions are
deliberately lenient because OCR over fax-scanned grids is noisy.

Most recent reassessment (printed 2023):
    Land $23,420 + Bldg $60,888 + OBldg $1,920 = Appraised $86,228.

Requires:
- Network access to fetch the HTML + scan images
- wkhtmltopdf installed (system dependency for pdfkit)
- Ollama running with the configured CARD_READER_EXTRACTION_MODEL
- docTR weights cached (downloaded on first OCR run)

Run:
    pytest src/tests/test_richmond_va_837.py -v -s
"""

from __future__ import annotations

import json
import logging

import httpx
import pytest

from ..config import CARD_READER_OLLAMA_HOST

logger = logging.getLogger(__name__)

CARD_URL = "https://www.richmondcountypropertycards.com/prc/837"


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
    pytest.mark.skipif(
        not _pdfkit_available(), reason="pdfkit/wkhtmltopdf not available",
    ),
]


@pytest.fixture(scope="module")
def card_result():
    from .. import read_property_card

    data, photo = read_property_card(CARD_URL)
    logger.info(
        "Richmond 837 extracted:\n%s",
        json.dumps(data, indent=2, default=str),
    )
    return data, photo


class TestRichmondVa837:
    """Record 878 — 518 Menokin Rd, Warsaw, Self/Fones, 1940 dwelling."""

    def test_ownername(self, card_result):
        """Card shows two deeded-owner lines: 'SELF PENELOPE J' and
        'FONES PENELOPE SELF'. Either surname is acceptable."""
        data, _ = card_result
        owner = str(data.get("ownername", "")).upper()
        assert "SELF" in owner or "FONES" in owner or "PENELOPE" in owner, (
            f"ownername: got {owner!r}"
        )

    def test_parceladdr(self, card_result):
        """Physical 911 Address: '518 MENOKIN RD'."""
        data, _ = card_result
        addr = str(data.get("parceladdr", "")).upper()
        assert "MENOKIN" in addr, f"parceladdr: got {addr!r}"

    def test_parcelstate(self, card_result):
        data, _ = card_result
        state = str(data.get("parcelstate", "")).upper()
        assert state in ("VA", "VIRGINIA"), f"parcelstate: got {state!r}"

    def test_yearbuilt(self, card_result):
        """Card shows 'YR BLT 1940'."""
        data, _ = card_result
        assert data.get("yearbuilt") == 1940, (
            f"yearbuilt: got {data.get('yearbuilt')!r}"
        )

    def test_bedrooms(self, card_result):
        """Building Information: 'BDRMS 2'."""
        data, _ = card_result
        assert data.get("bedrooms") == 2, f"bedrooms: got {data.get('bedrooms')!r}"

    def test_bldgsqft(self, card_result):
        """HSQFT 880 / Building section SQFT 880."""
        data, _ = card_result
        assert data.get("bldgsqft") == 880, (
            f"bldgsqft: got {data.get('bldgsqft')!r}"
        )

    def test_taxacres(self, card_result):
        """ACRES 1.3000 (1.0 homesite + 0.3 woodstacre)."""
        data, _ = card_result
        acres = data.get("taxacres")
        assert acres is not None and abs(acres - 1.3) < 0.05, (
            f"taxacres: got {acres!r}"
        )

    def test_landvalue(self, card_result):
        """Parcel summary: LAND VALUE $23,420."""
        data, _ = card_result
        assert data.get("landvalue") == 23420, (
            f"landvalue: got {data.get('landvalue')!r}"
        )

    def test_totalvalue(self, card_result):
        """Parcel summary: APPRAISED VALUE / TAXABLE VALUE $86,228."""
        data, _ = card_result
        assert data.get("totalvalue") == 86228, (
            f"totalvalue: got {data.get('totalvalue')!r}"
        )

    def test_imprvalue_includes_building(self, card_result):
        """Bldg=$60,888, OBldg=$1,920. Different assessors aggregate
        improvements differently — accept the building-only number,
        the bldg+outbldg combined number, or anything that, summed
        with the $23,420 land value, lands on the $86,228 total."""
        data, _ = card_result
        impr = data.get("imprvalue")
        assert impr in (60888, 62808), f"imprvalue: got {impr!r}"

    def test_value_consistency(self, card_result):
        data, _ = card_result
        land = data.get("landvalue")
        impr = data.get("imprvalue")
        total = data.get("totalvalue")
        if isinstance(land, int) and isinstance(impr, int) and isinstance(total, int):
            # Tolerate the $1,920 outbuilding being booked under either
            # imprvalue or as a separate line not reflected in the sum.
            diff = abs(land + impr - total)
            assert diff <= 1920, (
                f"land({land}) + impr({impr}) = {land + impr}, "
                f"total({total}); diff {diff} exceeds outbuilding allowance"
            )

    def test_heating_central(self, card_result):
        """Building section: HEAT 'C-HEAT' (central heat)."""
        data, _ = card_result
        heating = str(data.get("heating", "")).upper()
        assert heating, f"heating: expected non-empty, got {heating!r}"
        assert (
            "CENTRAL" in heating
            or "FORCED" in heating
            or "WARM" in heating
        ), f"heating: expected central/forced/warm, got {heating!r}"

    def test_cooling_central_air(self, card_result):
        """Building section: 'C-AIR' (central air)."""
        data, _ = card_result
        cooling = str(data.get("cooling", "")).upper()
        assert "CENTRAL" in cooling or "AIR" in cooling, (
            f"cooling: expected central air, got {cooling!r}"
        )

    def test_heatfuel_electric(self, card_result):
        """Building Properties: 'FUEL TYPE ELECTRIC'."""
        data, _ = card_result
        fuel = str(data.get("heatfuel", "")).upper()
        assert "ELECTRIC" in fuel or "ELEC" in fuel, f"heatfuel: got {fuel!r}"

    def test_roofstyle_gable(self, card_result):
        """Roof Type: GABLE."""
        data, _ = card_result
        rs = str(data.get("roofstyle", "")).upper()
        assert "GABLE" in rs, f"roofstyle: got {rs!r}"

    def test_roofcover_composition(self, card_result):
        """Roof Material: 'COMP SHGLS' (composition shingles)."""
        data, _ = card_result
        rc = str(data.get("roofcover", "")).upper()
        assert (
            "COMP" in rc
            or "SHINGLE" in rc
            or "ASPHALT" in rc
            or "SHGL" in rc
        ), f"roofcover: got {rc!r}"

    def test_zoningcode(self, card_result):
        """Land Properties: 'ZONING A-1'."""
        data, _ = card_result
        zoning = str(data.get("zoningcode", "")).upper()
        assert "A-1" in zoning or "A1" in zoning, f"zoningcode: got {zoning!r}"

    def test_print_data(self, card_result, capsys):
        data, photo = card_result
        with capsys.disabled():
            print("\n=== Richmond VA 837 (518 Menokin Rd) ===")
            print(json.dumps(data, indent=2, default=str))
            print(f"Photo: {'yes (' + str(len(photo)) + ' bytes)' if photo else 'none'}")
