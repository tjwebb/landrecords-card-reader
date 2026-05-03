"""Integration test for the Stafford County (VA) PublicAccessNow card.

Card: 412 Lexington Ct, Patton — single-family, heat pump (no A/C),
Grafton Village.

Requires:
- Network access to fetch the HTML property card
- wkhtmltopdf installed (system dependency for pdfkit)
- Ollama running with the configured CARD_READER_EXTRACTION_MODEL

Run:
    pytest src/tests/test_stafford_va.py -v -s
"""

from __future__ import annotations

import json
import logging

import httpx
import pytest

from ..config import CARD_READER_OLLAMA_HOST

logger = logging.getLogger(__name__)

STAFFORD_URL = (
    "https://va-stafford-assessor.publicaccessnow.com/PropertySearch/"
    "PropertyDetails.aspx?p=54L%2020%20%20%20317&a=35354"
)


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
def stafford_result():
    from .. import read_property_card

    data, photo = read_property_card(STAFFORD_URL)
    logger.info("Stafford extracted:\n%s", json.dumps(data, indent=2, default=str))
    return data, photo


class TestStaffordVa:
    """Card 54L 20 317 — Patton, 412 Lexington Ct, heat pump (no A/C)."""

    def test_parcelid(self, stafford_result):
        data, _ = stafford_result
        pid = str(data.get("parcelid", "")).upper()
        # Property ID is "54L 20 317"; OCR/HTML extraction may compress
        # internal whitespace, so just check the salient fragments.
        assert "54L" in pid and "317" in pid, f"parcelid: got {pid!r}"

    def test_ownername(self, stafford_result):
        data, _ = stafford_result
        owner = str(data.get("ownername", "")).upper()
        assert "PATTON" in owner, f"ownername: got {owner!r}"

    def test_parceladdr(self, stafford_result):
        data, _ = stafford_result
        addr = str(data.get("parceladdr", "")).upper()
        assert "LEXINGTON" in addr, f"parceladdr: got {addr!r}"

    def test_parcelstate(self, stafford_result):
        data, _ = stafford_result
        state = str(data.get("parcelstate", "")).upper()
        assert state in ("VA", "VIRGINIA"), f"parcelstate: got {state!r}"

    def test_yearbuilt(self, stafford_result):
        data, _ = stafford_result
        assert data.get("yearbuilt") == 1979, (
            f"yearbuilt: got {data.get('yearbuilt')!r}"
        )

    def test_bldgsqft(self, stafford_result):
        data, _ = stafford_result
        assert data.get("bldgsqft") == 1050, (
            f"bldgsqft: got {data.get('bldgsqft')!r}"
        )

    def test_taxacres(self, stafford_result):
        data, _ = stafford_result
        acres = data.get("taxacres")
        assert acres is not None and abs(acres - 0.2323) < 0.01, (
            f"taxacres: got {acres!r}"
        )

    def test_landvalue(self, stafford_result):
        """2026 reassessment land value (most recent year)."""
        data, _ = stafford_result
        assert data.get("landvalue") == 105000, (
            f"landvalue: got {data.get('landvalue')!r}"
        )

    def test_imprvalue(self, stafford_result):
        data, _ = stafford_result
        assert data.get("imprvalue") == 180000, (
            f"imprvalue: got {data.get('imprvalue')!r}"
        )

    def test_totalvalue(self, stafford_result):
        data, _ = stafford_result
        assert data.get("totalvalue") == 285000, (
            f"totalvalue: got {data.get('totalvalue')!r}"
        )

    def test_value_consistency(self, stafford_result):
        data, _ = stafford_result
        land = data.get("landvalue")
        impr = data.get("imprvalue")
        total = data.get("totalvalue")
        if isinstance(land, int) and isinstance(impr, int) and isinstance(total, int):
            assert land + impr == total, (
                f"land({land}) + impr({impr}) = {land + impr} != total({total})"
            )

    def test_heating_heat_pump(self, stafford_result):
        """Card explicitly says 'Heating: Heat pump' — heating must reflect that."""
        data, _ = stafford_result
        heating = str(data.get("heating", "")).upper()
        assert "HEAT PUMP" in heating, (
            f"heating: expected 'HEAT PUMP', got {heating!r}"
        )

    def test_heatfuel_electric(self, stafford_result):
        """Heat pump implies electric fuel (per prompt inference rules)."""
        data, _ = stafford_result
        fuel = str(data.get("heatfuel", "")).upper()
        assert fuel == "ELECTRIC", f"heatfuel: expected ELECTRIC, got {fuel!r}"

    def test_cooling_none(self, stafford_result):
        """Card explicitly says 'A/C None' — cooling must be NONE."""
        data, _ = stafford_result
        cooling = str(data.get("cooling", "")).upper()
        assert cooling == "NONE", f"cooling: expected NONE, got {cooling!r}"

    def test_extwall_vinyl(self, stafford_result):
        data, _ = stafford_result
        ext = str(data.get("extwall", "")).upper()
        assert "VINYL" in ext, f"extwall: got {ext!r}"

    def test_roofstyle_gable(self, stafford_result):
        data, _ = stafford_result
        rs = str(data.get("roofstyle", "")).upper()
        assert "GABLE" in rs, f"roofstyle: got {rs!r}"

    def test_print_data(self, stafford_result, capsys):
        data, photo = stafford_result
        with capsys.disabled():
            print("\n=== Stafford VA 54L 20 317 ===")
            print(json.dumps(data, indent=2, default=str))
            print(f"Photo: {'yes (' + str(len(photo)) + ' bytes)' if photo else 'none'}")
