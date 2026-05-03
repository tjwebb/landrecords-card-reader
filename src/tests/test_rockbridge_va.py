"""Integration test for the Rockbridge County (VA) webgis property card.

Card: 136 Pinehurst Dr, Trimble — single-family on 0.614 acres in
Meadows at Woods Creek (Buffalo District). Heat pump with central A/C.

Requires:
- Network access to fetch the HTML property card
- wkhtmltopdf installed (system dependency for pdfkit)
- Ollama running with the configured CARD_READER_EXTRACTION_MODEL

Run:
    pytest src/tests/test_rockbridge_va.py -v -s
"""

from __future__ import annotations

import json
import logging

import httpx
import pytest

from ..config import CARD_READER_OLLAMA_HOST

logger = logging.getLogger(__name__)

ROCKBRIDGE_URL = "https://www.webgis.net/va/rockbridge/pc.php?lrsn=18349"


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
def rockbridge_result():
    from .. import read_property_card

    data, photo = read_property_card(ROCKBRIDGE_URL)
    logger.info("Rockbridge extracted:\n%s", json.dumps(data, indent=2, default=str))
    return data, photo


class TestRockbridgeVa:
    """Card record 18349 — Trimble, 136 Pinehurst Dr, heat pump + central A/C."""

    def test_ownername(self, rockbridge_result):
        data, _ = rockbridge_result
        owner = str(data.get("ownername", "")).upper()
        assert "TRIMBLE" in owner, f"ownername: got {owner!r}"

    def test_parceladdr(self, rockbridge_result):
        data, _ = rockbridge_result
        addr = str(data.get("parceladdr", "")).upper()
        assert "PINEHURST" in addr, f"parceladdr: got {addr!r}"

    def test_parcelstate(self, rockbridge_result):
        data, _ = rockbridge_result
        state = str(data.get("parcelstate", "")).upper()
        assert state in ("VA", "VIRGINIA"), f"parcelstate: got {state!r}"

    def test_yearbuilt(self, rockbridge_result):
        data, _ = rockbridge_result
        assert data.get("yearbuilt") == 2007, (
            f"yearbuilt: got {data.get('yearbuilt')!r}"
        )

    def test_bedrooms(self, rockbridge_result):
        data, _ = rockbridge_result
        assert data.get("bedrooms") == 4, f"bedrooms: got {data.get('bedrooms')!r}"

    def test_fullbaths(self, rockbridge_result):
        data, _ = rockbridge_result
        assert data.get("fullbaths") == 3, f"fullbaths: got {data.get('fullbaths')!r}"

    def test_bldgsqft(self, rockbridge_result):
        """Card shows 'Building 2515.0 @ 86.40' — 2515 sqft is the base section size."""
        data, _ = rockbridge_result
        assert data.get("bldgsqft") == 2515, (
            f"bldgsqft: got {data.get('bldgsqft')!r}"
        )

    def test_taxacres(self, rockbridge_result):
        data, _ = rockbridge_result
        acres = data.get("taxacres")
        assert acres is not None and abs(acres - 0.614) < 0.01, (
            f"taxacres: got {acres!r}"
        )

    def test_landvalue(self, rockbridge_result):
        """Most-recent reassessment (2023): land=$100,000."""
        data, _ = rockbridge_result
        assert data.get("landvalue") == 100000, (
            f"landvalue: got {data.get('landvalue')!r}"
        )

    def test_imprvalue(self, rockbridge_result):
        data, _ = rockbridge_result
        assert data.get("imprvalue") == 387700, (
            f"imprvalue: got {data.get('imprvalue')!r}"
        )

    def test_totalvalue(self, rockbridge_result):
        data, _ = rockbridge_result
        assert data.get("totalvalue") == 487700, (
            f"totalvalue: got {data.get('totalvalue')!r}"
        )

    def test_value_consistency(self, rockbridge_result):
        data, _ = rockbridge_result
        land = data.get("landvalue")
        impr = data.get("imprvalue")
        total = data.get("totalvalue")
        if isinstance(land, int) and isinstance(impr, int) and isinstance(total, int):
            assert land + impr == total, (
                f"land({land}) + impr({impr}) = {land + impr} != total({total})"
            )

    def test_heating_heat_pump(self, rockbridge_result):
        """Card explicitly says 'Heat Type: HEAT PUMP'."""
        data, _ = rockbridge_result
        heating = str(data.get("heating", "")).upper()
        assert "HEAT PUMP" in heating, (
            f"heating: expected 'HEAT PUMP', got {heating!r}"
        )

    def test_heatfuel_electric(self, rockbridge_result):
        """Card explicitly says 'Fuel: ELECTRIC'."""
        data, _ = rockbridge_result
        fuel = str(data.get("heatfuel", "")).upper()
        assert fuel == "ELECTRIC", f"heatfuel: expected ELECTRIC, got {fuel!r}"

    def test_cooling_present(self, rockbridge_result):
        """Card shows 'Central A/C: Y' with $7,545 A/C Value — central air present.
        With heating=HEAT PUMP the same unit cools, so HEAT PUMP is also acceptable."""
        data, _ = rockbridge_result
        cooling = str(data.get("cooling", "")).upper()
        assert cooling, f"cooling: expected non-empty, got {cooling!r}"
        assert "HEAT PUMP" in cooling or "CENTRAL" in cooling, (
            f"cooling: expected HEAT PUMP or CENTRAL AIR, got {cooling!r}"
        )

    def test_extwall_brick(self, rockbridge_result):
        data, _ = rockbridge_result
        ext = str(data.get("extwall", "")).upper()
        assert "BRICK" in ext, f"extwall: got {ext!r}"

    def test_roofstyle_hip(self, rockbridge_result):
        data, _ = rockbridge_result
        rs = str(data.get("roofstyle", "")).upper()
        assert "HIP" in rs, f"roofstyle: got {rs!r}"

    def test_zoningcode(self, rockbridge_result):
        data, _ = rockbridge_result
        zoning = str(data.get("zoningcode", "")).upper()
        assert "R-1" in zoning or "R1" in zoning, f"zoningcode: got {zoning!r}"

    def test_saleamt(self, rockbridge_result):
        """Most recent sale 1/31/2024: $585,000."""
        data, _ = rockbridge_result
        assert data.get("saleamt") == 585000, (
            f"saleamt: got {data.get('saleamt')!r}"
        )

    def test_saledate(self, rockbridge_result):
        data, _ = rockbridge_result
        sd = str(data.get("saledate", ""))
        assert "2024-01-31" in sd, f"saledate: got {sd!r}"

    def test_print_data(self, rockbridge_result, capsys):
        data, photo = rockbridge_result
        with capsys.disabled():
            print("\n=== Rockbridge VA record 18349 ===")
            print(json.dumps(data, indent=2, default=str))
            print(f"Photo: {'yes (' + str(len(photo)) + ' bytes)' if photo else 'none'}")
