"""Integration test for the Charles City County (VA) ArcGIS property card.

Card: 010211 Dalmation Drive — Whittaker family, 5.12 acres, oil heat,
"HEAT CTRL" / "AIR COND" labels (Brunswick-style central HVAC).

Requires:
- Network access to fetch the PDF from the ArcGIS attachments endpoint
- Ollama running with the configured CARD_READER_EXTRACTION_MODEL

Run:
    pytest src/tests/test_charles_city_va.py -v -s
"""

from __future__ import annotations

import json
import logging

import httpx
import pytest

from ..config import CARD_READER_OLLAMA_HOST

logger = logging.getLogger(__name__)

CHARLES_CITY_URL = (
    "https://services5.arcgis.com/7ELQsoO4nWXrJNri/arcgis/rest/services/"
    "Parcels_with_Sales_Valuation_and_Reports/FeatureServer/0/3380/attachments/3356"
)


def _ollama_reachable() -> bool:
    try:
        with httpx.Client(timeout=3.0) as client:
            r = client.get(f"{CARD_READER_OLLAMA_HOST}/api/tags")
            return r.status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _ollama_reachable(), reason="Ollama not reachable")


@pytest.fixture(scope="module")
def charles_city_result():
    from .. import read_property_card

    data, photo = read_property_card(CHARLES_CITY_URL)
    logger.info(
        "Charles City extracted:\n%s",
        json.dumps(data, indent=2, default=str),
    )
    return data, photo


class TestCharlesCityVa:
    """Card 000003477-001 — 010211 Dalmation Drive, Whittaker, oil heat."""

    def test_ownername(self, charles_city_result):
        data, _ = charles_city_result
        owner = str(data.get("ownername", "")).upper()
        assert "WHITTAKER" in owner, f"ownername: got {owner!r}"

    def test_parceladdr(self, charles_city_result):
        data, _ = charles_city_result
        addr = str(data.get("parceladdr", "")).upper()
        assert "DALMATION" in addr, f"parceladdr: got {addr!r}"

    def test_parcelcity(self, charles_city_result):
        data, _ = charles_city_result
        city = str(data.get("parcelcity", "")).upper()
        assert "CHARLES CITY" in city, f"parcelcity: got {city!r}"

    def test_parcelstate(self, charles_city_result):
        data, _ = charles_city_result
        state = str(data.get("parcelstate", "")).upper()
        assert state in ("VA", "VIRGINIA"), f"parcelstate: got {state!r}"

    def test_yearbuilt(self, charles_city_result):
        data, _ = charles_city_result
        assert data.get("yearbuilt") == 1962, (
            f"yearbuilt: expected 1962 (NOT effective 1980), got {data.get('yearbuilt')!r}"
        )

    def test_bedrooms(self, charles_city_result):
        data, _ = charles_city_result
        assert data.get("bedrooms") == 3, f"bedrooms: got {data.get('bedrooms')!r}"

    def test_fullbaths(self, charles_city_result):
        data, _ = charles_city_result
        assert data.get("fullbaths") == 2, f"fullbaths: got {data.get('fullbaths')!r}"

    def test_fireplaces(self, charles_city_result):
        data, _ = charles_city_result
        assert data.get("fireplaces") == 1, f"fireplaces: got {data.get('fireplaces')!r}"

    def test_bldgsqft(self, charles_city_result):
        data, _ = charles_city_result
        assert data.get("bldgsqft") == 1722, f"bldgsqft: got {data.get('bldgsqft')!r}"

    def test_taxacres(self, charles_city_result):
        data, _ = charles_city_result
        acres = data.get("taxacres")
        assert acres is not None and abs(acres - 5.12) < 0.01, (
            f"taxacres: got {acres!r}"
        )

    def test_landvalue(self, charles_city_result):
        data, _ = charles_city_result
        assert data.get("landvalue") == 221300, (
            f"landvalue: got {data.get('landvalue')!r}"
        )

    def test_imprvalue(self, charles_city_result):
        data, _ = charles_city_result
        assert data.get("imprvalue") == 245900, (
            f"imprvalue: got {data.get('imprvalue')!r}"
        )

    def test_totalvalue(self, charles_city_result):
        data, _ = charles_city_result
        assert data.get("totalvalue") == 467200, (
            f"totalvalue: got {data.get('totalvalue')!r}"
        )

    def test_value_consistency(self, charles_city_result):
        data, _ = charles_city_result
        land = data.get("landvalue")
        impr = data.get("imprvalue")
        total = data.get("totalvalue")
        if isinstance(land, int) and isinstance(impr, int) and isinstance(total, int):
            assert land + impr == total, (
                f"land({land}) + impr({impr}) = {land + impr} != total({total})"
            )

    def test_heatfuel_oil(self, charles_city_result):
        """Card shows 'FUEL-OIL' — heatfuel must be OIL."""
        data, _ = charles_city_result
        fuel = str(data.get("heatfuel", "")).upper()
        assert fuel == "OIL", f"heatfuel: expected OIL, got {fuel!r}"

    def test_heating_central(self, charles_city_result):
        """Card shows 'HEAT CTRL' — heating must resolve to CENTRAL."""
        data, _ = charles_city_result
        heating = str(data.get("heating", "")).upper()
        assert "CENTRAL" in heating, f"heating: expected to contain 'CENTRAL', got {heating!r}"

    def test_cooling_central(self, charles_city_result):
        """Card shows 'AIR COND' for 1722 sqft — cooling must be CENTRAL AIR."""
        data, _ = charles_city_result
        cooling = str(data.get("cooling", "")).upper()
        assert "CENTRAL" in cooling, f"cooling: expected to contain 'CENTRAL', got {cooling!r}"

    def test_zoningcode_agricultural(self, charles_city_result):
        data, _ = charles_city_result
        zoning = str(data.get("zoningcode", "")).upper()
        assert "AGRICULT" in zoning, f"zoningcode: expected AGRICULTURAL, got {zoning!r}"

    def test_saledate(self, charles_city_result):
        data, _ = charles_city_result
        sd = str(data.get("saledate", ""))
        assert "2002-10-16" in sd, f"saledate: got {sd!r}"

    def test_saleamt(self, charles_city_result):
        data, _ = charles_city_result
        assert data.get("saleamt") == 189500, f"saleamt: got {data.get('saleamt')!r}"

    def test_print_data(self, charles_city_result, capsys):
        data, photo = charles_city_result
        with capsys.disabled():
            print("\n=== Charles City County 000003477-001 ===")
            print(json.dumps(data, indent=2, default=str))
            print(f"Photo: {'yes (' + str(len(photo)) + ' bytes)' if photo else 'none'}")
