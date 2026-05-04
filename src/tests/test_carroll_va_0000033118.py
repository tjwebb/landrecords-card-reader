"""Integration test for Carroll County VA property card 0000033118.

Card: 101 Spring Haven Dr, Fancy Gap — single-family conventional home
on 3.323 ac waterfront lot. 2011 build, Heat pump w/ central A/C,
Hardiplank siding, metal gable roof, 2 bed / 4 finished rooms,
1904 sqft finished, attached 24x24 garage, fireplace + generator.

Most recent (2025) reassessment: land $46,600 + improvements $452,700
= total $499,300. Sold 2021-05-28 for $345,000 by the Woods Trust.

Requires:
- Network access to fetch the PDF
- Ollama running with the configured CARD_READER_EXTRACTION_MODEL

Run:
    pytest src/tests/test_carroll_va_0000033118.py -v -s
"""

from __future__ import annotations

import json
import logging

import httpx
import pytest

from ..config import CARD_READER_OLLAMA_HOST

logger = logging.getLogger(__name__)

CARD_URL = (
    "https://carrollcova.interactivegis.com/resources/propertycards/0000033118.pdf"
)


def _ollama_reachable() -> bool:
    try:
        with httpx.Client(timeout=3.0) as client:
            r = client.get(f"{CARD_READER_OLLAMA_HOST}/api/tags")
            return r.status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _ollama_reachable(), reason="Ollama not reachable",
)


@pytest.fixture(scope="module")
def card_result():
    from .. import read_property_card

    data, photo = read_property_card(CARD_URL)
    logger.info(
        "0000033118 extracted:\n%s",
        json.dumps(data, indent=2, default=str),
    )
    return data, photo


class TestCarrollVa0000033118:
    """101 Spring Haven Dr — Byrd/Poole, 2011 conventional, heat pump."""

    def test_ownername(self, card_result):
        data, _ = card_result
        owner = str(data.get("ownername", "")).upper()
        assert "BYRD" in owner, f"ownername: got {owner!r}"

    def test_parceladdr(self, card_result):
        data, _ = card_result
        addr = str(data.get("parceladdr", "")).upper()
        assert "SPRING HAVEN" in addr, f"parceladdr: got {addr!r}"

    def test_parcelcity(self, card_result):
        data, _ = card_result
        city = str(data.get("parcelcity", "")).upper()
        assert "FANCY GAP" in city, f"parcelcity: got {city!r}"

    def test_parcelstate(self, card_result):
        data, _ = card_result
        state = str(data.get("parcelstate", "")).upper()
        assert state in ("VA", "VIRGINIA"), f"parcelstate: got {state!r}"

    def test_parcelzip(self, card_result):
        data, _ = card_result
        zip_code = str(data.get("parcelzip", ""))
        assert "24328" in zip_code, f"parcelzip: got {zip_code!r}"

    def test_yearbuilt(self, card_result):
        """Card shows Year Const 2011 for the dwelling."""
        data, _ = card_result
        assert data.get("yearbuilt") == 2011, (
            f"yearbuilt: got {data.get('yearbuilt')!r}"
        )

    def test_bedrooms(self, card_result):
        data, _ = card_result
        assert data.get("bedrooms") == 2, f"bedrooms: got {data.get('bedrooms')!r}"

    def test_bldgsqft(self, card_result):
        """Card shows 'Finished Area: 1904'."""
        data, _ = card_result
        assert data.get("bldgsqft") == 1904, (
            f"bldgsqft: got {data.get('bldgsqft')!r}"
        )

    def test_taxacres(self, card_result):
        """Legal Acres: 3.3230 (1.0 waterfront homesite + 2.3230 excess)."""
        data, _ = card_result
        acres = data.get("taxacres")
        assert acres is not None and abs(acres - 3.323) < 0.01, (
            f"taxacres: got {acres!r}"
        )

    def test_landvalue(self, card_result):
        """2025 reassessment: land = $46,600."""
        data, _ = card_result
        assert data.get("landvalue") == 46600, (
            f"landvalue: got {data.get('landvalue')!r}"
        )

    def test_imprvalue(self, card_result):
        """2025 reassessment: improvements = $452,700."""
        data, _ = card_result
        assert data.get("imprvalue") == 452700, (
            f"imprvalue: got {data.get('imprvalue')!r}"
        )

    def test_totalvalue(self, card_result):
        """2025 reassessment: total = $499,300."""
        data, _ = card_result
        assert data.get("totalvalue") == 499300, (
            f"totalvalue: got {data.get('totalvalue')!r}"
        )

    def test_value_consistency(self, card_result):
        data, _ = card_result
        land = data.get("landvalue")
        impr = data.get("imprvalue")
        total = data.get("totalvalue")
        if isinstance(land, int) and isinstance(impr, int) and isinstance(total, int):
            assert land + impr == total, (
                f"land({land}) + impr({impr}) = {land + impr} != total({total})"
            )

    def test_heating_heat_pump(self, card_result):
        """Card explicitly says 'Primary Heat: Heat pump'."""
        data, _ = card_result
        heating = str(data.get("heating", "")).upper()
        assert "HEAT PUMP" in heating or "PUMP" in heating, (
            f"heating: expected to reflect 'Heat pump', got {heating!r}"
        )

    def test_cooling_central_air(self, card_result):
        """Card shows Air Cond 1904 sqft on the main level — full central A/C."""
        data, _ = card_result
        cooling = str(data.get("cooling", "")).upper()
        assert "CENTRAL" in cooling or "AIR" in cooling, (
            f"cooling: expected central air, got {cooling!r}"
        )

    def test_roofcover_metal(self, card_result):
        data, _ = card_result
        rc = str(data.get("roofcover", "")).upper()
        assert "METAL" in rc, f"roofcover: got {rc!r}"

    def test_roofstyle_gable(self, card_result):
        data, _ = card_result
        rs = str(data.get("roofstyle", "")).upper()
        assert "GABLE" in rs, f"roofstyle: got {rs!r}"

    def test_extwall_hardiplank(self, card_result):
        """Exterior Cover: Hardiplank Siding."""
        data, _ = card_result
        ext = str(data.get("extwall", "")).upper()
        assert "HARDI" in ext or "FIBER" in ext or "SIDING" in ext, (
            f"extwall: got {ext!r}"
        )

    def test_print_data(self, card_result, capsys):
        data, photo = card_result
        with capsys.disabled():
            print("\n=== Carroll VA 0000033118 (101 Spring Haven Dr) ===")
            print(json.dumps(data, indent=2, default=str))
            print(f"Photo: {'yes (' + str(len(photo)) + ' bytes)' if photo else 'none'}")
