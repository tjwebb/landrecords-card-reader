"""Integration test for the Craig County (VA) webgis property card.

Card: 197 Meadow Brook Trail, Birtsch — 1.7-story log home on 0.866
acres in Johns Creek Acres. "Central Warm Air" heating, no A/C.

Requires:
- Network access to fetch the HTML property card
- wkhtmltopdf installed (system dependency for pdfkit)
- Ollama running with the configured CARD_READER_EXTRACTION_MODEL

Run:
    pytest src/tests/test_craig_va.py -v -s
"""

from __future__ import annotations

import json
import logging

import httpx
import pytest

from ..config import CARD_READER_OLLAMA_HOST

logger = logging.getLogger(__name__)

CRAIG_URL = "https://www.webgis.net/linkedfiles/va/craig/?lrsn=1839"


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
def craig_result():
    from .. import read_property_card

    data, photo = read_property_card(CRAIG_URL)
    logger.info("Craig extracted:\n%s", json.dumps(data, indent=2, default=str))
    return data, photo


class TestCraigVa:
    """Card 001979 — Birtsch, 197 Meadow Brook Trail, 'Central Warm Air' heat."""

    def test_parcelid(self, craig_result):
        data, _ = craig_result
        pid = str(data.get("parcelid", "")).upper()
        assert "001979" in pid or "1979" in pid, f"parcelid: got {pid!r}"

    def test_ownername(self, craig_result):
        data, _ = craig_result
        owner = str(data.get("ownername", "")).upper()
        assert "BIRTSCH" in owner, f"ownername: got {owner!r}"

    def test_parceladdr(self, craig_result):
        data, _ = craig_result
        addr = str(data.get("parceladdr", "")).upper()
        assert "MEADOW BROOK" in addr or "MEADOWBROOK" in addr, (
            f"parceladdr: got {addr!r}"
        )

    def test_parcelstate(self, craig_result):
        data, _ = craig_result
        state = str(data.get("parcelstate", "")).upper()
        assert state in ("VA", "VIRGINIA"), f"parcelstate: got {state!r}"

    def test_yearbuilt(self, craig_result):
        data, _ = craig_result
        assert data.get("yearbuilt") == 2008, (
            f"yearbuilt: got {data.get('yearbuilt')!r}"
        )

    def test_bedrooms(self, craig_result):
        data, _ = craig_result
        assert data.get("bedrooms") == 2, f"bedrooms: got {data.get('bedrooms')!r}"

    def test_bldgsqft(self, craig_result):
        """Card shows 'Finished Area: 1999'."""
        data, _ = craig_result
        assert data.get("bldgsqft") == 1999, (
            f"bldgsqft: got {data.get('bldgsqft')!r}"
        )

    def test_taxacres(self, craig_result):
        data, _ = craig_result
        acres = data.get("taxacres")
        assert acres is not None and abs(acres - 0.866) < 0.01, (
            f"taxacres: got {acres!r}"
        )

    def test_landvalue(self, craig_result):
        """Most recent assessment (2024 Reass): land=$30,000."""
        data, _ = craig_result
        assert data.get("landvalue") == 30000, (
            f"landvalue: got {data.get('landvalue')!r}"
        )

    def test_imprvalue(self, craig_result):
        data, _ = craig_result
        assert data.get("imprvalue") == 331200, (
            f"imprvalue: got {data.get('imprvalue')!r}"
        )

    def test_totalvalue(self, craig_result):
        data, _ = craig_result
        assert data.get("totalvalue") == 361200, (
            f"totalvalue: got {data.get('totalvalue')!r}"
        )

    def test_value_consistency(self, craig_result):
        data, _ = craig_result
        land = data.get("landvalue")
        impr = data.get("imprvalue")
        total = data.get("totalvalue")
        if isinstance(land, int) and isinstance(impr, int) and isinstance(total, int):
            assert land + impr == total, (
                f"land({land}) + impr({impr}) = {land + impr} != total({total})"
            )

    def test_heating_central_warm_air(self, craig_result):
        """Card explicitly says 'Primary Heat: Central Warm Air'."""
        data, _ = craig_result
        heating = str(data.get("heating", "")).upper()
        assert heating, f"heating: expected non-empty, got {heating!r}"
        # Accept any of the canonical phrasings: 'CENTRAL WARM AIR',
        # 'WARMED & COOLED AIR', 'CENTRAL', or 'FORCED AIR'.
        assert (
            "CENTRAL" in heating
            or "WARM" in heating
            or "FORCED AIR" in heating
        ), f"heating: expected to reflect 'Central Warm Air', got {heating!r}"

    def test_zoningcode(self, craig_result):
        """Card shows 'Zoning: A-1 Agricultural Ltd'."""
        data, _ = craig_result
        zoning = str(data.get("zoningcode", "")).upper()
        assert "A-1" in zoning or "A1" in zoning, f"zoningcode: got {zoning!r}"

    def test_extwall_log(self, craig_result):
        """Exterior Cover: Log solid."""
        data, _ = craig_result
        ext = str(data.get("extwall", "")).upper()
        assert "LOG" in ext, f"extwall: got {ext!r}"

    def test_roofcover_metal(self, craig_result):
        data, _ = craig_result
        rc = str(data.get("roofcover", "")).upper()
        assert "METAL" in rc, f"roofcover: got {rc!r}"

    def test_roofstyle_gable(self, craig_result):
        data, _ = craig_result
        rs = str(data.get("roofstyle", "")).upper()
        assert "GABLE" in rs, f"roofstyle: got {rs!r}"

    def test_print_data(self, craig_result, capsys):
        data, photo = craig_result
        with capsys.disabled():
            print("\n=== Craig VA 001979 ===")
            print(json.dumps(data, indent=2, default=str))
            print(f"Photo: {'yes (' + str(len(photo)) + ' bytes)' if photo else 'none'}")
