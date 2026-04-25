"""Integration tests for Spotsylvania County HTML property cards.

Requires:
- Network access to fetch the HTML property card
- wkhtmltopdf installed (system dependency for pdfkit)
- Ollama running with the configured CARD_READER_EXTRACTION_MODEL

Run:
    pytest src/tests/test_spotsylvania.py -v -s
"""

from __future__ import annotations

import json
import logging

import httpx
import pytest

from ..config import CARD_READER_OLLAMA_HOST

logger = logging.getLogger(__name__)

CRZ_URL = "https://apps.spotsylvaniacountyva.gov/assessment/assessment/Info.cfm?OID=A000000CRZ"
PV_URL = "https://apps.spotsylvaniacountyva.gov/assessment/assessment/Info.cfm?OID=A0000010PV"


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
def crz_result():
    from .. import read_property_card

    data, photo = read_property_card(CRZ_URL)
    logger.info("CRZ extracted:\n%s", json.dumps(data, indent=2, default=str))
    return data, photo


@pytest.fixture(scope="module")
def pv_result():
    from .. import read_property_card

    data, photo = read_property_card(PV_URL)
    logger.info("0PV extracted:\n%s", json.dumps(data, indent=2, default=str))
    return data, photo


class TestSpotsylvaniaCRZ:
    """Card OID=A000000CRZ — 11206 Silversmith LN, direct-vented gas heat."""

    def test_parcelid(self, crz_result):
        data, _ = crz_result
        pid = str(data.get("parcelid", "")).upper()
        assert "22T15" in pid and "84" in pid, f"parcelid: got {pid!r}"

    def test_ownername(self, crz_result):
        data, _ = crz_result
        owner = str(data.get("ownername", "")).upper()
        assert "WRIGHT" in owner, f"ownername: got {owner!r}"

    def test_parceladdr(self, crz_result):
        data, _ = crz_result
        addr = str(data.get("parceladdr", "")).upper()
        assert "SILVERSMITH" in addr, f"parceladdr: got {addr!r}"

    def test_parcelstate(self, crz_result):
        data, _ = crz_result
        state = str(data.get("parcelstate", "")).upper()
        assert state in ("VA", "VIRGINIA"), f"parcelstate: got {state!r}"

    def test_yearbuilt(self, crz_result):
        data, _ = crz_result
        assert data.get("yearbuilt") == 2001, f"yearbuilt: got {data.get('yearbuilt')!r}"

    def test_bedrooms(self, crz_result):
        data, _ = crz_result
        assert data.get("bedrooms") == 2, f"bedrooms: got {data.get('bedrooms')!r}"

    def test_fullbaths(self, crz_result):
        data, _ = crz_result
        assert data.get("fullbaths") == 2, f"fullbaths: got {data.get('fullbaths')!r}"

    def test_bldgsqft(self, crz_result):
        data, _ = crz_result
        assert data.get("bldgsqft") == 1857, f"bldgsqft: got {data.get('bldgsqft')!r}"

    def test_landvalue(self, crz_result):
        data, _ = crz_result
        assert data.get("landvalue") == 115000, f"landvalue: got {data.get('landvalue')!r}"

    def test_totalvalue(self, crz_result):
        data, _ = crz_result
        assert data.get("totalvalue") == 308300, f"totalvalue: got {data.get('totalvalue')!r}"

    def test_value_consistency(self, crz_result):
        data, _ = crz_result
        land = data.get("landvalue")
        impr = data.get("imprvalue")
        total = data.get("totalvalue")
        if isinstance(land, int) and isinstance(impr, int) and isinstance(total, int):
            assert land + impr == total, (
                f"land({land}) + impr({impr}) = {land + impr} != total({total})"
            )

    def test_heatfuel_gas(self, crz_result):
        data, _ = crz_result
        fuel = str(data.get("heatfuel", "")).upper()
        assert fuel == "GAS", f"heatfuel: expected GAS, got {fuel!r}"

    def test_saledate(self, crz_result):
        data, _ = crz_result
        sd = str(data.get("saledate", ""))
        assert "2017-05-22" in sd, f"saledate: got {sd!r}"

    def test_saleamt(self, crz_result):
        data, _ = crz_result
        assert data.get("saleamt") == 221000, f"saleamt: got {data.get('saleamt')!r}"

    def test_print_data(self, crz_result, capsys):
        data, photo = crz_result
        with capsys.disabled():
            print("\n=== Spotsylvania CRZ ===")
            print(json.dumps(data, indent=2, default=str))
            print(f"Photo: {'yes (' + str(len(photo)) + ' bytes)' if photo else 'none'}")


class TestSpotsylvania0PV:
    """Card OID=A0000010PV — 5613 Smith Station RD, heat pump."""

    def test_parcelid(self, pv_result):
        data, _ = pv_result
        pid = str(data.get("parcelid", "")).upper()
        assert "49-7-4" in pid, f"parcelid: got {pid!r}"

    def test_ownername(self, pv_result):
        data, _ = pv_result
        owner = str(data.get("ownername", "")).upper()
        assert "HAMLETT" in owner, f"ownername: got {owner!r}"

    def test_parceladdr(self, pv_result):
        data, _ = pv_result
        addr = str(data.get("parceladdr", "")).upper()
        assert "SMITH STATION" in addr, f"parceladdr: got {addr!r}"

    def test_parcelstate(self, pv_result):
        data, _ = pv_result
        state = str(data.get("parcelstate", "")).upper()
        assert state in ("VA", "VIRGINIA"), f"parcelstate: got {state!r}"

    def test_yearbuilt(self, pv_result):
        data, _ = pv_result
        assert data.get("yearbuilt") == 1985, f"yearbuilt: got {data.get('yearbuilt')!r}"

    def test_bedrooms(self, pv_result):
        data, _ = pv_result
        assert data.get("bedrooms") == 3, f"bedrooms: got {data.get('bedrooms')!r}"

    def test_fullbaths(self, pv_result):
        data, _ = pv_result
        assert data.get("fullbaths") == 2, f"fullbaths: got {data.get('fullbaths')!r}"

    def test_bldgsqft(self, pv_result):
        data, _ = pv_result
        assert data.get("bldgsqft") == 1662, f"bldgsqft: got {data.get('bldgsqft')!r}"

    def test_taxacres(self, pv_result):
        data, _ = pv_result
        acres = data.get("taxacres")
        assert acres is not None and abs(acres - 5.20) < 0.01, f"taxacres: got {acres!r}"

    def test_landvalue(self, pv_result):
        data, _ = pv_result
        assert data.get("landvalue") == 157200, f"landvalue: got {data.get('landvalue')!r}"

    def test_totalvalue(self, pv_result):
        data, _ = pv_result
        assert data.get("totalvalue") == 371900, f"totalvalue: got {data.get('totalvalue')!r}"

    def test_value_consistency(self, pv_result):
        data, _ = pv_result
        land = data.get("landvalue")
        impr = data.get("imprvalue")
        total = data.get("totalvalue")
        if isinstance(land, int) and isinstance(impr, int) and isinstance(total, int):
            assert land + impr == total, (
                f"land({land}) + impr({impr}) = {land + impr} != total({total})"
            )

    def test_fireplaces(self, pv_result):
        data, _ = pv_result
        assert data.get("fireplaces") == 1, f"fireplaces: got {data.get('fireplaces')!r}"

    def test_saledate(self, pv_result):
        data, _ = pv_result
        sd = str(data.get("saledate", ""))
        assert "2009-08-04" in sd, f"saledate: got {sd!r}"

    def test_print_data(self, pv_result, capsys):
        data, photo = pv_result
        with capsys.disabled():
            print("\n=== Spotsylvania 0PV ===")
            print(json.dumps(data, indent=2, default=str))
            print(f"Photo: {'yes (' + str(len(photo)) + ' bytes)' if photo else 'none'}")
