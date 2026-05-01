"""Integration test for the Radford City, VA property card pipeline.

Card: 544 West Rock Road, parcel 16-(1)-76. Used to verify the heating-type
extraction handles a "Primary Heat: Heat pump" label on a Radford-style
PDF card.

Requires:
- Network access to fetch the PDF
- Ollama running with the configured CARD_READER_EXTRACTION_MODEL

Run:
    pytest src/tests/test_radford.py -v -s
"""

from __future__ import annotations

import json
import logging

import httpx
import pytest

from ..config import CARD_READER_OLLAMA_HOST

logger = logging.getLogger(__name__)

RADFORD_URL = "https://radfordgis3.radford.va.us/propertycards/020003057.pdf"


def _ollama_reachable() -> bool:
    try:
        with httpx.Client(timeout=3.0) as client:
            r = client.get(f"{CARD_READER_OLLAMA_HOST}/api/tags")
            return r.status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _ollama_reachable(), reason="Ollama not reachable")


@pytest.fixture(scope="module")
def radford_result():
    from .. import read_property_card

    data, photo = read_property_card(RADFORD_URL)
    logger.info("Radford extracted:\n%s", json.dumps(data, indent=2, default=str))
    return data, photo


class TestRadfordHeatPump:
    """Card Tax ID 020003057 — heating system label is 'Primary Heat: Heat pump'."""

    def test_parcelid(self, radford_result):
        data, _ = radford_result
        pid = str(data.get("parcelid", "")).upper()
        assert "16" in pid and "76" in pid, f"parcelid: got {pid!r}"

    def test_ownername(self, radford_result):
        data, _ = radford_result
        owner = str(data.get("ownername", "")).upper()
        assert "TURNER" in owner, f"ownername: got {owner!r}"

    def test_parceladdr(self, radford_result):
        data, _ = radford_result
        addr = str(data.get("parceladdr", "")).upper()
        assert "ROCK" in addr, f"parceladdr: got {addr!r}"

    def test_heating_is_heat_pump(self, radford_result):
        """The card explicitly says 'Primary Heat: Heat pump' — the heating
        field must reflect that, not be omitted or set to a fuel value."""
        data, _ = radford_result
        heating = str(data.get("heating", "")).upper()
        assert "HEAT PUMP" in heating, (
            f"heating: expected 'HEAT PUMP', got {heating!r}"
        )

    def test_cooling_is_present(self, radford_result):
        """The card shows a nonzero 'Air Cond' row (1860 sqft on the main
        floor) and 'Air Condition 7440' in the improvement summary — AC is
        clearly present, so cooling must not be omitted. With heat=HEAT PUMP
        the same unit cools, so the expected value is HEAT PUMP (CENTRAL AIR
        is also acceptable for cards without a heat pump)."""
        data, _ = radford_result
        cooling = str(data.get("cooling", "")).upper()
        assert cooling, f"cooling: expected non-empty value, got {cooling!r}"
        assert "HEAT PUMP" in cooling or "CENTRAL" in cooling, (
            f"cooling: expected HEAT PUMP or CENTRAL AIR, got {cooling!r}"
        )

    def test_print_data(self, radford_result, capsys):
        data, photo = radford_result
        with capsys.disabled():
            print("\n=== Radford 020003057 ===")
            print(json.dumps(data, indent=2, default=str))
            print(f"Photo: {'yes (' + str(len(photo)) + ' bytes)' if photo else 'none'}")
