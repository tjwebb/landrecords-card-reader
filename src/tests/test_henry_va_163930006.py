"""Integration test for Henry County VA mobile-home card 163930006.

Card: 220 Airport Ridge Dr, Willey — mobile home on 1.176 ac with
NO HVAC installed (Heat Fuel cell empty, Central Air % 0).

Regression target: bare 2-letter codes that happen to appear elsewhere
on the card (e.g. "HP" in the district/class section) must NOT be
expanded to "HEAT PUMP" — heating should be omitted entirely when no
heating-delivery value is adjacent to a heating-system label.

Requires:
- Network access to fetch the PDF
- Ollama running with the configured CARD_READER_EXTRACTION_MODEL

Run:
    pytest src/tests/test_henry_va_163930006.py -v -s
"""

from __future__ import annotations

import json
import logging

import httpx
import pytest

from ..config import CARD_READER_OLLAMA_HOST

logger = logging.getLogger(__name__)

CARD_URL = (
    "https://henrycova.interactivegis.com/resources/landcards_2026/163930006.pdf"
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
        "163930006 extracted:\n%s",
        json.dumps(data, indent=2, default=str),
    )
    return data, photo


class TestHenryMobileHome:
    """Card 163930006 — mobile home, no HVAC."""

    def test_ownername(self, card_result):
        data, _ = card_result
        owner = str(data.get("ownername", "")).upper()
        assert "WILLEY" in owner, f"ownername: got {owner!r}"

    def test_parceladdr(self, card_result):
        data, _ = card_result
        addr = str(data.get("parceladdr", "")).upper()
        assert "AIRPORT RIDGE" in addr, f"parceladdr: got {addr!r}"

    def test_parcelstate(self, card_result):
        data, _ = card_result
        state = str(data.get("parcelstate", "")).upper()
        assert state in ("VA", "VIRGINIA"), f"parcelstate: got {state!r}"

    def test_taxacres(self, card_result):
        data, _ = card_result
        acres = data.get("taxacres")
        assert acres is not None and abs(acres - 1.176) < 0.01, (
            f"taxacres: got {acres!r}"
        )

    def test_landvalue(self, card_result):
        data, _ = card_result
        assert data.get("landvalue") == 7500, (
            f"landvalue: got {data.get('landvalue')!r}"
        )

    def test_totalvalue(self, card_result):
        data, _ = card_result
        assert data.get("totalvalue") == 18000, (
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

    def test_heating_not_inferred_from_stray_hp(self, card_result):
        """Headline regression: card has NO heating system — Heat Fuel cell
        is empty, Central Air % is 0, but a stray 'HP' code appears
        elsewhere on the card (district/class section). The LLM must
        NOT expand that stray HP to HEAT PUMP. Heating must be absent
        or empty."""
        data, _ = card_result
        heating = data.get("heating")
        assert heating in (None, "", "NONE"), (
            f"heating: expected absent/empty (no heating system on card), "
            f"got {heating!r}. The stray 'HP' code on the card is a "
            f"district/class abbreviation, NOT a heat type."
        )

    def test_cooling_not_inferred_when_zero(self, card_result):
        """Card explicitly shows 'Central Air % 0' — cooling must be
        absent or NONE, never CENTRAL AIR."""
        data, _ = card_result
        cooling = data.get("cooling")
        cooling_str = str(cooling).upper() if cooling else ""
        assert "CENTRAL" not in cooling_str, (
            f"cooling: 'Central Air % 0' on the card means no AC; "
            f"got {cooling!r}"
        )

    def test_print_data(self, card_result, capsys):
        data, photo = card_result
        with capsys.disabled():
            print("\n=== Henry VA 163930006 (mobile home, no HVAC) ===")
            print(json.dumps(data, indent=2, default=str))
            print(f"Photo: {'yes (' + str(len(photo)) + ' bytes)' if photo else 'none'}")
