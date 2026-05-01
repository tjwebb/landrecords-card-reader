"""Integration test against a Henry County (VA) interactiveGIS land card.

Requires:
- Ollama running at CARD_READER_OLLAMA_HOST with the configured extraction model
- Network access to download the example PDF

Run:
    pytest src/tests/test_henry_va_interactivegis.py -v -s
"""

import json
import logging

import pytest

from ..nodes import download_pdf, extract_data, extract_pdf_text

EXAMPLE_URL = "https://henrycova.interactivegis.com/resources/landcards_2026/075740001.pdf"


logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def initial_state():
    return {
        "pdf_url": EXAMPLE_URL,
        "pdf_bytes": None,
        "pdf_content": b"",
        "pdf_text": "",
        "property_photos": [],
        "property_data": {},
        "result": "",
    }


@pytest.fixture(scope="module")
def pdf_state(initial_state):
    """Download the PDF."""
    state = {**initial_state}
    state.update(download_pdf(state))
    return state


@pytest.fixture(scope="module")
def text_state(pdf_state):
    """Extract embedded text as markdown."""
    state = {**pdf_state}
    state.update(extract_pdf_text(state))
    return state


@pytest.fixture(scope="module")
def extracted_state(text_state):
    """Extract structured data from PDF markdown."""
    state = {**text_state}
    state.update(extract_data(state))
    return state


class TestDownloadPdf:
    def test_downloads_content(self, pdf_state):
        assert len(pdf_state["pdf_content"]) > 0, "Should download PDF bytes"


class TestExtractPdfText:
    def test_returns_non_empty_text(self, text_state):
        assert len(text_state["pdf_text"]) > 100, (
            f"PDF text extraction should return substantial text, "
            f"got {len(text_state['pdf_text'])} chars"
        )

    def test_contains_expected_keywords(self, text_state):
        text = text_state["pdf_text"].lower()
        keywords = ["parcel", "owner", "acre", "value", "land", "year", "tax"]
        found = [kw for kw in keywords if kw in text]
        assert len(found) >= 3, (
            f"Expected at least 3 property keywords in PDF text, found: {found}"
        )


class TestExtractData:
    def test_returns_dict(self, extracted_state):
        data = extracted_state["property_data"]
        assert isinstance(data, dict)

    def test_heatfuel_electric(self, extracted_state):
        """Card shows 'Heat Fuel ELECT' — heatfuel must be ELECTRIC."""
        data = extracted_state["property_data"]
        fuel = str(data.get("heatfuel", "")).upper()
        assert fuel == "ELECTRIC", f"heatfuel: expected ELECTRIC, got {fuel!r}"

    def test_heating_central_air(self, extracted_state):
        """Henry InteractiveGIS cards have no separate heating-delivery
        label — the only HVAC indicators are 'Heat Fuel' + 'Central Air %'.
        Both heating and cooling must resolve to CENTRAL AIR."""
        data = extracted_state["property_data"]
        heating = str(data.get("heating", "")).upper()
        assert "CENTRAL" in heating, (
            f"heating: expected to contain 'CENTRAL', got {heating!r}"
        )

    def test_cooling_central_air(self, extracted_state):
        """Card shows 'Central Air % 100' — cooling must be CENTRAL AIR."""
        data = extracted_state["property_data"]
        cooling = str(data.get("cooling", "")).upper()
        assert "CENTRAL" in cooling, (
            f"cooling: expected to contain 'CENTRAL', got {cooling!r}"
        )

    def test_print_extracted_data(self, extracted_state):
        """Not a real assertion — prints extracted data for manual review."""
        print("\n=== Extracted Property Data ===")
        print(json.dumps(extracted_state["property_data"], indent=2, default=str))
