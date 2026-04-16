"""Integration test for the land records card reader pipeline.

Requires:
- Ollama running at OLLAMA_BASE_URL with the configured extraction model
- Network access to download the example PDF

Run:
    pytest test_integration.py -v -s
"""

import json
import logging

import pytest

from .nodes import download_pdf, extract_data, extract_pdf_text

EXAMPLE_URL = "https://henrycova.interactivegis.com/resources/landcards_2026/009940000.pdf"

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def initial_state():
    return {
        "pdf_url": EXAMPLE_URL,
        "pdf_bytes": None,
        "pdf_content": b"",
        "pdf_markdown": "",
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
    def test_returns_non_empty_markdown(self, text_state):
        assert len(text_state["pdf_markdown"]) > 100, (
            f"PDF text extraction should return substantial markdown, "
            f"got {len(text_state['pdf_markdown'])} chars"
        )

    def test_contains_expected_keywords(self, text_state):
        text = text_state["pdf_markdown"].lower()
        keywords = ["parcel", "owner", "acre", "value", "land", "year", "tax"]
        found = [kw for kw in keywords if kw in text]
        assert len(found) >= 3, (
            f"Expected at least 3 property keywords in PDF text, found: {found}"
        )


class TestExtractData:
    def test_returns_dict(self, extracted_state):
        data = extracted_state["property_data"]
        assert isinstance(data, dict)

    def test_print_extracted_data(self, extracted_state):
        """Not a real assertion — prints extracted data for manual review."""
        print("\n=== Extracted Property Data ===")
        print(json.dumps(extracted_state["property_data"], indent=2, default=str))
