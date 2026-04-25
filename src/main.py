#!/usr/bin/env python3
"""CLI entry point for the land records card reader agent."""

import argparse
import json
import logging

from .nodes import download_pdf, extract_data, extract_pdf_and_photos


def main():
    parser = argparse.ArgumentParser(
        description="Extract property data from a property card PDF."
    )
    parser.add_argument("url", help="URL to a property card PDF")
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    state = {
        "pdf_url": args.url,
        "pdf_bytes": None,
        "pdf_content": b"",
        "pdf_text": "",
        "property_photos": [],
        "property_data": {},
        "result": "",
        "context": None,
    }

    state.update(download_pdf(state))
    state.update(extract_pdf_and_photos(state))
    state.update(extract_data(state))

    print(json.dumps(state["property_data"], indent=2, default=str))


if __name__ == "__main__":
    main()
