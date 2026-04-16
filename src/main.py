#!/usr/bin/env python3
"""CLI entry point for the land records card reader agent."""

import argparse
import json
import logging

from .graph import app


def main():
    parser = argparse.ArgumentParser(
        description="Extract property data from a property card PDF."
    )
    parser.add_argument("url", help="URL to a property card PDF")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run extraction and print results to stdout",
    )
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

    initial_state = {
        "pdf_url": args.url,
        "pdf_bytes": None,
        "pdf_content": b"",
        "pdf_markdown": "",
        "property_photos": [],
        "property_data": {},
        "result": "",
    }

    if args.dry_run:
        # Run the full extraction path and print results.
        from .nodes import (
            download_pdf,
            extract_data,
            extract_pdf_and_photos,
        )

        state = {**initial_state}
        state.update(download_pdf(state))
        state.update(extract_pdf_and_photos(state))
        state.update(extract_data(state))

        print("\n=== PDF Markdown ===")
        print(state["pdf_markdown"])

        print(f"\n=== Property Photos ({len(state['property_photos'])}) ===")
        for i, p in enumerate(state["property_photos"]):
            print(f"  [{i}] page={p['page']} {p['width']}x{p['height']} ext={p['ext']} ({len(p['bytes'])} bytes)")

        print("\n=== Extracted Property Data ===")
        print(json.dumps(state["property_data"], indent=2, default=str))
    else:
        result = app.invoke(initial_state)
        print(result["result"])


if __name__ == "__main__":
    main()
