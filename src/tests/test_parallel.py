"""Concurrency stress test: process 10 property cards in parallel.

Exists to verify the pipeline is safe to call from multiple threads in
the same Python process — specifically that

  * pypdfium2 page rendering (now isolated in a subprocess via
    ``_render_pages_with_pdfium``) doesn't race when several callers
    request it simultaneously;
  * pymupdf text extraction and photo discovery survive concurrent use
    from multiple threads;
  * the Ollama photo classifier (called via ThreadPoolExecutor inside
    ``extract_property_photos``) holds up when many requests overlap;
  * the Ollama text-extraction LLM call serialises gracefully under
    concurrent invocation.

The cards are local fixture PDFs so the test isn't gated on network
availability for the cards themselves. It still requires Ollama,
since extraction calls an LLM, and ``wkhtmltopdf`` is not needed.

Run:
    pytest src/tests/test_parallel.py -v -s
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import pytest

from ..config import CARD_READER_OLLAMA_HOST

logger = logging.getLogger(__name__)

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# 10 fixture PDFs — a deliberate mix of sources (Henry NC/VA, Wythe VA,
# Patrick VA, Russell VA, Carroll VA, Manassas VA, Williamsburg VA,
# Travis TX) so the concurrency stress isn't dominated by one layout.
PARALLEL_FIXTURES = [
    "carroll_va_0000028365.pdf",
    "henry_va_059800000.pdf",
    "henry_va_116650000.pdf",
    "henry_va_139290002.pdf",
    "henry_va_173760004.pdf",
    "henry_va_216430002.pdf",
    "patrick_va_5012120A.pdf",
    "wythe_va_4328.pdf",
    "wythe_va_13436.pdf",
    "travis_tx_372462.pdf",
]


def _ollama_reachable() -> bool:
    try:
        with httpx.Client(timeout=3.0) as client:
            r = client.get(f"{CARD_READER_OLLAMA_HOST}/api/tags")
            return r.status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _ollama_reachable(),
    reason=f"Ollama not reachable at {CARD_READER_OLLAMA_HOST}",
)


def _process_one(pdf_name: str) -> tuple[str, dict, bytes | None, float]:
    """Run the full pipeline against one fixture and return its result.

    Imported lazily so each thread re-uses the cached module rather than
    re-importing. Returns (pdf_name, data, photo_bytes, elapsed_seconds).
    """
    from .. import read_property_card

    pdf_path = FIXTURES_DIR / pdf_name
    pdf_bytes = pdf_path.read_bytes()

    t0 = time.perf_counter()
    data, photo = read_property_card(
        f"file://{pdf_path}",
        pdf_bytes=pdf_bytes,
    )
    elapsed = time.perf_counter() - t0
    return pdf_name, data, photo, elapsed


class TestParallel:
    def test_ten_cards_in_parallel(self, capsys):
        """Fan out 10 card extractions across 10 threads, assert all succeed."""
        # Pre-warm the docTR model in the parent process. Otherwise the
        # first thread would pay the model-load tax (and possibly serialize
        # other threads waiting on the lazy initializer's GIL section), which
        # contaminates the parallelism timing measurement.
        from .. import warm_ocr_cache
        warm_ocr_cache()

        n = len(PARALLEL_FIXTURES)
        results: dict[str, tuple[dict, bytes | None, float]] = {}
        errors: dict[str, BaseException] = {}

        wall_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n) as pool:
            futures = {
                pool.submit(_process_one, name): name
                for name in PARALLEL_FIXTURES
            }
            for fut in as_completed(futures):
                name = futures[fut]
                try:
                    pdf_name, data, photo, elapsed = fut.result()
                    results[pdf_name] = (data, photo, elapsed)
                    logger.info(
                        "Parallel extraction OK: %s (%.1fs)", pdf_name, elapsed,
                    )
                except BaseException as e:
                    errors[name] = e
                    logger.exception("Parallel extraction failed: %s", name)
        wall_elapsed = time.perf_counter() - wall_start

        # All 10 must complete without exceptions / segfaults / deadlocks.
        assert not errors, (
            f"{len(errors)}/{n} parallel extractions failed:\n"
            + "\n".join(f"  - {name}: {type(e).__name__}: {e}"
                        for name, e in errors.items())
        )
        assert len(results) == n, (
            f"expected {n} results, got {len(results)}"
        )

        # Each result must be a dict with at least one populated field —
        # if extraction silently degraded under concurrency we'd see empty
        # dicts here.
        for pdf_name, (data, _photo, _elapsed) in results.items():
            assert isinstance(data, dict), (
                f"{pdf_name}: expected dict result, got {type(data).__name__}"
            )
            assert len(data) > 0, (
                f"{pdf_name}: extracted dict was empty (concurrency may "
                f"have corrupted state)"
            )

        # Sanity check: the wall-clock time should be less than the sum of
        # per-card times. If parallelism is silently broken (e.g. a global
        # lock serialising everything), wall time would equal the sum.
        # Threshold is loose because Ollama may serialize GPU work
        # internally — we just want to see *some* overlap.
        sum_elapsed = sum(t for _, _, t in results.values())
        with capsys.disabled():
            print(
                f"\n=== Parallel extraction summary ===\n"
                f"  cards:       {n}\n"
                f"  wall time:   {wall_elapsed:.1f}s\n"
                f"  sum of times: {sum_elapsed:.1f}s\n"
                f"  speedup:     {sum_elapsed / wall_elapsed:.2f}x\n"
            )
        assert wall_elapsed < sum_elapsed, (
            f"No parallelism detected: wall={wall_elapsed:.1f}s "
            f">= sum={sum_elapsed:.1f}s. Pipeline may be serialising "
            "internally."
        )
