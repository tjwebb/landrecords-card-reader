"""Land records card reader agent — callable as a Python function.

Usage:
    from landrecords_card_reader import read_property_card

    data, photo = read_property_card("https://example.com/card.pdf")
    # data  — dict of extracted property fields
    # photo — raw image bytes of the first property photo, or None

    # With photo analysis to fill missing building details:
    data, photo = read_property_card(url, analyze_photo=True)

    # From any URL (PDF or HTML property page):
    from landrecords_card_reader import read_property_card_from_url

    data, photo = read_property_card_from_url("https://beacon.schneidercorp.com/...")

    # Pre-warm the OCR model cache (avoid first-call download latency in
    # short-lived workers, e.g. Ray actors). Idempotent — re-uses an
    # already-loaded predictor on subsequent calls.
    from landrecords_card_reader import warm_ocr_cache
    warm_ocr_cache()
"""

from .state import AgentState


def warm_ocr_cache() -> None:
    """Force-load the docTR OCR model so subsequent calls skip the download.

    Useful when:
      - The hosting environment spawns short-lived Python workers (Ray,
        Lambda, Celery) and the per-process startup latency is visible.
      - You want to fail fast on first deploy if model weights can't be
        fetched (e.g. behind a strict egress firewall).

    Resolution order for weights:
      1. ``$DOCTR_CACHE_DIR`` if explicitly set
      2. The package-bundled cache (``<install>/doctr_cache``) when the
         wheel was built with ``prefetch_doctr_models()``
      3. ``~/.cache/doctr/`` (docTR's default; downloads on first use)
    """
    from .nodes import _get_ocr_model
    _get_ocr_model()


def prefetch_doctr_models() -> None:
    """Download docTR OCR weights into the package-bundled cache.

    Intended as a build-time step: run once on the publisher's machine
    before ``python -m build`` so the wheel ships with the weights and
    end users don't pay first-call download latency. Idempotent.
    """
    from .nodes import prefetch_doctr_models as _prefetch
    _prefetch()


def read_property_card(
    pdf_url: str,
    *,
    pdf_bytes: bytes | None = None,
    analyze_photo: bool = False,
    context: str | None = None,
) -> tuple[dict, bytes | None]:
    """Extract property data and the primary photo from a property card PDF.

    Args:
        pdf_url: URL to a property card PDF.
        pdf_bytes: Pre-downloaded PDF content. If provided, skips the HTTP
            download.
        analyze_photo: If True, send the first property photo to the
            extraction model to fill in any missing fields that can be
            inferred visually (building type, exterior walls, roof style,
            number of floors, visible features like pools/decks/etc.).
        context: Optional extra instructions appended to the extraction
            prompt (e.g. county-specific hints about where to find a field).

    Returns:
        ``(property_data, image_bytes)`` where *property_data* is a dict of
        extracted fields and *image_bytes* is the raw bytes of the first
        embedded property photo (or ``None`` if the card has no photos).
    """
    from .nodes import (
        download_pdf,
        extract_data,
        extract_pdf_and_photos,
        fill_from_photo,
    )

    state: dict = {
        "pdf_url": pdf_url,
        "pdf_bytes": pdf_bytes,
        "pdf_content": b"",
        "pdf_text": "",
        "property_photos": [],
        "property_data": {},
        "result": "",
        "context": context,
    }

    state.update(download_pdf(state))
    state.update(extract_pdf_and_photos(state))

    state.update(extract_data(state))

    data = state["property_data"]
    photos = state.get("property_photos", [])
    image_bytes = photos[0]["bytes"] if photos else None

    if analyze_photo and image_bytes:
        data = fill_from_photo(data, image_bytes)

    return data, image_bytes


def read_property_card_from_url(
    url: str,
    *,
    analyze_photo: bool = False,
    context: str | None = None,
) -> tuple[dict, bytes | None]:
    """Fetch a URL and extract property data, handling both PDF and HTML pages.

    If the URL returns a PDF, the bytes are passed directly to
    :func:`read_property_card`.  If it returns HTML, the page is converted
    to PDF via pdfkit (wkhtmltopdf) first.

    Args:
        url: URL to a property card PDF or HTML property report page.
        analyze_photo: If True, send the first property photo to the
            extraction model to fill in missing building details.
        context: Optional extra instructions appended to the extraction
            prompt. Forwarded to :func:`read_property_card`.

    Returns:
        ``(property_data, image_bytes)`` — same as :func:`read_property_card`.
    """
    import httpx

    from .nodes import _html_to_pdf, _is_pdf

    with httpx.Client(timeout=60, follow_redirects=True, verify=False) as client:
        resp = client.get(url)
        resp.raise_for_status()
    content = resp.content

    if not _is_pdf(content):
        content = _html_to_pdf(content, url)

    return read_property_card(
        url,
        pdf_bytes=content,
        analyze_photo=analyze_photo,
        context=context,
    )
