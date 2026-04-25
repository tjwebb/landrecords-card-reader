import json
import logging
import re
import sys
import warnings

import httpx

# County assessor sites often ship broken/expired certs. Silence the
# verify=False noise so it doesn't clutter logs.
warnings.filterwarnings("ignore", message="Unverified HTTPS request")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ssl")
try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except ImportError:
    pass

from .config import (
    CARD_READER_EXTRACTION_MODEL,
    CARD_READER_OLLAMA_HOST,
    CARD_READER_PHOTO_CLASSIFICATION_MODEL,
)
from .state import AgentState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# All columns in combine.reftable_enriched that we might populate for parcel
# ---------------------------------------------------------------------------
VALID_COLUMNS = {
    "parcelid", "parcelid2", "taxacctnum", "taxyear", "usecode", "usedesc",
    "zoningcode", "zoningdesc", "numbldgs", "numunits", "yearbuilt",
    "bldgsqft", "bedrooms", "halfbaths", "fullbaths", "imprvalue",
    "landvalue", "agvalue", "totalvalue", "taxacres", "saleamt", "saledate",
    "ownername", "owneraddr", "ownercity", "ownerstate", "ownerzip",
    "parceladdr", "parcelcity", "parcelstate", "parcelzip", "legaldesc",
    "book", "page", "block", "lot", "landusecode", "landusedesc",
    "house", "category", "near", "house_number", "road", "unit", "level",
    "bldgtype", "numfloors", "yearremodel", "livingarea", "bldgquality",
    "bldgcondition", "architecture", "totalrooms", "fireplaces", "heating",
    "heatfuel", "cooling", "foundation", "attic", "atticsqft", "intwall", "extwall",
    "roofstyle", "roofcover", "roofheight", "fsagpfeet", "siding", "framing",
    "basementsqft", "attgaragesqft", "detgaragesqft", "garagestalls",
    "situsaddress", "appraisedvalue", "assessedvalue",
    "boatlift", "boatdock", "boathouse", "pool", "gazebo", "irrigation",
    "riprap", "solarium", "carport", "greenhouse", "openporch", "enclporch",
    "sauna", "wooddeck", "hottub", "patio", "shed", "workshop",
    "taxdistrict",
}

# ---------------------------------------------------------------------------
# Extraction prompt template
# ---------------------------------------------------------------------------
EXTRACTION_PROMPT = """\
You are a property data extraction expert. Below is text extracted from a
property card document. Extract all available property information and return
ONLY a valid JSON object. DO NOT print any null values.

Field mapping guide:
    parcelid text -- Unique identifier for the parcel, e.g. apn, pid, pin, parcel_number, gpin.
        IMPORTANT: this is an explicitly-labeled parcel identifier (look for
        "Parcel ID", "APN", "PIN", "PID", "GPIN", "Parcel #", "Map ID").
        Usually contains 4-12 characters, sometimes with dashes or spaces.
    parcelid2 text -- Secondary identifier for the parcel, if available. e.g. lrsn, alternate_pid.
    taxacctnum text -- Tax account number associated with the parcel, e.g. tax_id, taxacctnum.
    taxyear int4 -- The tax year for which the data is relevant, e.g. 2025.
    usecode text -- The land use code assigned to the parcel.
    usedesc text -- A description of the land use associated with the parcel.

    zoningcode text -- The zoning code assigned to the parcel.
    zoningdesc text -- A description of the zoning associated with the parcel.
    numbldgs int4 -- Number of buildings on the parcel.
    numunits int4 -- Number of units on the parcel.
    yearbuilt int4 -- The year the primary building on the parcel was built.
        IMPORTANT: when multiple year fields exist (YrBlt, YrEff, YrRmd,
        YearBuilt, EffYr, Year Effective, Remodeled), always use YrBlt /
        Year Built / Original Year. Never use YrEff, EffYr, or "effective
        year" — that is a depreciation-adjusted year, not the build year.
    bldgsqft int4 -- Total square footage of the primary building on the parcel.
        IMPORTANT: use the LIVING / HEATED / MAIN / FINISHED area (labels
        such as "Living Area", "Heated SF", "Main Area", "Total Living
        Area", "Finished Area", "Gross Living Area"). Do NOT use "Gross
        Building Area" or "Gross Area".
    bedrooms int4 -- Number of bedrooms in the primary building on the parcel.
    halfbaths int4 -- Number of half bathrooms in the primary building on the parcel.
    fullbaths int4 -- Number of full bathrooms in the primary building on the parcel.
    imprvalue int8 -- Improvement value of the parcel.

    landvalue int8 -- Land value of the parcel.
        IMPORTANT: use the total land value shown in the summary/valuation
        row (labeled "Land", "Land Value", or "Total Land Value"), NOT a
        per-segment base rate or an adjacent column such as "Other",
        "Build", or "Improvement". If the card shows multiple land segments
        (e.g. BLDG SITE, OPEN, OPEN SPACE) with individual rates, sum them
        only if no total is given; otherwise use the total.
    agvalue int8 -- Agricultural value of the parcel.
    totalvalue int8 -- Total value of the parcel.
    taxacres float8 -- Assessed acres of the parcel.
    saleamt int8 -- Amount of the most recent for the parcel. IGNORE older sale records if multiple are present.
    saledate date -- Date of the MOST RECENT sale of the parcel. IGNORE older sale records if multiple are present.
    ownername text -- Name of the parcel owner.
    owneraddr text -- Address of the parcel owner.
    ownercity text -- City of the parcel owner.
    ownerstate text -- State of the parcel owner.

    ownerzip text -- ZIP code of the parcel owner.
    parceladdr text -- Address of the parcel.
        IMPORTANT: this is the PHYSICAL LOCATION of the property itself
        (labels like "Property Location", "Location", "Situs Address",
        "Site Address", "Property Address"). It is NOT the owner's mailing
        address. When the card lists the owner block with an address AND a
        separate property/site/situs address, always pick the
        property/site/situs address — those can differ when the owner lives
        elsewhere. The same rule applies to parcelcity, parcelstate,
        parcelzip: they describe the PROPERTY, not the owner. Use the
        owneraddr/ownercity/ownerstate/ownerzip fields for the owner's
        mailing address.
    parcelcity text -- City of the parcel.
    parcelstate text -- State of the parcel.
    parcelzip text -- ZIP code of the parcel.
    legaldesc text -- Legal description of the parcel.

    book text -- Book reference for the parcel.
    page text -- Page reference for the parcel.
    block text -- Block information for the parcel.
    lot text -- Lot information for the parcel.
    landusecode text -- The land use assigned to the parcel.

    landusedesc text -- A description of the land use associated with the parcel.
    house text -- The name of the house or building
    category text -- The category of the place (e.g., cafe, hospital)
    near text -- A landmark or nearby place
    house_number text -- The house or building number
    road text -- The name of the street or road
    unit text -- The unit, apartment, or suite number
    level text -- The level or floor number

    bldgtype text -- Type of building on the parcel (e.g., single-family, multi-family, commercial).
    numfloors int4 -- Number of floors in the primary building on the parcel.
    yearremodel int4 -- The year the primary building on the parcel was last remodeled.
    livingarea int4 -- Living area square footage of the primary building on the parcel.
    bldgquality text -- Quality rating of the primary building on the parcel.
    bldgcondition text -- Condition rating of the primary building on the parcel.
    architecture text -- Architectural style of the primary building on the parcel.

    totalrooms int4 -- Total number of rooms in the primary building on the parcel.
    fireplaces int4 -- Number of fireplaces in the primary building on the parcel.
    heating text -- Type of heating system in the primary building on the parcel
        (e.g. FORCED AIR, HEAT PUMP, BASEBOARD, RADIANT, WARMED & COOLED AIR).
        This describes the delivery/system type, NOT the fuel.
    heatfuel text -- Fuel used by the heating system. Allowed values: GAS, OIL,
        ELECTRIC, PROPANE, WOOD, SOLAR, COAL, NONE. Infer from any label that
        names a fuel in the heating/HVAC section, e.g. "Direct-Vented, Gas",
        "Gas Furnace", "Gas Pack", "Natural Gas" → GAS; "Oil Furnace",
        "Fuel Oil" → OIL; "Electric Heat", "Heat Pump (Electric)", "Baseboard
        Electric" → ELECTRIC; "LP", "Propane" → PROPANE; "Wood Stove",
        "Woodburning" → WOOD.
    cooling text -- Type of cooling system in the primary building on the parcel.
    foundation text -- Type of foundation of the primary building on the parcel.
    attic text -- Type of attic in the primary building on the parcel.
    atticsqft int4 -- Square footage of the attic in the primary building on the parcel.
    intwall text -- Type of interior walls in the primary building on the parcel.
        "intwall" will never be a material of an exterior wall, e.g. Stucco, Vinyl, or Brick.

    extwall text -- Type of exterior walls in the primary building on the parcel.
        extwall will never be a material of an interior wall, e.g. Drywall or Plaster.

    roofstyle text -- Style of the roof of the primary building on the parcel.

    roofcover text -- Type of roof covering/material of the primary building on the parcel.
    roofheight int4 -- Height of the roof of the primary building on the parcel.
    fsagpfeet int4 -- First story feet above ground level for FEMA Special Flood Hazard Area designation.
    siding text -- Type of siding on the primary building on the parcel.
    framing text -- Type of framing of the primary building on the parcel.
    basementsqft int4 -- Square footage of the basement in the primary building on the parcel.
    attgaragesqft int4 -- Square footage of the attached garage in the primary building on the parcel.
    detgaragesqft int4 -- Square footage of the detached garage on the parcel.
    garagestalls int4 -- Number of garage stalls on the parcel.
    situsaddress text -- Situs address of the parcel.

    appraisedvalue int8 -- Appraised value of the parcel.
    assessedvalue int8 -- Assessed value of the parcel.
    boatlift bool -- Indicates presence of a boat lift on the parcel.
    boatdock bool -- Indicates presence of a boat dock on the parcel.
    boathouse bool -- Indicates presence of a boathouse on the parcel.
    pool bool -- Indicates presence of a pool on the parcel.
    gazebo bool -- Indicates presence of a gazebo on the parcel.
    irrigation bool -- Indicates presence of an irrigation system on the parcel.
    riprap bool -- Indicates presence of riprap on the parcel.
    solarium bool -- Indicates presence of a solarium on the parcel.

    carport bool -- Indicates presence of a carport on the parcel.
    greenhouse bool -- Indicates presence of a greenhouse on the parcel.
    openporch bool -- Indicates presence of an open porch on the parcel.
    enclporch bool -- Indicates presence of an enclosed porch on the parcel.
    sauna bool -- Indicates presence of a sauna on the parcel.
    wooddeck bool -- Indicates presence of a wood deck on the parcel.
    hottub bool -- Indicates presence of a hot tub on the parcel.
    patio bool -- Indicates presence of a patio on the parcel.
    shed bool -- Indicates presence of a shed on the parcel.
    workshop bool -- Indicates presence of a workshop on the parcel.

    taxdistrict text -- Name of the tax district
);

Rules:
- Do not include any NULL values; omit fields that are null.
- Return ONLY valid JSON — no markdown, no explanation, no code fences.
- Monetary values must be integers with no $ signs or commas.
- Dates must be in YYYY-MM-DD format.
- Numeric fields must be numbers, not strings.
- It's okay if data is missing, but do not guess or fabricate data.
- parcelid, parcelid2, and taxacctnum cannot be equal to each other, or any other value on the card.

DOCUMENT TEXT:
{document_text}"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stream_llm(messages: list[dict], *, label: str = "extraction") -> str:
    """Stream a chat completion from Ollama and return the full response text."""
    stream_to_console = logger.isEnabledFor(logging.DEBUG)

    payload = {
        "model": CARD_READER_EXTRACTION_MODEL,
        "messages": messages,
        "stream": True,
        "think": False,
    }

    if stream_to_console:
        sys.stderr.write(f"ollama ({label})> ")
        sys.stderr.flush()

    chunks: list[str] = []
    metadata: dict = {}

    with httpx.Client(base_url=CARD_READER_OLLAMA_HOST, timeout=300) as client:
        with client.stream("POST", "/api/chat", json=payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.strip():
                    continue
                data = json.loads(line)
                if data.get("done"):
                    metadata = data
                else:
                    token = data.get("message", {}).get("content", "")
                    if token:
                        chunks.append(token)
                        if stream_to_console:
                            sys.stderr.write(token)
                            sys.stderr.flush()

    if stream_to_console:
        sys.stderr.write("\n")
        sys.stderr.flush()

    full_text = "".join(chunks)

    prompt_tokens = metadata.get("prompt_eval_count", 0)
    output_tokens = metadata.get("eval_count", 0)
    total_tokens = prompt_tokens + output_tokens
    if prompt_tokens or output_tokens:
        logger.info(
            "Ollama %s complete: %d prompt tokens + %d output tokens = %d total "
            "(%d response chars)",
            label, prompt_tokens, output_tokens, total_tokens, len(full_text),
        )
    else:
        logger.info("Ollama %s complete (%d response chars)", label, len(full_text))

    return full_text


def _parse_json_response(text: str) -> dict:
    """Extract JSON object from an LLM response, handling code fences."""
    # Strip <think>...</think> blocks (qwen3-style)
    text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
    # Try code-fenced JSON first
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        text = m.group(1)
    # Find the outermost JSON object
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        return json.loads(m.group(0))
    raise json.JSONDecodeError("No JSON object found in response", text, 0)


def _coerce_types(data: dict) -> dict:
    """Coerce extracted values to their expected Python types."""
    int_fields = {
        "taxyear", "numbldgs", "numunits", "yearbuilt", "numfloors",
        "bldgsqft", "bedrooms", "halfbaths", "fullbaths", "fireplaces",
    }
    bigint_fields = {"imprvalue", "landvalue", "agvalue", "totalvalue", "saleamt"}
    float_fields = {"taxacres" }

    coerced: dict = {}
    for k, v in data.items():
        if v is None:
            continue
        try:
            if k in int_fields or k in bigint_fields:
                # strip common formatting artifacts
                if isinstance(v, str):
                    v = v.replace(",", "").replace("$", "").strip()
                coerced[k] = int(float(v))
            elif k in float_fields:
                if isinstance(v, str):
                    v = v.replace(",", "").strip()
                coerced[k] = float(v)
            elif k == "saledate":
                if isinstance(v, str) and v.strip():
                    coerced[k] = v.strip()
            else:
                if isinstance(v, str) and v.strip():
                    upper = v.strip().upper()
                    if upper in ("YES", "TRUE"):
                        coerced[k] = True
                    elif upper in ("NO", "FALSE"):
                        coerced[k] = False
                    else:
                        coerced[k] = upper
                elif not isinstance(v, str):
                    coerced[k] = v
        except (ValueError, TypeError):
            logger.warning("Could not coerce field %s=%r, skipping", k, v)
    return coerced


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------


def _is_pdf(content: bytes) -> bool:
    """Return True if the content looks like a PDF (starts with %PDF magic)."""
    return content[:5] == b"%PDF-"


def _html_to_pdf(html: bytes, url: str) -> bytes:
    """Convert HTML content to PDF using pdfkit (wkhtmltopdf).

    Feeds the already-fetched HTML to wkhtmltopdf via stdin so it doesn't
    re-download the page (and so we don't need cert-bypass flags for the
    main URL). A ``<base href="...">`` tag is injected so wkhtmltopdf can
    resolve relative/protocol-relative URLs for sub-resources; any
    sub-resources that still fail are ignored via ``load-error-handling``.
    """
    import pdfkit

    logger.info("URL returned HTML; converting to PDF via pdfkit")
    html_str = html.decode("utf-8", errors="replace")

    base_tag = f'<base href="{url}">'
    if re.search(r"<head[^>]*>", html_str, flags=re.IGNORECASE):
        html_str = re.sub(
            r"(<head[^>]*>)", r"\1" + base_tag, html_str, count=1, flags=re.IGNORECASE,
        )
    else:
        html_str = base_tag + html_str

    options = {
        "load-error-handling": "ignore",
        "load-media-error-handling": "ignore",
        "quiet": "",
    }
    return pdfkit.from_string(html_str, False, options=options)


def download_pdf(state: AgentState) -> dict:
    """Download the PDF bytes.

    If the URL points to an HTML page instead of a PDF, the page is
    converted to PDF via pdfkit (wkhtmltopdf) before continuing.
    """
    pdf_bytes = state.get("pdf_bytes")
    if pdf_bytes:
        logger.info("Using pre-downloaded PDF bytes (%d bytes)", len(pdf_bytes))
        content = pdf_bytes
    else:
        url = state["pdf_url"]
        logger.info("Downloading PDF from %s", url)
        with httpx.Client(timeout=60, follow_redirects=True, verify=False) as client:
            resp = client.get(url)
            resp.raise_for_status()
        content = resp.content

    if not _is_pdf(content):
        content = _html_to_pdf(content, state["pdf_url"])

    return {"pdf_content": content}


def _iter_text_image_regions(content: bytes):
    """Yield ``(page_num, idx, ext, image_bytes)`` for each PDF image region
    that is plausibly text-bearing.

    Skips:
      * images smaller than 30px on either side (no readable glyphs fit), and
      * images >= 400px on both sides with aspect ratio <= 2 (likely photos).

    Used by both the OCR step and the test suite (which writes each region to
    disk as a JPG for debugging).
    """
    import pymupdf as fitz

    MIN_DIM = 30           # smaller than this can't hold readable glyphs
    PHOTO_DIM = 400        # both dims this large + near-square = treat as photo
    PHOTO_ASPECT = 2.0
    CLIP_DPI = 200         # render dpi for inline (xref=0) image regions

    doc = fitz.open(stream=content, filetype="pdf")
    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            seen_xrefs: set[int] = set()
            idx = 0

            for info in page.get_image_info(xrefs=True):
                w, h = info.get("width", 0), info.get("height", 0)
                if w < MIN_DIM or h < MIN_DIM:
                    continue

                # Skip images that look like property photos.
                if w >= PHOTO_DIM and h >= PHOTO_DIM:
                    long_side, short_side = max(w, h), min(w, h)
                    if short_side > 0 and long_side / short_side <= PHOTO_ASPECT:
                        continue

                xref = info.get("xref", 0) or 0
                try:
                    if xref:
                        if xref in seen_xrefs:
                            continue
                        seen_xrefs.add(xref)
                        extracted = doc.extract_image(xref)
                        img_bytes = extracted["image"]
                        ext = extracted.get("ext", "png")
                    else:
                        bbox = info.get("bbox")
                        if not bbox:
                            continue
                        pix = page.get_pixmap(clip=fitz.Rect(bbox), dpi=CLIP_DPI)
                        img_bytes = pix.tobytes("png")
                        ext = "png"
                except Exception as e:
                    logger.debug("Failed to extract image on page %d: %s", page_num + 1, e)
                    continue

                yield page_num + 1, idx, ext, img_bytes
                idx += 1
    finally:
        doc.close()


def _ocr_image_regions(content: bytes) -> str:
    """OCR text that's encoded as image regions inside the PDF.

    Some property cards bake field labels and headings as raster images rather
    than embedded text — pymupdf4llm cannot read those. This runs Tesseract on
    each text-bearing image region and returns one markdown section per page
    containing the joined OCR output.

    Regions are OCR'd in parallel via a thread pool: pytesseract shells out
    to the ``tesseract`` binary per call, so each invocation is its own OS
    process and threads exist only to wait on subprocess output. Cards
    routinely contain 25-100+ small text regions; serial OCR was the bulk
    of pipeline latency.
    """
    import io
    import os
    from concurrent.futures import ThreadPoolExecutor
    try:
        import pytesseract
        from PIL import Image
    except ModuleNotFoundError:
        logger.debug("pytesseract/Pillow not installed; skipping OCR")
        return ""

    regions = list(_iter_text_image_regions(content))
    if not regions:
        return ""

    def _ocr_one(region: tuple) -> tuple:
        page_num, idx, _ext, img_bytes = region
        try:
            img = Image.open(io.BytesIO(img_bytes))
            text = pytesseract.image_to_string(img).strip()
            return page_num, idx, img.width, img.height, text, None
        except Exception as e:
            return page_num, idx, 0, 0, "", e

    # Cap workers at cpu_count — each task spawns a tesseract subprocess so
    # over-subscription just thrashes. Fall back to 4 if cpu_count is unknown.
    max_workers = min(len(regions), os.cpu_count() or 4)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        # ``map`` preserves input order, which keeps OCR text on each page in
        # the same top-to-bottom order yielded by _iter_text_image_regions.
        results = list(pool.map(_ocr_one, regions))

    by_page: dict[int, list[str]] = {}
    region_count = 0
    for page_num, idx, w, h, text, err in results:
        if err is not None:
            logger.debug("OCR failed on page %d image %d: %s", page_num, idx, err)
            continue
        if text:
            region_count += 1
            logger.info(
                "OCR region p%d#%d (%dx%d) -> %r",
                page_num, idx, w, h,
                text if len(text) <= 200 else text[:200] + "…",
            )
            by_page.setdefault(page_num, []).append(text)

    sections = [
        f"## Page {page_num} — Image-Encoded Text\n\n" + "\n".join(texts)
        for page_num, texts in sorted(by_page.items())
    ]
    combined = "\n\n".join(sections)
    if combined:
        logger.info(
            "OCR'd image regions: %d region(s) across %d worker(s), "
            "%d chars across %d page section(s)",
            region_count, max_workers, len(combined), len(sections),
        )
        logger.info("OCR image-region full text:\n%s", combined)
    return combined


def _ocr_full_pages(content: bytes, dpi: int = 300) -> str:
    """OCR each page by rendering it as a full-page bitmap.

    Falls back when embedded text + image-region OCR miss content — for
    example, property cards that bake every value (owner, year built, heat
    fuel, etc.) into a single large rasterized card image, or PDFs whose
    values are scattered across hundreds of tiny inline images too small for
    the per-region OCR path to pick up.
    """
    import io
    try:
        import pymupdf as fitz
        import pytesseract
        from PIL import Image
    except ModuleNotFoundError:
        logger.debug("pytesseract/Pillow not installed; skipping full-page OCR")
        return ""

    doc = fitz.open(stream=content, filetype="pdf")
    sections: list[str] = []
    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            try:
                pix = page.get_pixmap(dpi=dpi)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                text = pytesseract.image_to_string(img).strip()
            except Exception as e:
                logger.debug("Full-page OCR failed on page %d: %s", page_num + 1, e)
                continue
            if text:
                logger.info(
                    "Full-page OCR p%d (%dx%d @ %d DPI) -> %d chars:\n%s",
                    page_num + 1, img.width, img.height, dpi, len(text), text,
                )
                sections.append(f"## Page {page_num + 1} — Full-Page OCR\n\n{text}")
    finally:
        doc.close()

    combined = "\n\n".join(sections)
    if combined:
        logger.info("Full-page OCR: %d chars across %d page(s)",
                    len(combined), len(sections))
    return combined


def _extract_markdown(content: bytes) -> str:
    """Extract embedded text from the PDF and clean it up as markdown."""
    import pymupdf as fitz
    import pymupdf4llm

    try:
        doc = fitz.open(stream=content, filetype="pdf")
        markdown = pymupdf4llm.to_markdown(doc, use_ocr=False)
        logger.info("Extracted markdown via pymupdf4llm (%d chars)", len(markdown))
    except Exception as e:
        logger.warning("pymupdf4llm extraction failed (%s).", e)
        return ""
    finally:
        doc.close()

    # Deduplicate repeated table cells BEFORE <br> conversion (while cells
    # are still on single lines). pymupdf4llm sometimes emits the same cell
    # content once per table column:
    #   |Owner:<br>230506<br>...|Owner:<br>230506<br>...|
    # This wastes context and reinforces wrong label→value associations.
    def _dedup_table_row(line: str) -> str:
        if "|" not in line:
            return line
        cells = [c.strip() for c in line.split("|")]
        seen: list[str] = []
        for cell in cells:
            if cell and cell not in seen:
                seen.append(cell)
        if not seen:
            return ""
        return "| " + " | ".join(seen) + " |"
    markdown = "\n".join(_dedup_table_row(line) for line in markdown.splitlines())

    # Convert <br> to newlines so the LLM sees labels and values on
    # separate lines.
    markdown = re.sub(r"<br\s*/?>", "\n", markdown, flags=re.IGNORECASE)

    # Remove floorplan / sketch noise: lines made entirely of box-drawing
    # characters (+, -, |, whitespace) and runs of empty table cells (||||).
    markdown = re.sub(r"^[\s|+\-]+$", "", markdown, flags=re.MULTILINE)
    markdown = re.sub(r"\|{3,}", "", markdown)

    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    return markdown


# When the combined embedded-text + image-region OCR yields fewer chars than
# this, trigger full-page OCR as a last resort. Property cards that bake all
# field values into large rasterized images (rather than selectable PDF text)
# otherwise end up with a mostly-empty context.
_FULL_PAGE_OCR_FALLBACK_THRESHOLD = 1500


def extract_pdf_text(state: AgentState) -> dict:
    """Extract embedded text from the PDF and format it as markdown.

    Also OCRs any image-encoded text regions (Tesseract) and appends that
    text to the markdown so labels/headings rendered as bitmaps aren't lost.
    If the combined text is still thin, falls back to full-page OCR.
    """
    content = state["pdf_content"]

    markdown = _extract_markdown(content)

    # Append text recovered from image-encoded regions via Tesseract.
    image_text = _ocr_image_regions(content)
    if image_text:
        markdown = f"{markdown}\n\n{image_text}" if markdown.strip() else image_text

    if len(markdown) < _FULL_PAGE_OCR_FALLBACK_THRESHOLD:
        page_text = _ocr_full_pages(content)
        if page_text:
            markdown = f"{markdown}\n\n{page_text}" if markdown.strip() else page_text

    return {"pdf_markdown": markdown}


def extract_pdf_and_photos(state: AgentState) -> dict:
    """Run markdown extraction, Tesseract OCR, and photo classification in parallel.

    Combines the results of extract_pdf_text and extract_property_photos but
    executes the three independent I/O-bound tasks concurrently via threads.
    """
    from concurrent.futures import ThreadPoolExecutor

    content = state["pdf_content"]

    with ThreadPoolExecutor(max_workers=3) as pool:
        md_future = pool.submit(_extract_markdown, content)
        ocr_future = pool.submit(_ocr_image_regions, content)
        photos_future = pool.submit(extract_property_photos, state)

    markdown = md_future.result()
    image_text = ocr_future.result()
    if image_text:
        markdown = f"{markdown}\n\n{image_text}" if markdown.strip() else image_text

    if len(markdown) < _FULL_PAGE_OCR_FALLBACK_THRESHOLD:
        page_text = _ocr_full_pages(content)
        if page_text:
            markdown = f"{markdown}\n\n{page_text}" if markdown.strip() else page_text

    photos_result = photos_future.result()

    return {"pdf_markdown": markdown, **photos_result}


def _is_property_photo(image_bytes: bytes) -> bool:
    """Ask the vision model whether an image is a photo of a property.

    Uses CARD_READER_PHOTO_CLASSIFICATION_MODEL (a lightweight vision model) via Ollama
    with an image token budget of 70 to keep classification fast.

    Returns True for photographs of buildings, houses, structures, or land/lots.
    Returns False for sketches, floorplans, maps, logos, signatures, charts, etc.
    """
    import base64

    b64 = base64.b64encode(image_bytes).decode()
    payload = {
        "model": CARD_READER_PHOTO_CLASSIFICATION_MODEL,
        "messages": [{
            "role": "user",
            "content": (
                "Is this a PHOTOGRAPH of a property, building, house, "
                "structure, or land/lot? Sketches, floorplans, maps, "
                "diagrams, logos, and charts are NOT photographs. "
                "Answer only YES or NO."
            ),
            "images": [b64],
        }],
        "stream": False,
        "think": False,
        "options": {
            "visual_token_budget": 70,
        },
    }

    try:
        with httpx.Client(base_url=CARD_READER_OLLAMA_HOST, timeout=60) as client:
            resp = client.post("/api/chat", json=payload)
            resp.raise_for_status()
            answer = resp.json().get("message", {}).get("content", "").strip().upper()
            is_photo = answer.startswith("YES")
            logger.info("Photo classification (%s): %r -> %s",
                       CARD_READER_PHOTO_CLASSIFICATION_MODEL, answer, is_photo)
            return is_photo
    except Exception as e:
        logger.warning("Photo classification failed (%s); assuming not a property photo", e)
        return False


def extract_property_photos(state: AgentState) -> dict:
    """Extract embedded photos of the property from the PDF.

    Candidate images are discovered via ``page.get_image_info`` and filtered
    by size / aspect ratio. Each candidate is then sent to the Ollama vision
    model for classification — only actual photographs of properties,
    buildings, or land are kept (sketches, floorplans, maps, etc. are
    discarded).

    Returns each photo as a dict with raw bytes:
        {"page": int, "width": int, "height": int, "ext": str, "bytes": bytes}
    """
    import pymupdf as fitz

    MIN_DIM = 200          # smallest side, in image pixels
    MAX_ASPECT_RATIO = 4   # skip very long banners
    CLIP_DPI = 150         # render dpi for inline (xref=0) images

    content = state["pdf_content"]
    candidates: list[dict] = []
    seen_xrefs: set[int] = set()

    doc = fitz.open(stream=content, filetype="pdf")
    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            for info in page.get_image_info(xrefs=True):
                w, h = info.get("width", 0), info.get("height", 0)
                if w < MIN_DIM or h < MIN_DIM:
                    continue
                long_side, short_side = max(w, h), min(w, h)
                if short_side <= 0 or long_side / short_side > MAX_ASPECT_RATIO:
                    continue

                xref = info.get("xref", 0) or 0
                if xref:
                    if xref in seen_xrefs:
                        continue
                    seen_xrefs.add(xref)
                    extracted = doc.extract_image(xref)
                    candidates.append({
                        "page": page_num + 1,
                        "width": extracted.get("width", w),
                        "height": extracted.get("height", h),
                        "ext": extracted.get("ext", "bin"),
                        "bytes": extracted["image"],
                    })
                else:
                    bbox = info.get("bbox")
                    if not bbox:
                        continue
                    pix = page.get_pixmap(clip=fitz.Rect(bbox), dpi=CLIP_DPI)
                    candidates.append({
                        "page": page_num + 1,
                        "width": pix.width,
                        "height": pix.height,
                        "ext": "jpeg",
                        "bytes": pix.tobytes("jpeg"),
                    })
    finally:
        doc.close()

    logger.info("Found %d candidate image(s), classifying with vision model", len(candidates))

    photos = [c for c in candidates if _is_property_photo(c["bytes"])]

    logger.info("Kept %d/%d image(s) as property photos", len(photos), len(candidates))
    return {"property_photos": photos}


def _run_extraction_llm(source_text: str, context: str | None = None) -> dict:
    """Run the extraction LLM on a body of text and return coerced property data.

    Streams the response from Ollama and emits each chunk to the DEBUG log as
    it arrives, so you can watch generation progress with --log-cli-level=DEBUG.

    If ``context`` is provided, it is appended to the extraction prompt as
    additional caller-supplied instructions.
    """
    prompt = EXTRACTION_PROMPT.format(document_text=source_text)
    if context and context.strip():
        prompt = f"{prompt}\n\nAdditional instructions:\n{context.strip()}"
    logger.info("Extracting structured data (model=%s)", CARD_READER_EXTRACTION_MODEL)

    messages = [
        {"role": "system", "content": "You are a precise data extraction assistant. Return only valid JSON."},
        {"role": "user", "content": prompt},
    ]

    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        full_text = _stream_llm(messages, label="extraction")
        try:
            raw = _parse_json_response(full_text)
            break
        except json.JSONDecodeError:
            logger.warning(
                "Attempt %d/%d: could not parse JSON from response (length=%d): %.500s",
                attempt, max_attempts, len(full_text), full_text,
            )
            if attempt == max_attempts:
                raise
    return _coerce_types(raw)


def _reconcile_value_totals(data: dict) -> dict:
    """Ensure landvalue + imprvalue == totalvalue.

    If exactly one of the three is missing but the other two are present,
    compute the missing value arithmetically. If all three are present but
    inconsistent, log a warning and keep the extracted values as-is.
    """
    land = data.get("landvalue")
    impr = data.get("imprvalue")
    total = data.get("totalvalue")

    have_land = isinstance(land, int)
    have_impr = isinstance(impr, int)
    have_total = isinstance(total, int)
    present = have_land + have_impr + have_total

    if present == 2:
        if not have_total:
            data["totalvalue"] = land + impr
            logger.info("Computed missing totalvalue: %d + %d = %d", land, impr, data["totalvalue"])
        elif not have_impr:
            data["imprvalue"] = total - land
            logger.info("Computed missing imprvalue: %d - %d = %d", total, land, data["imprvalue"])
        elif not have_land:
            data["landvalue"] = total - impr
            logger.info("Computed missing landvalue: %d - %d = %d", total, impr, data["landvalue"])
        return data

    if present < 3:
        return data

    if land + impr != total:
        logger.warning(
            "Value totals inconsistent: land=%d + impr=%d = %d != total=%d; "
            "keeping extracted values",
            land, impr, land + impr, total,
        )
    return data


PARCELID_RETRY_PROMPT = """\
A property card was parsed and the extracted parcel ID was: "{parcelid}"

That value is too short or does not look like a real parcel identifier.
Re-read the property card text below and find the CORRECT parcel ID.

Look for labels like "Parcel ID", "APN", "PIN", "GPIN", "Tax Map", "Map ID",
"Parcel #", "Parcel Number", "PID", "Alt ID", "GIS ID". The value is usually
4-20 characters and may contain digits, dots, dashes, slashes, or spaces
(e.g. "3208.794.862", "101-02-003", "32.5(000)000/052").

Do NOT return a ZIP code, tax account number, sale price, book/page reference,
or vision/record ID.

Return ONLY a JSON object like: {{"parcelid": "<value>"}}

PROPERTY CARD TEXT:
{document_text}
"""


def _retry_parcelid(data: dict, markdown: str) -> dict:
    """If the extracted parcelid is fewer than 3 characters, re-query the LLM
    once with a focused prompt to find a better one."""
    pid = data.get("parcelid")
    if not isinstance(pid, str) or len(pid) >= 3:
        return data

    logger.warning(
        "parcelid %r is fewer than 3 chars; re-querying LLM for a better match",
        pid,
    )

    full_text = _stream_llm(
        [{"role": "user", "content": PARCELID_RETRY_PROMPT.format(
            parcelid=pid, document_text=markdown,
        )}],
        label="parcelid-retry",
    )

    try:
        raw = _parse_json_response(full_text)
    except json.JSONDecodeError:
        logger.warning("parcelid retry returned non-JSON; keeping original")
        return data

    new_pid = raw.get("parcelid")
    if isinstance(new_pid, str) and len(new_pid) >= 3:
        logger.info("parcelid corrected: %r -> %r", pid, new_pid.upper())
        data["parcelid"] = new_pid.strip().upper()
    else:
        logger.warning("parcelid retry returned %r; keeping original %r", new_pid, pid)

    return data


PHOTO_ANALYSIS_PROMPT = """\
You are a property data extraction expert. Below is a photograph of a property.
The following fields are MISSING from the property record and could not be
determined from the document text. Examine the photo and fill in any fields
you can confidently determine from what you see.

Missing fields:
{missing_fields}

Return ONLY a valid JSON object containing the fields you can fill in.
Omit any field you cannot confidently determine from the photo.
No $ signs, no commas in numbers, no explanation.
"""

# Fields that can plausibly be inferred from a property photo.
PHOTO_INFERABLE_FIELDS = {
    "bldgtype", "numfloors", "architecture", "extwall", "roofstyle",
    "roofcover", "siding", "pool", "gazebo", "carport", "shed",
    "boatlift", "boatdock", "boathouse", "solarium", "hottub", "patio",
    "wooddeck", "openporch", "enclporch"
}


def fill_from_photo(data: dict, image_bytes: bytes) -> dict:
    """Use the vision model to fill missing fields by analyzing the property photo.

    Only attempts fields that are plausibly inferable from a photograph
    (building style, exterior materials, visible features, etc.). Fields
    already populated in *data* are never overwritten.
    """
    import base64

    missing = PHOTO_INFERABLE_FIELDS - set(data.keys())
    if not missing:
        logger.info("No photo-inferable fields missing; skipping photo analysis")
        return data

    field_descriptions = "\n".join(f"    {f}" for f in sorted(missing))
    prompt = PHOTO_ANALYSIS_PROMPT.format(missing_fields=field_descriptions)

    b64 = base64.b64encode(image_bytes).decode()
    logger.info("Analyzing property photo for %d missing field(s)", len(missing))

    try:
        # Use Ollama directly to set visual_token_budget for gemma4 models.
        payload = {
            "model": CARD_READER_EXTRACTION_MODEL,
            "messages": [{"role": "user", "content": prompt, "images": [b64]}],
            "stream": False,
            "think": False,
            "options": {"visual_token_budget": 1120},
        }
        with httpx.Client(base_url=CARD_READER_OLLAMA_HOST, timeout=300) as client:
            resp = client.post("/api/chat", json=payload)
            resp.raise_for_status()
            result = resp.json()
            full_text = result.get("message", {}).get("content", "")
            prompt_tokens = result.get("prompt_eval_count", 0)
            output_tokens = result.get("eval_count", 0)
            logger.info(
                "Ollama photo-analysis complete: %d prompt + %d output = %d total",
                prompt_tokens, output_tokens, prompt_tokens + output_tokens,
            )
        raw = _parse_json_response(full_text)
        new_fields = _coerce_types(raw)
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("Photo analysis failed (%s); no fields added", e)
        return data

    added = 0
    for k, v in new_fields.items():
        if k in PHOTO_INFERABLE_FIELDS and k not in data and v not in (None, "", 0):
            data[k] = v
            added += 1

    logger.info("Photo analysis added %d field(s): %s", added,
               [k for k in new_fields if k in PHOTO_INFERABLE_FIELDS and k not in data] if added else "none")
    logger.info("Photo analysis added %d field(s): %s", added, new_fields)
    return data


# Fuel-abbreviation -> canonical heatfuel value. Anchored so "ELECT" matches
# but "ELECTRONICS" does not. Order matters only for readability.
_HEATFUEL_TOKEN_MAP: list[tuple[str, str]] = [
    (r"ELEC(?:T(?:R(?:IC)?)?)?", "ELECTRIC"),
    (r"PROPANE|LP\s*GAS|LPG", "PROPANE"),
    (r"NATURAL\s*GAS|N\.?G\.?|GAS", "GAS"),
    (r"FUEL\s*OIL|OIL", "OIL"),
    (r"WOOD(?:\s*STOVE|BURN(?:ING)?)?", "WOOD"),
    (r"SOLAR", "SOLAR"),
    (r"COAL", "COAL"),
    (r"NONE", "NONE"),
]


def _post_extract_heatfuel(data: dict, markdown: str) -> dict:
    """Fill heatfuel from a 'Heat Fuel' / 'Fuel Type' label in the markdown.

    The extraction LLM occasionally drops heatfuel even when the card clearly
    labels it (e.g. "Heat Fuel ELECT"). This pass scans the markdown for the
    label and maps the adjacent token to a canonical fuel. Already-populated
    heatfuel values are never overwritten.
    """
    if data.get("heatfuel"):
        return data

    # Look for "Heat Fuel" / "Fuel Type" / "Fuel:" followed by a value within
    # the next ~30 chars (same line or next line on the card).
    label_re = re.compile(
        r"(?:heat\s*fuel|fuel\s*type|fuel)\s*[:\-]?\s*([A-Za-z./\s-]{1,25})",
        re.IGNORECASE,
    )
    for m in label_re.finditer(markdown):
        candidate = m.group(1).strip().upper()
        for pattern, canonical in _HEATFUEL_TOKEN_MAP:
            if re.search(rf"\b{pattern}\b", candidate):
                logger.info(
                    "heatfuel recovered from markdown: %r -> %s", m.group(0).strip(), canonical,
                )
                data["heatfuel"] = canonical
                return data
    return data


# Per-field label regex + short retry hint. When a label appears in the
# markdown but the field is empty after the main extraction (and any
# deterministic post-processors), the retry below re-asks the LLM in one
# focused batch. Each pattern is anchored at word boundaries to keep label
# detection conservative — false positives just trigger an extra LLM call,
# but false negatives leave fields unfilled.
_FIELD_RETRY_HINTS: dict[str, tuple[re.Pattern, str]] = {
    "heatfuel": (
        re.compile(
            r"\b(?:heat\s*fuel|heating\s*fuel|fuel\s*type|fuel\s*source|"
            r"heating\s*source|energy\s*source|heat\s*type|heating\s*system)\b",
            re.IGNORECASE,
        ),
        "Allowed values: GAS, OIL, ELECTRIC, PROPANE, WOOD, SOLAR, COAL, NONE. "
        "Common abbreviations: 'ELECT'/'ELEC' -> ELECTRIC, 'LP'/'LPG' -> PROPANE, "
        "'N.G.'/'Natural Gas' -> GAS, 'Fuel Oil' -> OIL.",
    ),
    "heating": (
        re.compile(r"\b(?:heating(?:\s*system|\s*type)?|heat\s*type|hvac\s*system)\b", re.IGNORECASE),
        "Heating system delivery type (FORCED AIR, HEAT PUMP, BASEBOARD, RADIANT, "
        "WARMED & COOLED AIR). NOT the fuel.",
    ),
    "cooling": (
        re.compile(r"\b(?:cooling|central\s*air|a\s*/?\s*c\s*type|air\s*condition)\b", re.IGNORECASE),
        "Cooling system type (CENTRAL AIR, NONE, WINDOW UNIT, HEAT PUMP, etc.).",
    ),
    "yearbuilt": (
        re.compile(
            r"\b(?:year\s*built|yr\s*blt|original\s*year|year\s*of\s*construction)\b",
            re.IGNORECASE,
        ),
        "Original year built (4-digit integer). NOT effective year, effective age, "
        "or remodel year.",
    ),
    "yearremodel": (
        re.compile(r"\b(?:year\s*remodel(?:ed)?|yr\s*rmd|remodel\s*year)\b", re.IGNORECASE),
        "Year of last remodel (4-digit integer).",
    ),
    "bldgsqft": (
        re.compile(
            r"\b(?:living\s*area|heated\s*s(?:f|q\.?\s*ft)|finished\s*area|"
            r"total\s*living(?:\s*area)?|main\s*area|total\s*square\s*foot)\b",
            re.IGNORECASE,
        ),
        "Living / heated / finished area square footage (integer). NOT gross "
        "building area.",
    ),
    "bedrooms": (
        re.compile(
            r"\b(?:total\s*bedrooms?|number\s*of\s*bedrooms?|#\s*bedrooms?|bedrooms?)\b",
            re.IGNORECASE,
        ),
        "Bedroom count (integer).",
    ),
    "fullbaths": (
        re.compile(
            r"\b(?:full\s*baths?|total\s*bathrooms?|number\s*of\s*baths?)\b",
            re.IGNORECASE,
        ),
        "Full bathroom count (integer).",
    ),
    "halfbaths": (
        re.compile(r"\b(?:half\s*baths?|1\s*/\s*2\s*baths?|powder\s*rooms?)\b", re.IGNORECASE),
        "Half bathroom count (integer).",
    ),
    "fireplaces": (
        re.compile(
            r"\b(?:#\s*of\s*fireplaces?|number\s*of\s*fireplaces?|fireplaces?)\b",
            re.IGNORECASE,
        ),
        "Fireplace count (integer).",
    ),
    "imprvalue": (
        re.compile(
            r"\b(?:improvement\s*value|building\s*value|impr\s*value)\b",
            re.IGNORECASE,
        ),
        "Improvement / building value (integer, no commas / $).",
    ),
    "landvalue": (
        re.compile(r"\bland\s*value\b", re.IGNORECASE),
        "Land value (integer, no commas / $).",
    ),
    "totalvalue": (
        re.compile(r"\b(?:total\s*value|total\s*assessed|grand\s*total)\b", re.IGNORECASE),
        "Total assessed value (integer, no commas / $).",
    ),
    "saleamt": (
        re.compile(r"\bsale\s*(?:price|amount|amt)\b", re.IGNORECASE),
        "Most recent sale price (integer). Use the latest transfer only.",
    ),
    "saledate": (
        re.compile(r"\bsale\s*date\b", re.IGNORECASE),
        "Most recent sale date in YYYY-MM-DD format. Use the latest transfer only.",
    ),
    "ownername": (
        re.compile(r"\b(?:current\s*owner|owner\s*name|primary\s*owner)\b", re.IGNORECASE),
        "Current owner name (text).",
    ),
    "parceladdr": (
        re.compile(
            r"\b(?:property\s*(?:address|location)|situs(?:\s*address)?|"
            r"site\s*address|location\s*address)\b",
            re.IGNORECASE,
        ),
        "Physical property address (text). NOT the owner's mailing address.",
    ),
    "legaldesc": (
        re.compile(r"\b(?:legal\s*description|legal\s*desc)\b", re.IGNORECASE),
        "Legal description (text).",
    ),
    "foundation": (
        re.compile(r"\bfoundation(?:\s*wall)?\b", re.IGNORECASE),
        "Foundation type (text).",
    ),
    "extwall": (
        re.compile(r"\bext(?:erior)?\s*wall\b", re.IGNORECASE),
        "Exterior wall material (text). NOT an interior wall material.",
    ),
    "intwall": (
        re.compile(r"\bint(?:erior)?\s*wall\b", re.IGNORECASE),
        "Interior wall material (text). NOT an exterior wall material.",
    ),
    "roofcover": (
        re.compile(r"\broof\s*cover\b", re.IGNORECASE),
        "Roof cover material (text).",
    ),
    "roofstyle": (
        re.compile(r"\broof\s*(?:style|type|shape)\b", re.IGNORECASE),
        "Roof style (text, e.g. GABLE, HIP, FLAT).",
    ),
    "bldgquality": (
        re.compile(r"\b(?:grade(?:\s*\%)?|building\s*grade|building\s*quality)\b", re.IGNORECASE),
        "Building grade / quality (text, often a letter or letter+number).",
    ),
    "numfloors": (
        re.compile(
            r"\b(?:stories|number\s*of\s*floors|num\s*floors|floor\s*count)\b",
            re.IGNORECASE,
        ),
        "Number of floors / stories (integer; round 1.5 -> 1, 2.5 -> 2).",
    ),
    "usecode": (
        re.compile(r"\b(?:property\s*use|use\s*code|land\s*use\s*code)\b", re.IGNORECASE),
        "Use code (short alphanumeric, e.g. 'R1', '00', '200R').",
    ),
    "zoningcode": (
        re.compile(r"\bzoning(?:\s*code)?\b", re.IGNORECASE),
        "Zoning code (short alphanumeric, e.g. 'SR', 'R-1', 'A1').",
    ),
}


# Canonical heatfuel set used to normalize the LLM's retry response.
_CANONICAL_HEATFUEL = {"GAS", "OIL", "ELECTRIC", "PROPANE", "WOOD", "SOLAR", "COAL", "NONE"}


def _normalize_heatfuel(value: str) -> str | None:
    """Map a free-form heatfuel string to its canonical value if possible."""
    if not isinstance(value, str):
        return None
    candidate = value.strip().upper()
    if not candidate:
        return None
    if candidate in _CANONICAL_HEATFUEL:
        return candidate
    for pattern, mapped in _HEATFUEL_TOKEN_MAP:
        if re.fullmatch(pattern, candidate):
            return mapped
    return None


MISSING_FIELDS_RETRY_PROMPT = """\
The property card text below carries labels for fields that were NOT filled
in on the first extraction pass. Re-read the card and fill in any of the
fields below that you can confidently determine.

Missing fields with hints:
{field_block}

CRITICAL rules — read carefully:
- A label being PRESENT does NOT mean the field has a value. If the label's
  value cell is blank, empty, dashed, or whitespace, OMIT that field. Do
  NOT guess and do NOT borrow a value from a nearby cell.
- Each field must come from its OWN labeled cell on the card. Never reuse
  a value from one field as the value for another (e.g. a "Primary Use"
  code is never a "Zoning Code", a "Year Remodeled" is never the
  "Year Built").
- Return ONLY a JSON object containing only the fields you can confidently
  fill in. Numeric fields must be numbers (no $, no commas). Dates must be
  YYYY-MM-DD.
- If you cannot confidently determine ANY of these fields, return {{}}.

PROPERTY CARD TEXT:
{document_text}
"""


def _retry_missing_labeled_fields(data: dict, markdown: str) -> dict:
    """Re-query the LLM for any field whose label appears in the markdown
    but was not filled in by the main extraction.

    Catches the common failure mode where individual fields get dropped
    under context noise — a busy OCR'd construction-detail block buries
    individual values, but the labels are right there. By naming the
    missing fields explicitly we focus the model's attention.

    Already-populated values are never overwritten.
    """
    missing: list[tuple[str, str]] = []
    for field, (label_re, hint) in _FIELD_RETRY_HINTS.items():
        existing = data.get(field)
        if existing not in (None, ""):
            continue
        if label_re.search(markdown):
            missing.append((field, hint))

    if not missing:
        return data

    field_block = "\n".join(f"  {f}: {h}" for f, h in missing)
    logger.warning(
        "Missing-field retry: %d labeled field(s) unfilled: %s",
        len(missing), ", ".join(f for f, _ in missing),
    )

    try:
        full_text = _stream_llm(
            [{"role": "user", "content": MISSING_FIELDS_RETRY_PROMPT.format(
                field_block=field_block, document_text=markdown,
            )}],
            label="missing-fields-retry",
        )
    except Exception as e:
        logger.warning("Missing-field retry failed (%s); leaving fields unset", e)
        return data

    try:
        raw = _parse_json_response(full_text)
    except json.JSONDecodeError:
        logger.warning("Missing-field retry returned non-JSON; leaving fields unset")
        return data

    if not isinstance(raw, dict):
        return data

    requested = {f for f, _ in missing}
    restricted = {k: v for k, v in raw.items() if k in requested}

    # Heatfuel needs canonical normalization since the LLM may emit "ELECT".
    if "heatfuel" in restricted:
        normalized = _normalize_heatfuel(str(restricted["heatfuel"]))
        if normalized is None:
            logger.warning(
                "heatfuel retry returned non-canonical %r; dropping", restricted["heatfuel"],
            )
            del restricted["heatfuel"]
        else:
            restricted["heatfuel"] = normalized

    coerced = _coerce_types(restricted)

    # Reject retry values that collide with an already-extracted value for a
    # different field. This is the LLM's main misallocation failure mode:
    # when it can't find the labeled value, it falls back to a nearby cell
    # (e.g. emitting Primary Use as zoningcode). Comparing as upper-case
    # strings catches both numeric-as-string and case-folded matches.
    existing_values = {
        str(v).strip().upper()
        for k, v in data.items()
        if k not in coerced and v not in (None, "")
    }

    added: list[str] = []
    rejected: list[str] = []
    for k, v in coerced.items():
        if data.get(k) not in (None, ""):
            continue
        if str(v).strip().upper() in existing_values:
            rejected.append(f"{k}={v!r} (collides with another field)")
            continue
        data[k] = v
        added.append(k)

    if added:
        logger.info(
            "Missing-field retry recovered %d field(s): %s",
            len(added), ", ".join(f"{k}={data[k]!r}" for k in added),
        )
    if rejected:
        logger.warning(
            "Missing-field retry rejected %d field(s): %s",
            len(rejected), "; ".join(rejected),
        )
    if not added and not rejected:
        logger.info("Missing-field retry yielded no usable fields")

    return data


def extract_data(state: AgentState) -> dict:
    """Extract structured property data from the markdown PDF text."""
    markdown = state.get("pdf_markdown", "") or ""
    if len(markdown.strip()) < 50:
        logger.info(
            "PDF yielded %d chars of text — skipping extraction",
            len(markdown.strip()),
        )
        return {"property_data": {}}

    data = _run_extraction_llm(markdown, state.get("context"))
    data = _reconcile_value_totals(data)
    data = _retry_parcelid(data, markdown)
    data = _post_extract_heatfuel(data, markdown)
    data = _retry_missing_labeled_fields(data, markdown)
    logger.info("Extracted %d fields from PDF text", len(data))
    return {"property_data": data}
