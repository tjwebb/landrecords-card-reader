import json
import logging
import re
import sys

import httpx

from .config import (
    CLASSIFICATION_CONTEXT_LENGTH,
    EXTRACTION_CONTEXT_LENGTH,
    EXTRACTION_MODEL,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    OLLAMA_BASE_URL,
    PHOTO_CLASSIFICATION_MODEL,
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
    "cooling", "foundation", "attic", "atticsqft", "intwall", "extwall",
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
    heating text -- Type of heating system in the primary building on the parcel.
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


def _stream_ollama(messages: list[dict], *, label: str = "extraction") -> str:
    """Stream a chat completion from Ollama and return the full response text."""
    stream_to_console = logger.isEnabledFor(logging.DEBUG)

    payload = {
        "model": EXTRACTION_MODEL,
        "messages": messages,
        "stream": True,
        "think": False,
    }
    if EXTRACTION_CONTEXT_LENGTH:
        payload.setdefault("options", {})["num_ctx"] = EXTRACTION_CONTEXT_LENGTH

    if stream_to_console:
        sys.stderr.write(f"ollama ({label})> ")
        sys.stderr.flush()

    chunks: list[str] = []
    metadata: dict = {}

    with httpx.Client(base_url=OLLAMA_BASE_URL, timeout=300) as client:
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


def _stream_gemini(messages: list[dict], *, label: str = "extraction") -> str:
    """Stream a chat completion from the Gemini API and return the full response text."""
    stream_to_console = logger.isEnabledFor(logging.DEBUG)

    # Convert Ollama-style messages to Gemini format.
    # Gemini uses "user"/"model" roles; system instructions go in a separate field.
    system_parts: list[str] = []
    contents: list[dict] = []
    for msg in messages:
        role = msg.get("role", "user")
        if role == "system":
            system_parts.append(msg["content"])
            continue

        gemini_role = "model" if role == "assistant" else "user"
        parts: list[dict] = [{"text": msg["content"]}]

        # Attach images (base64) if present — Gemini uses inline_data.
        for img_b64 in msg.get("images", []):
            parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": img_b64,
                },
            })

        contents.append({"role": gemini_role, "parts": parts})

    payload: dict = {
        "contents": contents,
        "generationConfig": {
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    if system_parts:
        payload["system_instruction"] = {
            "parts": [{"text": t} for t in system_parts],
        }

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:streamGenerateContent?alt=sse&key={GEMINI_API_KEY}"
    )

    if stream_to_console:
        sys.stderr.write(f"gemini ({label})> ")
        sys.stderr.flush()

    chunks: list[str] = []
    prompt_tokens = 0
    output_tokens = 0

    with httpx.Client(timeout=300) as client:
        with client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = json.loads(line[6:])

                # Accumulate text from candidates.
                for candidate in data.get("candidates", []):
                    for part in candidate.get("content", {}).get("parts", []):
                        token = part.get("text", "")
                        if token:
                            chunks.append(token)
                            if stream_to_console:
                                sys.stderr.write(token)
                                sys.stderr.flush()

                # Capture usage from the final chunk.
                usage = data.get("usageMetadata")
                if usage:
                    prompt_tokens = usage.get("promptTokenCount", 0)
                    output_tokens = usage.get("candidatesTokenCount", 0)

    if stream_to_console:
        sys.stderr.write("\n")
        sys.stderr.flush()

    full_text = "".join(chunks)
    total_tokens = prompt_tokens + output_tokens

    if prompt_tokens or output_tokens:
        logger.info(
            "Gemini %s complete: %d prompt tokens + %d output tokens = %d total "
            "(%d response chars)",
            label, prompt_tokens, output_tokens, total_tokens, len(full_text),
        )
    else:
        logger.info("Gemini %s complete (%d response chars)", label, len(full_text))

    return full_text


def _stream_llm(messages: list[dict], *, label: str = "extraction") -> str:
    """Route to Gemini or Ollama based on whether GEMINI_API_KEY is set."""
    if GEMINI_API_KEY:
        return _stream_gemini(messages, label=label)
    return _stream_ollama(messages, label=label)


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
    """Convert HTML content to PDF using pdfkit (wkhtmltopdf)."""
    import pdfkit

    logger.info("URL returned HTML; converting to PDF via pdfkit")
    return pdfkit.from_url(url, False)


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
        with httpx.Client(timeout=60, follow_redirects=True) as client:
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
    """
    import io
    try:
        import pytesseract
        from PIL import Image
    except ModuleNotFoundError:
        logger.debug("pytesseract/Pillow not installed; skipping OCR")
        return ""

    by_page: dict[int, list[str]] = {}
    for page_num, _idx, _ext, img_bytes in _iter_text_image_regions(content):
        try:
            img = Image.open(io.BytesIO(img_bytes))
            text = pytesseract.image_to_string(img).strip()
            if text:
                by_page.setdefault(page_num, []).append(text)
        except Exception as e:
            logger.debug("OCR failed on page %d image: %s", page_num, e)

    sections = [
        f"## Page {page_num} — Image-Encoded Text\n\n" + "\n".join(texts)
        for page_num, texts in sorted(by_page.items())
    ]
    combined = "\n\n".join(sections)
    if combined:
        logger.info("OCR'd image regions: %d chars across %d page section(s)",
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


def extract_pdf_text(state: AgentState) -> dict:
    """Extract embedded text from the PDF and format it as markdown.

    Also OCRs any image-encoded text regions (Tesseract) and appends that
    text to the markdown so labels/headings rendered as bitmaps aren't lost.
    """
    content = state["pdf_content"]

    markdown = _extract_markdown(content)

    # Append text recovered from image-encoded regions via Tesseract.
    image_text = _ocr_image_regions(content)
    if image_text:
        markdown = f"{markdown}\n\n{image_text}" if markdown.strip() else image_text

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

    photos_result = photos_future.result()

    return {"pdf_markdown": markdown, **photos_result}


def _is_property_photo(image_bytes: bytes) -> bool:
    """Ask the vision model whether an image is a photo of a property.

    Uses PHOTO_CLASSIFICATION_MODEL (a lightweight vision model) via Ollama
    with an image token budget of 70 to keep classification fast.

    Returns True for photographs of buildings, houses, structures, or land/lots.
    Returns False for sketches, floorplans, maps, logos, signatures, charts, etc.
    """
    import base64

    b64 = base64.b64encode(image_bytes).decode()
    payload = {
        "model": PHOTO_CLASSIFICATION_MODEL,
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
            **({"num_ctx": CLASSIFICATION_CONTEXT_LENGTH} if CLASSIFICATION_CONTEXT_LENGTH else {}),
        },
    }

    try:
        with httpx.Client(base_url=OLLAMA_BASE_URL, timeout=60) as client:
            resp = client.post("/api/chat", json=payload)
            resp.raise_for_status()
            answer = resp.json().get("message", {}).get("content", "").strip().upper()
            is_photo = answer.startswith("YES")
            logger.info("Photo classification (%s): %r -> %s",
                       PHOTO_CLASSIFICATION_MODEL, answer, is_photo)
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


def _run_extraction_llm(source_text: str) -> dict:
    """Run the extraction LLM on a body of text and return coerced property data.

    Streams the response from Ollama and emits each chunk to the DEBUG log as
    it arrives, so you can watch generation progress with --log-cli-level=DEBUG.
    """
    prompt = EXTRACTION_PROMPT.format(document_text=source_text)
    logger.info("Extracting structured data (model=%s)", EXTRACTION_MODEL)

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


RECONCILE_PROMPT = """\
A property card was parsed and produced these three monetary values:
    landvalue = {landvalue}
    imprvalue = {imprvalue}
    totalvalue = {totalvalue}

These are INCONSISTENT because landvalue + imprvalue ({sum}) does NOT equal
totalvalue ({totalvalue}).

Re-read the property card text below and return the CORRECT values. They must
satisfy: landvalue + imprvalue == totalvalue.

Labels to look for:
  - landvalue: "Land", "Land Value", "Total Land Value", "Total Assessed Land"
  - imprvalue: "Improvements", "Building", "Bldg", "Improvement Value",
               "Total Building Value", "Total Assessed Bldg", "Building Value"
               — this must roll up ALL building-related value on the parcel
               (main building plus outbuildings, extra features, other
               structures). If the card breaks these out separately (e.g.
               "Build" + "Other"), sum them into imprvalue.
  - totalvalue: "Total", "Total Value", "Total Assessed", "Total Market Value"

Return ONLY a JSON object like:
  {{"landvalue": <int>, "imprvalue": <int>, "totalvalue": <int>}}
No $ signs, no commas, no explanation.

PROPERTY CARD TEXT:
{document_text}
"""


def _reconcile_value_totals(data: dict, markdown: str) -> dict:
    """Ensure landvalue + imprvalue == totalvalue.

    1. If exactly one of the three is missing but the other two are present,
       compute the missing value arithmetically (no LLM call needed).
    2. If all three are present but inconsistent, re-query the LLM.
    3. If fewer than two are present, nothing to reconcile — return as-is.
    """
    land = data.get("landvalue")
    impr = data.get("imprvalue")
    total = data.get("totalvalue")

    have_land = isinstance(land, int)
    have_impr = isinstance(impr, int)
    have_total = isinstance(total, int)
    present = have_land + have_impr + have_total

    # Fill in a missing value when the other two are known.
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

    # All three present — check consistency.
    if land + impr == total:
        return data

    logger.warning(
        "Value totals inconsistent: land=%d + impr=%d = %d != total=%d; "
        "re-querying LLM to reconcile",
        land, impr, land + impr, total,
    )

    prompt = RECONCILE_PROMPT.format(
        landvalue=land, imprvalue=impr, totalvalue=total,
        sum=land + impr, document_text=markdown,
    )

    full_text = _stream_llm(
        [{"role": "user", "content": prompt}],
        label="reconcile",
    )
    try:
        raw = _parse_json_response(full_text)
    except json.JSONDecodeError:
        logger.warning("Reconcile LLM returned non-JSON; keeping original values")
        return data

    new = _coerce_types(raw)
    new_land = new.get("landvalue")
    new_impr = new.get("imprvalue")
    new_total = new.get("totalvalue")

    if not (isinstance(new_land, int) and isinstance(new_impr, int) and isinstance(new_total, int)):
        logger.warning(
            "Reconcile returned non-integer values land=%r impr=%r total=%r; "
            "keeping original", new_land, new_impr, new_total,
        )
        return data

    if new_land + new_impr != new_total:
        logger.warning(
            "Reconcile still inconsistent: land=%d + impr=%d = %d != total=%d; "
            "keeping original",
            new_land, new_impr, new_land + new_impr, new_total,
        )
        return data

    logger.info(
        "Reconciled values: land %d->%d, impr %d->%d, total %d->%d",
        land, new_land, impr, new_impr, total, new_total,
    )
    data["landvalue"] = new_land
    data["imprvalue"] = new_impr
    data["totalvalue"] = new_total
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
    """If the extracted parcelid is fewer than 5 characters, re-query the LLM
    with a focused prompt to find a better one."""
    pid = data.get("parcelid")
    if not isinstance(pid, str) or len(pid) >= 5:
        return data

    logger.warning(
        "parcelid %r is fewer than 5 chars; re-querying LLM for a better match",
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
    if isinstance(new_pid, str) and len(new_pid) >= 5:
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
        if GEMINI_API_KEY:
            full_text = _stream_gemini(
                [{"role": "user", "content": prompt, "images": [b64]}],
                label="photo-analysis",
            )
        else:
            # Use Ollama directly to set visual_token_budget for gemma4 models.
            payload = {
                "model": EXTRACTION_MODEL,
                "messages": [{"role": "user", "content": prompt, "images": [b64]}],
                "stream": False,
                "think": False,
                "options": {"visual_token_budget": 1120},
            }
            if EXTRACTION_CONTEXT_LENGTH:
                payload["options"]["num_ctx"] = EXTRACTION_CONTEXT_LENGTH
            with httpx.Client(base_url=OLLAMA_BASE_URL, timeout=300) as client:
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


def extract_data(state: AgentState) -> dict:
    """Extract structured property data from the markdown PDF text."""
    markdown = state.get("pdf_markdown", "") or ""
    if len(markdown.strip()) < 50:
        logger.info(
            "PDF yielded %d chars of text — skipping extraction",
            len(markdown.strip()),
        )
        return {"property_data": {}}

    data = _run_extraction_llm(markdown)
    data = _reconcile_value_totals(data, markdown)
    data = _retry_parcelid(data, markdown)
    logger.info("Extracted %d fields from PDF text", len(data))
    return {"property_data": data}