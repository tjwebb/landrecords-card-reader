# landrecords-card-reader

Extract structured property data from assessment card PDFs using LLM-powered text extraction.

Property cards (also called land cards or assessment cards) are PDF documents produced by county tax assessors.

## Installation

```bash
pip install landrecords-card-reader
```

With optional extras:

```bash
# Tesseract OCR for image-encoded text regions
pip install landrecords-card-reader[ocr]

# Everything
pip install landrecords-card-reader[all]
```

### System dependencies

- **Ollama** running locally or on a remote host with a text model loaded
  (e.g. `gemma4:26b-a4b-it-q8_0`)
- **Tesseract** (optional, for the `[ocr]` extra):
  ```bash
  sudo apt-get install tesseract-ocr
  ```

## Quick start

```python
from landrecords_card_reader import read_property_card

data, photo = read_property_card("https://example.com/card.pdf")

print(data["ownername"])    # "SMITH, JOHN A"
print(data["totalvalue"])   # 285000
print(data["parceladdr"])   # "123 MAIN ST"

# photo is raw image bytes of the first property photo, or None
if photo:
    with open("photo.jpg", "wb") as f:
        f.write(photo)
```

Use `analyze_photo=True` to send the property photo (if it exists) to the
vision model, filling in missing building details (exterior walls, roof
style, number of floors, etc.):

```python
data, photo = read_property_card(url, analyze_photo=True)
```

If you already have the PDF bytes, pass them directly to skip the download:

```python
data, photo = read_property_card(url, pdf_bytes=raw_bytes)
```

For URLs that might be HTML property report pages (e.g. Beacon, Tyler,
or other county assessment sites), use `read_property_card_from_url`.
It fetches the URL, detects whether the response is a PDF or HTML, and
converts HTML pages to PDF via pdfkit (wkhtmltopdf) automatically:

```python
from landrecords_card_reader import read_property_card_from_url

data, photo = read_property_card_from_url(
    "https://www.webgis.net/LinkedFiles/va/pulaski/pc/cards/PC17759.htm"
)
```

## CLI

```bash
landrecords-card-reader https://example.com/card.pdf --dry-run -v
```

## Configuration

Set via environment variables or a `.env` file:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `EXTRACTION_MODEL` | `gemma4:26b-a4b-it-q8_0` | Model for structured extraction |
| `PHOTO_CLASSIFICATION_MODEL` | `gemma4:e2b` | Lightweight vision model for photo classification |
| `EXTRACTION_CONTEXT_LENGTH` | (model default) | Override Ollama context window for extraction |
| `CLASSIFICATION_CONTEXT_LENGTH` | (model default) | Override Ollama context window for photo classification |
| `GEMINI_API_KEY` | (none) | Use Google Gemini API instead of Ollama |
| `GEMINI_MODEL` | `gemini-3-flash` | Gemini model name (when using Gemini) |


## Extracted fields

The extraction prompt maps over 80 property-card fields including:

- **Identity**: parcelid, taxacctnum, taxyear
- **Owner**: ownername, owneraddr, ownercity, ownerstate, ownerzip
- **Location**: parceladdr, parcelcity, parcelstate, parcelzip, legaldesc
- **Valuation**: landvalue, imprvalue, totalvalue, assessedvalue, appraisedvalue
- **Building**: yearbuilt, bldgsqft, bedrooms, fullbaths, halfbaths, bldgtype
- **Construction**: foundation, roofcover, extwall, heating, cooling
- **Sale**: saleamt, saledate
- **Zoning**: zoningcode, zoningdesc, zoningtype

## How it works

1. **Download** the PDF (or accept pre-downloaded bytes)
2. **In parallel**:
   - **Extract embedded text** via pymupdf4llm (markdown)
   - **OCR image regions** via Tesseract for text baked into raster images
   - **Extract & classify property photos** — candidate images are filtered
     by size/aspect ratio, then sent to a vision model to keep only actual
     photographs (discarding sketches, floorplans, maps, etc.)
3. **Extract structured data** by sending the markdown to an Ollama LLM
4. **Reconcile values** — verifies `landvalue + imprvalue == totalvalue` and
   retries if inconsistent

## License

MIT