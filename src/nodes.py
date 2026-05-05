import json
import logging
import os
import re
import sys
import threading
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

# matplotlib (pulled in transitively via docTR's visualization helpers)
# scans system fonts at import time and emits INFO-level noise like
# "Failed to extract font properties from /usr/share/fonts/.../NotoColorEmoji.ttf"
# whenever it encounters a font it can't parse. These are harmless on
# Linux hosts that ship emoji/unifont packages, but spam every worker
# process. Pin the font_manager logger to WARNING so they're hidden but
# real font errors still surface.
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

from .config import (
    CARD_READER_EXTRACTION_MODEL,
    CARD_READER_OLLAMA_HOST,
    CARD_READER_PHOTO_CLASSIFICATION_MODEL,
)
from .state import AgentState

# Several assessor sites (e.g. propertysearch.arlingtonva.us) reject the
# default ``python-httpx/X.Y.Z`` UA with HTTP 403. Send a plausible
# browser UA instead — almost no site filters this and we don't want
# the pipeline to fail on basic anti-bot heuristics.
_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

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
# Chunked positional output schema for the extraction LLM.
#
# Why chunks: the LLM is asked to emit ONE labelled CSV row per chunk
# instead of a single 94-cell row. A long-row positional format is
# brittle — the model has to count to 94 commas with most cells empty,
# and a single missed comma shifts every subsequent field by one
# position, corrupting the whole extraction. Chunking caps each row
# at ~10 cells so:
#
#   * The model only ever counts within a short, comprehensible block.
#   * An off-by-one in one chunk corrupts only that chunk's ~10
#     fields, not all 94.
#   * Per-chunk parsing is independent — a malformed chunk doesn't
#     poison neighbouring chunks.
#   * No per-field key strings are emitted (the original token-saving
#     reason for going to a positional format), only ~14 short chunk
#     labels per response.
#
# DO NOT reorder columns within a chunk without updating the prompt's
# CHUNKS section — the LLM has no way to recover from a column-order
# drift, and downstream code would mis-attribute fields. Adding a new
# chunk also requires updating EXTRACTION_PROMPT_CHUNK_LIST.
# ---------------------------------------------------------------------------
EXTRACTION_CHUNKS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("IDS", ("parcelid", "parcelid2", "taxacctnum", "taxyear")),
    ("ZONE", ("usecode", "usedesc", "zoningcode", "zoningdesc",
              "landusecode", "landusedesc", "taxdistrict")),
    ("BUILD", ("yearbuilt", "yearremodel", "bldgsqft", "livingarea",
               "numfloors", "numbldgs", "numunits", "bldgtype",
               "bldgquality", "bldgcondition")),
    ("ROOMS", ("bedrooms", "halfbaths", "fullbaths", "totalrooms",
               "fireplaces", "architecture")),
    ("HVAC", ("heating", "heatfuel", "cooling")),
    ("STRUCT", ("foundation", "attic", "atticsqft", "intwall", "extwall",
                "roofstyle", "roofcover", "roofheight")),
    ("EXTERIOR", ("fsagpfeet", "siding", "framing", "basementsqft",
                  "attgaragesqft", "detgaragesqft", "garagestalls")),
    ("VALUE", ("imprvalue", "landvalue", "agvalue", "totalvalue",
               "appraisedvalue", "assessedvalue", "saleamt", "saledate",
               "taxacres")),
    ("OWNER", ("ownername", "owneraddr", "ownercity", "ownerstate",
               "ownerzip")),
    ("PARCEL", ("parceladdr", "parcelcity", "parcelstate", "parcelzip",
                "situsaddress", "legaldesc")),
    ("LEGAL", ("book", "page", "block", "lot")),
    ("LOC", ("house", "category", "near", "house_number", "road", "unit",
             "level")),
    ("FEAT1", ("boatlift", "boatdock", "boathouse", "pool", "gazebo",
               "irrigation", "riprap", "solarium", "carport")),
    ("FEAT2", ("greenhouse", "openporch", "enclporch", "sauna", "wooddeck",
               "hottub", "patio", "shed", "workshop")),
)
_chunk_columns: list[str] = [c for _, cols in EXTRACTION_CHUNKS for c in cols]
assert set(_chunk_columns) == VALID_COLUMNS and len(_chunk_columns) == len(VALID_COLUMNS), (
    f"EXTRACTION_CHUNKS drift: missing={VALID_COLUMNS - set(_chunk_columns)}, "
    f"extra={set(_chunk_columns) - VALID_COLUMNS}, "
    f"duplicates={len(_chunk_columns) - len(set(_chunk_columns))}"
)
del _chunk_columns


# ---------------------------------------------------------------------------
# Extraction prompt template
# ---------------------------------------------------------------------------
EXTRACTION_PROMPT = """\
You are a property data extraction expert. Below is the raw OCR output of a
property card document — it may contain layout artifacts, line-broken
labels, and minor OCR errors. Extract all available property information
and return ONE labelled CSV row per chunk listed under "CHUNKS" below.
Each line of your output is exactly:

    LABEL:cell1,cell2,cell3,...

where ``LABEL`` is the chunk label (uppercase, exact spelling), and the
cells are the extracted values for that chunk's columns IN THE ORDER
LISTED for that chunk. You do NOT count columns across chunks — each
chunk is independent and short (≤ 10 cells), so within-chunk counting
is the only counting you have to do.

Field mapping guide:
    parcelid text -- Unique identifier for the parcel, e.g. apn, pid, pin, parcel_number, gpin.
        IMPORTANT: this is an explicitly-labeled parcel identifier (look for
        "Parcel ID", "APN", "PIN", "PID", "GPIN", "Parcel #", "Map ID").
        Usually contains 4-12 characters, sometimes with dashes or spaces.
    parcelid2 text -- Secondary identifier for the parcel, if available. e.g. lrsn, alternate_pid.
    taxacctnum text -- Tax account number associated with the parcel, e.g. tax_id, taxacctnum. Usually numeric.
    taxyear int4 -- The tax year for which the data is relevant, e.g. 2026.
    usecode text -- The land use code assigned to the parcel.
    usedesc text -- A description of the land use associated with the parcel.

    zoningcode text -- The zoning code assigned to the parcel — the
        municipal/county zoning classification, e.g. "SF-2", "R-1",
        "A1", "PUD", "C-2", "MF-3", "PUL_R4".

        Look for the value next to a "Zoning" / "Zoning Code" label.
        Prefer the LONGER, more specific code when several short
        candidates appear nearby — most jurisdictions use a letter
        prefix + numeric/dashed suffix (SF-2, R-1, MF-3). A bare
        single-letter code like "R" is almost never the full zoning
        code; on Travis County TX cards a bare "R" is the residential
        property TYPE class, listed adjacent to but distinct from the
        "Zoning: SF-2" cell. When you see "Zoning: SF-2" followed by
        another single-letter code, ALWAYS pick "SF-2".
    zoningdesc text -- A description of the zoning associated with the parcel.
    numbldgs int4 -- Number of buildings on the parcel.
    numunits int4 -- Number of units on the parcel.
    yearbuilt int4 -- The year the primary building on the parcel was built.
        IMPORTANT: when multiple year fields exist (YrBlt, YrEff, YrRmd,
        YearBuilt, EffYr, Year Effective, Remodeled), always use YrBlt /
        Year Built / Original Year. Never use YrEff, EffYr, or "effective
        year" — that is a depreciation-adjusted year, not the build year.
    bldgsqft int4 -- Total square footage of the primary building on the
        parcel. Use the LIVING / HEATED / MAIN / FINISHED area.

        Acceptable labels (in preference order when more than one is
        present): "Living Area", "Total Living Area", "Heated SF",
        "Heated Area", "Finished Area", "Main Area", "Gross Living
        Area", "Total Square Foot", "Total Area" (when it's the only
        sqft figure given for the residence — e.g. Spotsylvania County
        VA cards label the heated area as "Total Area: 1,857 sqft").

        Do NOT use "Gross Building Area", "Gross Area" (when a separate
        Living/Finished area exists), basement-only sqft, garage sqft,
        or porch/deck sqft.
    bedrooms int4 -- Number of bedrooms in the primary building on the parcel.
    halfbaths int4 -- Number of half bathrooms in the primary building on the parcel.
    fullbaths int4 -- Number of full bathrooms in the primary building on the parcel.
    imprvalue int8 -- Improvement value of the parcel — the TOTAL value
        of ALL improvements (primary building + outbuildings + extra
        features). Look for labels like "Improvement Value", "Building
        Value", "Total Improvement Value", "Total Improvement", "Imp
        Value", "Building & Extra Features", "Building Appraised",
        "Improvement Appraised", "Improv", "Bldgs+Improv". On Travis
        County TX cards the improvement value is labeled "Total
        Improvement" or, in the value-history grid, simply "Improvement"
        / "Appraised" (the per-year improvement-appraised column).

        CRITICAL: this is for IMPROVEMENTS ONLY (buildings, structures,
        features) — NEVER the land value. Do not swap with landvalue.
        It MUST be the parcel-level total improvement value, NOT a
        single sub-row. Cards often show a per-building or per-feature
        breakdown (e.g. "DWELL 200,000 / SHED 1,000 / DECK 500")
        followed by a "Total Improvement Value" row — always pick the
        total. Never pick a sub-component value when a total exists,
        and never pick a single building's value on a parcel with
        multiple buildings. The math invariant landvalue + imprvalue
        == totalvalue MUST hold; if your candidate violates it, you've
        picked the wrong row.

    landvalue int8 -- Land value of the parcel.
        Look for labels like "Land", "Land Value", "Total Land Value",
        "Total Land", "Land Market", "Land Homesite", "Land Appraised".
        On Travis County TX cards the land value is labeled "Land
        Homesite" (residential) or, in the value-history grid, "Land
        Market" — use that figure (e.g. $250,000), NOT the bigger
        "Net Appraised" / "Total" figure that combines land+improvement.

        IMPORTANT: this is for LAND ONLY — NEVER the building or
        improvement value. Do not swap with imprvalue. Use the total
        land value shown in the summary/valuation row, NOT a
        per-segment base rate or an adjacent column such as "Other",
        "Build", or "Improvement". If the card shows multiple land
        segments (e.g. BLDG SITE, OPEN, OPEN SPACE) with individual
        rates, sum them only if no total is given; otherwise use the
        total. The math invariant landvalue + imprvalue == totalvalue
        MUST hold.
    agvalue int8 -- Agricultural value of the parcel.
    totalvalue int8 -- Total assessed/appraised value of the parcel
        (labels: "Total", "Total Value", "Total Assessed Value",
        "Grand Total", "Assessed Total").

        The math invariant landvalue + imprvalue == totalvalue MUST
        hold. Before emitting, verify your three values add up. If they
        don't, you've picked a sub-row for one of them (most often
        imprvalue, where a single building's value gets used instead of
        the parcel total) — find the row that makes the math work.

        When the card shows several years of valuation history (e.g.
        "01/01/2020", "01/01/2024" rows), use the MOST RECENT year's
        total. land/impr/total must all come from the SAME assessment
        year — never mix a current land value with an older total or
        vice versa.
    taxacres float8 -- Assessed acres of the parcel.
    saleamt int8 -- Amount of the most recent for the parcel. IGNORE older sale records if multiple are present.
    saledate date -- Date of the most recent sale of the parcel. IGNORE older sale records if multiple are present.
    ownername text -- Full name of the parcel owner, as ONE value.

        CRITICAL: many property cards print the owner name across two
        or three lines on the page (e.g. last name on line 1, first +
        middle on line 2, joint owner on line 3 — "REAVES" /
        "JAMES MICHAEL" / "& JANE DOE"). These multi-line printings
        are a SINGLE owner-name value — concatenate them into one
        cell with single spaces between segments
        ("REAVES JAMES MICHAEL & JANE DOE"). Do NOT split a name
        across multiple cells. owneraddr is the MAILING ADDRESS, not
        a name fragment; if the value you'd put in owneraddr does
        not look like a street/PO-box/city address, you have split
        the name incorrectly — fold it back into ownername.

    owneraddr text -- Mailing address of the parcel owner. This is
        a STREET ADDRESS or PO BOX line — number + street name (e.g.
        "PO BOX 453", "101 SPRING HAVEN DR", "1234 ELM ST APT 5").
        It is NOT a name, name fragment, city, or state. If you do
        not see a clear street/PO-box address line for the owner,
        OMIT this field — never use it as overflow for a long owner
        name.
    ownercity text -- City of the parcel owner.
    ownerstate text -- State of the parcel owner.

    ownerzip text -- ZIP code or postal code of the parcel owner.
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
        (e.g. FORCED AIR, HEAT PUMP, BASEBOARD, RADIANT, CENTRAL,
        WARMED & COOLED AIR). This describes the delivery/system type, NOT
        the fuel.

        The label varies widely across counties. Treat ANY of these as the
        heating-system label, not just the word "Heating": "HEAT", "HEATING",
        "HEAT TYPE", "HEATING TYPE", "HEATING SYSTEM", "HEAT SYSTEM",
        "HVAC", "HVAC SYSTEM", "HEAT/AC", "H/AC", "H/A". A bare "HEAT"
        column header on a construction-detail row IS the heating label —
        do not skip it just because the word "heating" is absent.

        Expand these abbreviations to their full canonical form ONLY
        when the abbreviation appears in the value cell directly next
        to a heating-system label (one of the labels listed above):
        "CTRL"/"CENT"/"CNTL" -> CENTRAL, "FA"/"F/A" -> FORCED AIR,
        "HP"/"H/P" -> HEAT PUMP, "BB"/"BSBD" -> BASEBOARD,
        "RAD" -> RADIANT, "GFA" -> FORCED AIR (gas-forced-air system).
        If the value cell shows e.g. "HEAT CTRL", emit heating="CENTRAL".

        CRITICAL — do NOT expand bare 2-letter codes that appear
        anywhere ELSE on the card. Property cards are full of stray
        2-letter codes (district codes, map codes, class codes, parcel
        prefixes, owner-state abbreviations, MH = mobile home, etc.).
        A token like "HP", "FA", "BB" found loose in the text — not
        adjacent to a heating-system label — is NOT the heating type
        and MUST be ignored. If you cannot find a heating value
        adjacent to a heating-system label, OMIT the heating field
        entirely. Better empty than wrong.

        Do NOT confuse a fireplace appliance row ("FIREPLACE", "B-FIREPLACE
        GAS", "1-FIREPLACE") with the heating system. Those are features in
        the outbuildings/extras list, not the primary heating delivery.

        Some cards (e.g. Henry County VA InteractiveGIS) do NOT spell out a
        heating delivery label — they only show a "Heat Fuel" / "Fuel Type"
        cell paired with a "Central Air" / "Central A/C %" / "AC %" cell
        carrying a nonzero value. In that case the building has a CENTRAL
        forced-air HVAC system that does both heating and cooling — emit
        heating="CENTRAL AIR" (and cooling="CENTRAL AIR" per the cooling
        rules below). If the AC % is 0 or the Heat Fuel cell is empty,
        do NOT apply this rule — OMIT heating entirely.

    heatfuel text -- Fuel used by the PRIMARY HEATING SYSTEM. Allowed values:
        GAS, OIL, ELECTRIC, PROPANE, WOOD, SOLAR, COAL, NONE.

        The value should come from a "Heat Fuel", "Heating Fuel", or
        "Fuel Type" label in the construction-detail / HVAC section.
        Common abbreviations: "ELECT"/"ELEC" → ELECTRIC,
        "LP"/"LPG"/"PROPANE" → PROPANE, "Natural Gas"/"N.G."/"GAS"
        (when labeled as the heat fuel) → GAS, "Fuel Oil"/"OIL" → OIL,
        "WOOD" → WOOD.

        Inferred sources (when no explicit "Heat Fuel" label exists):
        - "Direct-Vented, Gas" / "Direct-Vent Gas" / "Gas Furnace" /
          "Gas-Fired Boiler" / "Gas Pack Unit" entries in the
          construction-detail or building-features list (typical of
          Spotsylvania County VA cards) describe a gas-fueled primary
          heating system — emit heatfuel=GAS.
        - "Heat Pump" entries imply ELECTRIC fuel (heat pumps are
          electrically driven). Emit heatfuel=ELECTRIC unless the card
          explicitly says otherwise.
        - "Oil Furnace" / "Oil Boiler" / "Oil-Fired" entries → OIL.
        - "Electric Baseboard" / "Electric Furnace" / "Electric Heat" →
          ELECTRIC.

        DO NOT use a fireplace's fuel as heatfuel. "B-FIREPLACE GAS",
        "GAS FIREPLACE", "1-FIREPLACE GAS", or any "FIREPLACE <fuel>"
        entry describes a fireplace appliance in the
        outbuildings/features list, NOT the primary heating system.
        Likewise, "GAS RANGE", "GAS DRYER", "GAS WATER HEATER", and other
        appliances are not the heating fuel. If the only mention of GAS
        on the card is in a fireplace or appliance line, do NOT set
        heatfuel to GAS — use whatever the "Heat Fuel" label says (often
        ELECTRIC), or omit the field entirely.

    cooling text -- Type of cooling system in the primary building on the
        parcel. Allowed values: CENTRAL AIR, HEAT PUMP, WINDOW UNIT,
        EVAPORATIVE, NONE.

        Many cards do NOT spell out the cooling type — they just record
        whether AC is present, often as a square-footage row. Treat ANY of
        these signals as cooling=CENTRAL AIR (unless an explicit type is
        given): an "Air Cond", "AC", "A/C", "Central Air", or "Cooling"
        row showing nonzero sqft / area; an "Air Condition" / "AC" line
        with a nonzero dollar value in the improvement summary; a "% AC"
        or "AC %" value greater than 0. The presence of nonzero AC area
        means the building has central AC — emit cooling=CENTRAL AIR.

        Special case: if the primary heating system is a HEAT PUMP and the
        card also indicates AC is present (any of the signals above),
        emit cooling=HEAT PUMP — the same unit provides both heating and
        cooling.

        Only emit cooling=NONE if the card EXPLICITLY says so (e.g. the
        "Air Cond" row is all zeros AND a label like "Cooling: None"
        appears, or the literal text "NONE" is the value next to a cooling
        label). If the card is silent on cooling, OMIT the field — do not
        guess.

        Sometimes labeled as "HVAC".

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

    taxdistrict text -- Name of the tax district.
);

Rules:
- Emit ONE line per chunk in the CHUNKS list below — every chunk
  must appear, in the order given, even if all of its cells are
  empty. Each line is ``LABEL:`` followed by exactly the number of
  cells the chunk specifies, separated by commas. Position WITHIN
  THE CHUNK is significant — emit a comma for every column whether
  or not you have a value for it.
- RFC 4180 quoting: wrap any cell value containing a comma or
  double-quote in double-quotes. Inside a quoted cell, escape an
  internal double-quote by doubling it. Cells without those
  characters do NOT need quoting.
- VALUES ARE SINGLE-LINE. Never put a newline inside a cell value,
  even if the source card prints the value across multiple lines.
  If a value on the source card spans multiple lines (e.g. a
  multi-line owner name like "REAVES" + "JAMES MICHAEL", a
  multi-line legal description, or a multi-line address),
  CONCATENATE the lines into a single cell with single spaces
  between segments. The line break exists in the source layout
  only — it is not part of the value.
- Leave a cell EMPTY (a bare comma) if the card does not provide a
  value. Do NOT substitute ``N/A``, ``NA``, ``UNKNOWN``, ``UNK``,
  ``null``, ``-``, ``--``, ``0``, or any other placeholder for
  missing data.
- "NONE" is a real value, NOT a placeholder. Only emit the literal
  text ``NONE`` for a cell when the card EXPLICITLY shows the text
  "NONE" (or "None") as that field's value. If a card cell is
  blank, dashed, whitespace-only, or absent, leave the CSV cell
  EMPTY — never write NONE for missing data.
- Monetary values: integer, no ``$`` sign, no thousands commas
  inside the cell. The CSV comma is only the field separator.
- Dates: YYYY-MM-DD (e.g. ``2021-05-28``).
- Numeric fields: bare numbers, no quotes, no units.
- Boolean cells: leave EMPTY when the card does not address the
  feature. Emit ``true`` only when the card explicitly indicates the
  feature is present; emit ``false`` only when the card explicitly
  indicates it is absent.
- No JSON, no markdown, no code fences, no commentary, no header
  row. Each line is ``LABEL:cells``. One line per chunk. Nothing
  else.
- It's okay if data is missing, but do not guess or fabricate data.
- parcelid, parcelid2, and taxacctnum cannot be equal to each other,
  or any other value on the card.
- heating and heatfuel cannot be equal to each other.

CHUNKS (emit one line per chunk, cells in this exact order):
{chunk_spec}

EXAMPLE OUTPUT (illustrative — use values from the actual document
below, not these):
{chunk_example}

DOCUMENT TEXT:
{document_text}"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stream_llm(messages: list[dict], *, label: str = "extraction") -> str:
    """Stream a chat completion from Ollama and return the full response text.

    When ``CARD_READER_STREAM_TO_CONSOLE=1`` is set in the environment,
    each response chunk is also written to stderr as it arrives, so the
    LLM's output can be watched live (useful for tests and interactive
    debugging). The stream-to-console flag is decoupled from the log
    level so that turning on streaming doesn't also enable verbose
    DEBUG logging from the rest of the module.
    """
    stream_to_console = (
        os.environ.get("CARD_READER_STREAM_TO_CONSOLE", "").lower()
        in ("1", "true", "yes")
    )

    payload = {
        "model": CARD_READER_EXTRACTION_MODEL,
        "messages": messages,
        "stream": True,
        "think": False,
        "num_keep": 0,
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


_CSV_PLACEHOLDER_VALUES = {
    "", "N/A", "NA", "UNKNOWN", "UNK", "NULL", "-", "--",
}


def _parse_chunked_csv(
    text: str,
    chunks: tuple[tuple[str, tuple[str, ...]], ...],
) -> dict:
    """Parse the LLM's chunked-CSV response into a dict.

    Each chunk is one line of the form ``LABEL:cell1,cell2,...``.
    Per-chunk parsing is independent — a malformed chunk only
    loses its own ≤10 fields, not subsequent chunks' fields, which
    is the whole point of chunking vs. a single 94-cell row.

    Defenses against typical model output noise:

    - Strips ``<think>...</think>`` reasoning blocks (qwen3-style)
    - Strips markdown code fences (```csv ... ``` etc.)
    - Skips lines without a colon, lines starting with JSON syntax,
      and lines whose label isn't in ``chunks``
    - Pads / truncates each chunk row to the expected cell count
      (logging WARNING) rather than dropping the whole chunk
    - First-write-wins on key collisions (across chunks)
    - csv.reader handles RFC-4180 quoting for cells that contain
      commas, double-quotes, or embedded newlines

    Raises ``ValueError`` only when the response yields zero
    recognised fields across all chunks, so the caller can retry.
    """
    import csv
    import io

    text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
    m = re.search(r"```[a-zA-Z]*\s*([\s\S]*?)```", text)
    if m:
        text = m.group(1).strip()

    chunk_map = {label: cols for label, cols in chunks}
    out: dict = {}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        if line[0] in "{}[]":
            continue

        label, _, csv_part = line.partition(":")
        label = label.strip().upper()
        if label not in chunk_map:
            continue

        cols = chunk_map[label]
        expected = len(cols)

        try:
            row = next(csv.reader(io.StringIO(csv_part), delimiter=",", quotechar='"'))
        except StopIteration:
            row = []

        if len(row) != expected:
            logger.warning(
                "Chunk %s has %d cells, expected %d — pad/truncating; "
                "fields in this chunk may be misaligned",
                label, len(row), expected,
            )
            if len(row) < expected:
                row = row + [""] * (expected - len(row))
            else:
                row = row[:expected]

        for col, val in zip(cols, row):
            # Collapse any internal whitespace — including newlines
            # the model may have leaked into a quoted multi-line
            # cell — into single spaces. Property data fields don't
            # need preserved internal whitespace, and this prevents
            # multi-line owner names / addresses / legal
            # descriptions from carrying line breaks into the
            # extracted dict.
            v = re.sub(r"\s+", " ", val.strip().strip('"')).strip()
            if not v or v.upper() in _CSV_PLACEHOLDER_VALUES:
                continue
            if col not in out:
                out[col] = v

    if not out:
        raise ValueError(
            f"No chunks recognised in extraction response: {text[:500]!r}"
        )
    return out


def _build_chunk_spec(
    chunks: tuple[tuple[str, tuple[str, ...]], ...]
) -> str:
    """Render the CHUNKS section of the extraction prompt.

    Each line names a chunk label and its column order, formatted as
    ``LABEL (N cells): col1, col2, ...``. The cell count is shown
    explicitly so the model has a single number to count toward.
    """
    return "\n".join(
        f"- {label} ({len(cols)} cells): {', '.join(cols)}"
        for label, cols in chunks
    )


def _build_chunk_example(
    chunks: tuple[tuple[str, tuple[str, ...]], ...]
) -> str:
    """Render an illustrative output example showing exact format.

    Uses the Carroll County 0000033118 card values (a real, verified
    extraction) so the example doesn't accidentally teach malformed
    cells. Any field not populated for that card is shown as an
    empty cell so the model sees the all-empty pattern (`,,,,`)
    explicitly.
    """
    sample: dict[str, str] = {
        "parcelid": "126-12-8",
        "taxacctnum": "33789",
        "yearbuilt": "2011",
        "bldgsqft": "1904",
        "bedrooms": "2",
        "imprvalue": "452700",
        "landvalue": "46600",
        "totalvalue": "499300",
        "taxacres": "3.323",
        "ownername": '"BYRD CHARLES L JR & POOLE JENNIFER A"',
        "parceladdr": "101 SPRING HAVEN DR",
        "parcelcity": "FANCY GAP",
        "parcelstate": "VA",
        "parcelzip": "24328",
        "heating": "HEAT PUMP",
        "heatfuel": "ELECTRIC",
        "cooling": "CENTRAL AIR",
        "roofstyle": "GABLE",
        "roofcover": "METAL",
        "extwall": "HARDIPLANK",
        "saledate": "2021-05-28",
        "saleamt": "345000",
    }
    lines: list[str] = []
    for label, cols in chunks:
        cells = ",".join(sample.get(c, "") for c in cols)
        lines.append(f"{label}:{cells}")
    return "\n".join(lines)


_CHUNK_SPEC = _build_chunk_spec(EXTRACTION_CHUNKS)
_CHUNK_EXAMPLE = _build_chunk_example(EXTRACTION_CHUNKS)


# Placeholder strings the LLM occasionally emits despite being told to omit
# missing fields. These are NEVER kept; "NONE" is allowed only if the OCR
# text explicitly contains it (verified separately by _drop_unverified_none).
_PLACEHOLDER_STRINGS = {"N/A", "NA", "UNKNOWN", "UNK", "NULL", "-", "--"}


def _coerce_types(data: dict) -> dict:
    """Coerce extracted values to their expected Python types."""
    int_fields = {
        "taxyear", "numbldgs", "numunits", "yearbuilt", "yearremodel",
        "numfloors", "bldgsqft", "livingarea", "bedrooms", "halfbaths",
        "fullbaths", "totalrooms", "fireplaces", "atticsqft", "roofheight",
        "fsagpfeet", "basementsqft", "attgaragesqft", "detgaragesqft",
        "garagestalls",
    }
    bigint_fields = {
        "imprvalue", "landvalue", "agvalue", "totalvalue", "saleamt",
        "appraisedvalue", "assessedvalue",
    }
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
                    if upper in _PLACEHOLDER_STRINGS:
                        # LLM-emitted placeholder for a missing value —
                        # drop it; absence is more honest than a fake.
                        continue
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


_SCAN_IMG_EXT_RE = re.compile(r"\.(jpe?g|png|tiff?)(\?|$)", re.IGNORECASE)
# Filename / URL substrings that mark an <img> as branding/chrome (a
# county logo, navigation icon, layout spacer) rather than a property-
# card scan. Matching here removes the IMG from scan-candidate
# consideration entirely — important for rich-data HTML pages
# (e.g. Arlington's property search) that embed a county logo: without
# this filter the logo would falsely qualify the page as "image-only"
# and route it through the scan-bypass instead of wkhtmltopdf.
_SCAN_IMG_BLACKLIST_RE = re.compile(
    r"logo|icon|banner|spacer|pixel|favicon|sprite|background|placeholder",
    re.IGNORECASE,
)
# Image-scan pages have at most a few hundred chars of header/disclaimer
# chrome after stripping <script>/<style>/comments; rich assessor data
# pages have thousands. 1000 keeps the known scan card (Richmond, ~860
# chars of boilerplate) within bounds while excluding pages with even
# modest amounts of property data text (Arlington, ~1467 chars).
_SCAN_IMG_TEXT_THRESHOLD = 1000


def _extract_scan_image_urls(html: bytes, base_url: str) -> list[str]:
    """Return resolved image URLs if the HTML is an image-only scan card.

    Some assessor sites (e.g. richmondcountypropertycards.com) serve the
    property card as an HTML wrapper around one or more JPEG scans of
    the paper card. Routing those through wkhtmltopdf is destructive:
    wkhtmltopdf re-encodes the JPEG into the rendered PDF and pypdfium2
    later re-rasterises that PDF for OCR — two lossy passes that wipe
    out thin glyphs (e.g. the "1" in "A-1" zoning codes).

    When this heuristic fires, the caller should fetch the JPEGs
    directly and build a PDF that embeds them at native resolution.

    Heuristic — return URLs only if BOTH:
    1. There is at least one ``<img>`` whose src looks like a raster
       photo (.jpg/.jpeg/.png/.tif/.tiff). Vector logos (.svg, .gif)
       and tracking pixels are ignored.
    2. After stripping all tags, the HTML contains less than
       ``_SCAN_IMG_TEXT_THRESHOLD`` non-whitespace characters of body
       text. Any HTML with substantive text content (real assessor
       data tables) falls through to wkhtmltopdf.

    Conservative by design: any uncertainty falls through to the
    existing wkhtmltopdf path.
    """
    from urllib.parse import urljoin

    html_str = html.decode("utf-8", errors="replace")

    img_srcs = re.findall(
        r'<img[^>]+src\s*=\s*["\']([^"\']+)["\']',
        html_str,
        flags=re.IGNORECASE,
    )
    scan_srcs = [
        src for src in img_srcs
        if _SCAN_IMG_EXT_RE.search(src)
        and not src.startswith("data:")
        and not _SCAN_IMG_BLACKLIST_RE.search(src)
    ]
    if not scan_srcs:
        return []

    # Strip <script>, <style>, and HTML comments before counting body
    # text — inline CSS rules and JS blobs are not "content" and would
    # otherwise push every page over the threshold.
    body = re.sub(r"<script\b[^>]*>.*?</script>", " ", html_str, flags=re.IGNORECASE | re.DOTALL)
    body = re.sub(r"<style\b[^>]*>.*?</style>", " ", body, flags=re.IGNORECASE | re.DOTALL)
    body = re.sub(r"<!--.*?-->", " ", body, flags=re.DOTALL)
    text_only = re.sub(r"<[^>]+>", " ", body)
    text_only = re.sub(r"\s+", " ", text_only).strip()
    if len(text_only) >= _SCAN_IMG_TEXT_THRESHOLD:
        return []

    return [urljoin(base_url, src) for src in scan_srcs]


def _images_to_pdf(image_urls: list[str]) -> bytes:
    """Fetch raster scans and assemble a PDF that embeds them losslessly.

    Each image becomes one PDF page sized to match the image's native
    pixel dimensions. PIL writes JPEGs into the PDF stream verbatim
    (no re-encoding), so the downstream pypdfium2 render pass is a
    pure upsample — no signal loss vs. the source scan.
    """
    import io

    from PIL import Image

    images: list[Image.Image] = []
    with httpx.Client(
        timeout=60,
        follow_redirects=True,
        verify=False,
        headers={"user-agent": _BROWSER_UA},
    ) as client:
        for url in image_urls:
            resp = client.get(url)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content))
            # PDF can't carry alpha; flatten to RGB if needed.
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            images.append(img)

    if not images:
        raise RuntimeError("No images fetched from scan URLs")

    buf = io.BytesIO()
    head, *rest = images
    head.save(buf, format="PDF", save_all=True, append_images=rest)
    return buf.getvalue()


def _html_to_pdf(html: bytes, url: str) -> bytes:
    """Convert HTML content to PDF.

    Two paths:

    1. **Image-scan HTML** (assessor sites that wrap JPEGs of the paper
       card in a thin HTML shell). Detected via
       :func:`_extract_scan_image_urls`; the source images are fetched
       directly and embedded into a PDF at native resolution. This
       avoids the wkhtmltopdf + pypdfium2 double-rasterisation that
       drops thin glyphs in OCR.

    2. **Rich HTML** (server-rendered assessor reports with real text
       and tables). Sent to wkhtmltopdf via stdin so it doesn't
       re-download the page. A ``<base href="...">`` tag is injected
       so wkhtmltopdf can resolve relative/protocol-relative URLs for
       sub-resources; any sub-resources that still fail are ignored
       via ``load-error-handling``.
    """
    scan_urls = _extract_scan_image_urls(html, url)
    if scan_urls:
        logger.info(
            "HTML wraps %d scan image(s); embedding directly into PDF "
            "(bypassing wkhtmltopdf to preserve OCR fidelity)",
            len(scan_urls),
        )
        return _images_to_pdf(scan_urls)

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
        # JavaScript on assessor sites tends to HIDE content rather than
        # add it: jQuery UI tabs collapse extra panels (Stafford County
        # VA's "Floor Areas" / "Exterior Features" tabs), inline-style
        # injectors set display:none on detail rows, etc. Property data
        # is universally server-rendered into the initial HTML, so
        # disabling JS reliably exposes more content and never hides
        # any. Also speeds up conversion (no javascript-delay wait).
        "disable-javascript": "",
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
        with httpx.Client(
            timeout=60,
            follow_redirects=True,
            verify=False,
            headers={"user-agent": _BROWSER_UA},
        ) as client:
            resp = client.get(url)
            resp.raise_for_status()
        content = resp.content

    if not _is_pdf(content):
        content = _html_to_pdf(content, state["pdf_url"])

    return {"pdf_content": content}


# Cached docTR predictor. Model weights (~100MB+) are downloaded to
# ~/.cache/doctr/ on first call and re-used for subsequent OCR runs in the
# same process. Loading is deferred until first use so importing this
# module stays fast and doesn't pull torch into RAM unnecessarily.
#
# The lock guards the lazy initialiser against the classic check-then-act
# race: without it, N threads calling _get_ocr_model() concurrently before
# the first model load completes will all see _OCR_MODEL is None, all
# call doctr's downloader, and clobber each other writing the same
# weights file (manifests as: "corrupted download, the hash of ... does
# not match its expected value"). With the lock, only the first thread
# downloads + initialises; the rest wait, observe the populated
# _OCR_MODEL on the inner check, and return the same predictor.
_OCR_MODEL = None
_OCR_MODEL_LOCK = threading.Lock()

# OCR architecture — picking the highest-accuracy models docTR ships.
# Detection: db_resnet50 (top of docTR's published recall/precision on
# the FUNSD/CORD benchmarks for detection).
# Recognition: parseq (transformer-based, leads docTR's published word
# accuracy ~89% vs. ~86% for crnn_vgg16_bn). It's slower and slightly
# heavier than CRNN but the gain is meaningful on the dense
# small-glyph table cells found on assessor cards, where a single
# missed character (e.g. WYTHEVILLE -> WYTHVILLE) can flip a field.
# Override either via CARD_READER_OCR_DET_ARCH / CARD_READER_OCR_RECO_ARCH.
_OCR_DET_ARCH = os.getenv("CARD_READER_OCR_DET_ARCH", "db_resnet50")
_OCR_RECO_ARCH = os.getenv("CARD_READER_OCR_RECO_ARCH", "parseq")

# pypdfium2 render scale used by DocumentFile.from_pdf. Default 2 (~144 DPI)
# is too low for the dense table grids on county property cards — adjacent
# cell glyphs touch in the rasterised image and the recognition model
# emits spurious characters ("511E BROWNTOWNF" instead of "511 BROWNTOWN").
# Scale 4 ≈ 288 DPI is a good quality/cost balance; bump higher for cards
# with very small font sizes.
_OCR_RENDER_SCALE = float(os.getenv("CARD_READER_OCR_RENDER_SCALE", "4"))


def _doctr_cache_dir() -> str:
    """Return the docTR weights cache directory (creating it if missing)."""
    cache_dir = os.environ.get(
        "DOCTR_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "doctr"),
    )
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _get_ocr_model():
    """Return the lazy-loaded docTR OCR predictor (thread- and process-safe).

    Layered locking:

    * **Process-local** ``_OCR_MODEL_LOCK`` (a ``threading.Lock``) makes
      the in-process initialiser idempotent. Without it, N threads
      calling this concurrently would all see ``_OCR_MODEL is None`` and
      all enter the load path.

    * **Cross-process** advisory file lock (``fcntl.flock`` on a
      sentinel inside the docTR cache dir) makes the initial weight
      download safe across separate Python processes that share a
      filesystem — typical in Ray clusters where N workers in the same
      cluster, all importing this module concurrently, would otherwise
      all call docTR's downloader, all write to the same .pt path, and
      all corrupt each other (manifests as: "corrupted download, the
      hash of ... does not match"). The first process acquires the
      lock, downloads, releases. The next process acquires, finds the
      file already on disk, and docTR's internal hash check makes its
      download a no-op.

    Built with ``assume_straight_pages=True`` to skip the rotation-
    correction pass — county cards are always upright, so paying for
    skew estimation on every page is pure overhead. The predictor is
    moved to CUDA when available; PyTorch falls back to CPU silently if
    the runtime has no CUDA build or no usable GPU.
    """
    global _OCR_MODEL
    # Fast path: model already loaded, no lock needed.
    if _OCR_MODEL is not None:
        return _OCR_MODEL

    with _OCR_MODEL_LOCK:
        # Re-check under the in-process lock — another thread may have
        # finished loading while we were blocked acquiring it.
        if _OCR_MODEL is not None:
            return _OCR_MODEL

        # Cross-process file lock around the download. Best-effort:
        # fcntl is Unix-only, so on Windows we skip the file lock (the
        # in-process lock above is still active). Lock is held for the
        # full ocr_predictor() call so docTR sees a stable filesystem
        # while it's verifying / downloading weights.
        cache_dir = _doctr_cache_dir()
        lock_path = os.path.join(cache_dir, ".pcr-loader.lock")
        lock_handle = None
        try:
            import fcntl
            lock_handle = open(lock_path, "w")
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError) as e:
            logger.debug("fcntl file lock unavailable (%s); using in-process lock only", e)
            if lock_handle is not None:
                lock_handle.close()
                lock_handle = None

        try:
            from doctr.models import ocr_predictor
            try:
                import torch
                use_cuda = torch.cuda.is_available()
            except ImportError:
                use_cuda = False

            device_label = "GPU/CUDA" if use_cuda else "CPU"
            logger.info(
                "Loading docTR OCR model on %s (det=%s, reco=%s) "
                "(first call: weight download + init)",
                device_label, _OCR_DET_ARCH, _OCR_RECO_ARCH,
            )
            model = ocr_predictor(
                det_arch=_OCR_DET_ARCH,
                reco_arch=_OCR_RECO_ARCH,
                pretrained=True,
                assume_straight_pages=True,
            )
            if use_cuda:
                try:
                    model = model.cuda()
                    logger.info("docTR predictor moved to CUDA")
                except Exception as e:
                    logger.warning(
                        "Failed to move docTR predictor to CUDA (%s); staying on CPU", e,
                    )
            _OCR_MODEL = model
            return _OCR_MODEL
        finally:
            if lock_handle is not None:
                try:
                    import fcntl
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass
                lock_handle.close()


# Per-page threshold: a page that returns at least this many non-whitespace
# chars from pymupdf's get_text() AND passes the informativeness check is
# treated as having a real text layer and is read directly (no OCR).
# Below the chars threshold the page is assumed scanned/image-only.
# 100 chars is loose enough that a minimal cover-page text layer counts as
# "has text" but tight enough that a few stray glyphs from a scanned page
# don't suppress OCR.
_NATIVE_TEXT_MIN_CHARS = 100

# Minimum fraction of whitespace-split tokens that must contain a digit.
# Defends against degenerate text layers — e.g. Henry County VA
# InteractiveGIS cards place label boxes and value boxes as independent
# text objects, and pymupdf's reader-flow extraction silently drops
# entire value sections (HVAC, sale price, etc.). The result is text
# that's almost all label words with very few values. A real
# assessor-card text layer is dense with numeric tokens (account
# numbers, dates, $ values, acreage, sqft); the Henry-style degenerate
# one falls under ~15%. Threshold of 20% cleanly separates the cases
# observed in fixtures (Henry: 15% degenerate; Wythe: 29% good;
# Radford: 35-37% good).
#
# This check is necessary but not sufficient: a page with a healthy
# digit-token ratio whose VALUES were also dropped by pymupdf would slip
# through. So far no fixture exhibits that — degenerate layouts tend to
# drop both labels and values together.
_NATIVE_TEXT_MIN_DIGIT_TOKEN_RATIO = 0.20


def _is_native_text_useful(text: str) -> bool:
    """Decide whether a page's native text layer carries enough values to
    be used in place of OCR. See ``_NATIVE_TEXT_MIN_CHARS`` and
    ``_NATIVE_TEXT_MIN_DIGIT_TOKEN_RATIO`` for the thresholds.
    """
    if len(text) < _NATIVE_TEXT_MIN_CHARS:
        return False
    tokens = text.split()
    if not tokens:
        return False
    digit_tokens = sum(1 for t in tokens if any(c.isdigit() for c in t))
    return (digit_tokens / len(tokens)) >= _NATIVE_TEXT_MIN_DIGIT_TOKEN_RATIO


def _ocr_page_images(images: list[bytes]) -> list[str]:
    """Run docTR over a list of pre-rendered page images (PNG bytes) and
    return one text block per image, in the same order."""
    from doctr.io import DocumentFile

    if not images:
        return []
    doc_for_ocr = DocumentFile.from_images(images)
    model = _get_ocr_model()
    result = model(doc_for_ocr)

    out: list[str] = []
    for page in result.pages:
        page_text = "\n".join(
            " ".join(word.value for word in line.words)
            for block in page.blocks
            for line in block.lines
            if line.words
        ).strip()
        out.append(page_text)
    return out


def _pdfium_render_worker(
    content: bytes, page_indices: list[int], scale: float
) -> list[bytes]:
    """Subprocess entry point: render PDF pages to PNG bytes via pypdfium2.

    Defined at module level so ``multiprocessing`` (spawn context) can
    pickle and import it in the child interpreter. Caller is
    :func:`_render_pages_with_pdfium`; do not invoke directly.
    """
    import io

    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(content)
    images: list[bytes] = []
    try:
        for i in page_indices:
            pil_image = pdf[i].render(scale=scale).to_pil()
            buf = io.BytesIO()
            pil_image.save(buf, format="PNG")
            images.append(buf.getvalue())
    finally:
        pdf.close()
    return images


_INLINE_CLIP_MIN_DPI = 150
_INLINE_CLIP_MAX_DPI = 600


def _pymupdf_clip_worker(
    content: bytes,
    clips: list[tuple[int, tuple[float, float, float, float], int]],
) -> list[bytes]:
    """Subprocess entry point: rasterise per-bbox PDF regions to JPEG bytes.

    ``clips`` is a list of ``(page_index, (x0, y0, x1, y1), native_w_px)``
    tuples in PDF user-space points. Each region is rendered to JPEG
    at the DPI required to preserve the source image's native pixel
    width within the bbox, clamped to
    ``[_INLINE_CLIP_MIN_DPI, _INLINE_CLIP_MAX_DPI]``.

    Why per-clip DPI: an inline image embedded at small page-size but
    high source resolution (typical for property-card photos: 2250 px
    in a 283 pt = 3.9 inch wide region) would be downsampled ~4× at
    a fixed 150 DPI. The vision-model classifier rejects the
    downsampled images. Computing DPI from native_w_px keeps the
    render at native resolution.

    Defined at module level so ``multiprocessing`` (spawn context) can
    pickle and import it in the child interpreter. Caller is
    :func:`_clip_pdf_regions`; do not invoke directly.

    Why subprocess: pymupdf's mupdf backend has segfaulted reproducibly
    on certain wkhtmltopdf-generated PDFs when calling get_pixmap with
    a clip rect. A SIGSEGV in C-extension code is fatal to the whole
    process, but in a subprocess it's fatal only to the subprocess —
    the parent gets a clean BrokenProcessPool exception and can carry
    on with no inline photos.
    """
    import io
    import pymupdf as fitz

    doc = fitz.open(stream=content, filetype="pdf")
    out: list[bytes] = []
    try:
        for page_idx, bbox, native_w_px in clips:
            x0, _, x1, _ = bbox
            bbox_w_pt = max(x1 - x0, 1.0)
            needed = int(72.0 * native_w_px / bbox_w_pt) if native_w_px > 0 else _INLINE_CLIP_MIN_DPI
            dpi = max(_INLINE_CLIP_MIN_DPI, min(needed, _INLINE_CLIP_MAX_DPI))
            page = doc[page_idx]
            pix = page.get_pixmap(clip=fitz.Rect(*bbox), dpi=dpi)
            buf = io.BytesIO()
            buf.write(pix.tobytes("jpeg"))
            out.append(buf.getvalue())
    finally:
        doc.close()
    return out


def _clip_pdf_regions(
    content: bytes,
    clips: list[tuple[int, tuple[float, float, float, float], int]],
) -> list[bytes]:
    """Rasterise per-bbox PDF regions to JPEG bytes in an isolated subprocess.

    Returns one JPEG per clip in input order. Returns an empty list if
    the subprocess crashes (segfault, broken pool, etc.) — callers
    should treat that as "no inline photos available" rather than a
    hard failure.

    Spawn context (not fork) for the same reason as
    :func:`_render_pages_with_pdfium`: avoid copy-on-write inheritance
    of docTR/PyTorch state in the child.
    """
    if not clips:
        return []

    import multiprocessing
    from concurrent.futures import BrokenExecutor, ProcessPoolExecutor

    ctx = multiprocessing.get_context("spawn")
    try:
        with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool:
            future = pool.submit(_pymupdf_clip_worker, content, clips)
            return future.result()
    except (BrokenExecutor, OSError) as e:
        logger.warning(
            "Inline-image rasterisation subprocess crashed (%s); "
            "skipping %d inline image(s)", e, len(clips),
        )
        return []


def _render_pages_with_pdfium(content: bytes, page_indices: list[int]) -> list[bytes]:
    """Render specific PDF pages to PNG bytes in an isolated subprocess.

    Two reasons rendering runs out-of-process:

    1. **Thread safety.** PDFium's C API is not thread-safe, and the
       pipeline runs ``_ocr_pdf`` and ``extract_property_photos``
       concurrently from a ThreadPoolExecutor — concurrent in-process
       PDFium calls can race and corrupt internal state. A fresh
       subprocess is the only PDFium caller in its interpreter, so
       there's nothing to race with.

    2. **Segfault containment.** mupdf (pymupdf's backend) and
       occasionally PDFium itself have segfaulted on wkhtmltopdf-
       generated PDFs in certain Linux environments. A SIGSEGV in
       Python C code is fatal to the whole process — but in a
       subprocess it's fatal only to the subprocess, and the parent
       gets a clean ``BrokenProcessPool`` / ``Process`` exit code we
       can surface as a normal Python exception.

    ``multiprocessing.get_context("spawn")`` is used (not the default
    ``fork`` on Linux) so the child inherits no copy-on-write state
    from the parent — this avoids the well-known fork-after-threads
    deadlocks docTR/PyTorch can trigger.
    """
    if not page_indices:
        return []

    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool:
        future = pool.submit(
            _pdfium_render_worker, content, page_indices, _OCR_RENDER_SCALE,
        )
        return future.result()


def _ocr_pdf(content: bytes) -> str:
    """Extract per-page text from the PDF, using OCR only where required.

    For each page, pymupdf's ``get_text()`` is tried first. When the page
    has a usable native text layer (>= _NATIVE_TEXT_MIN_CHARS of non-
    whitespace text), that text is used verbatim — it's faster, lossless,
    and free of OCR errors on small glyphs.

    Pages without a usable text layer (typical of scanned cards or pages
    where the text is baked into a raster image) are rendered to PNG at
    ``CARD_READER_OCR_RENDER_SCALE`` (~288 DPI by default) via
    pypdfium2 and run through docTR. Only those pages pay the OCR cost;
    pages with a native text layer skip the model entirely.

    Why pypdfium2 for rendering when pymupdf is already open: pymupdf's
    mupdf backend has segfaulted reproducibly on wkhtmltopdf-generated
    PDFs in certain Linux environments. pypdfium2 (Chromium's PDFium)
    handles those PDFs cleanly and is the same renderer docTR uses
    natively, so output quality is identical.

    Returned text follows original page order, with ``=== Page N ===``
    headers when the document has more than one populated page.
    """
    import pymupdf as fitz

    doc = fitz.open(stream=content, filetype="pdf")
    page_texts: list[str | None] = []
    pages_needing_ocr: list[int] = []

    try:
        for i in range(len(doc)):
            native = (doc[i].get_text() or "").strip()
            if _is_native_text_useful(native):
                page_texts.append(native)
            else:
                page_texts.append(None)
                pages_needing_ocr.append(i)
    finally:
        doc.close()

    rendered_images: list[bytes] = (
        _render_pages_with_pdfium(content, pages_needing_ocr)
        if pages_needing_ocr
        else []
    )

    native_count = len(page_texts) - len(pages_needing_ocr)
    if pages_needing_ocr:
        logger.info(
            "PDF text: %d/%d page(s) have a native text layer; OCRing the "
            "remaining %d page(s) without one",
            native_count, len(page_texts), len(pages_needing_ocr),
        )
        ocr_texts = _ocr_page_images(rendered_images)
        for idx, text in zip(pages_needing_ocr, ocr_texts):
            page_texts[idx] = text
    else:
        logger.info(
            "PDF text: native text layer found on all %d page(s); skipping OCR",
            len(page_texts),
        )

    sections: list[str] = []
    total_chars = 0
    populated = sum(1 for t in page_texts if t)
    for i, text in enumerate(page_texts):
        if not text:
            continue
        total_chars += len(text)
        if populated > 1:
            sections.append(f"=== Page {i + 1} ===\n{text}")
        else:
            sections.append(text)

    combined = "\n\n".join(sections)
    if combined:
        logger.info(
            "PDF text total: %d chars across %d page(s)",
            total_chars, populated,
        )
    return combined


def extract_pdf_text(state: AgentState) -> dict:
    """Extract text from the PDF, preferring the native text layer.

    Pages with a usable native text layer are read directly via pymupdf;
    only pages without one (scanned / image-only) are rasterised and
    OCR'd via docTR. The combined per-page text is fed verbatim to the
    extraction LLM — no markdown conversion, no layout reconstruction,
    no table parsing.
    """
    content = state["pdf_content"]
    text = _ocr_pdf(content)
    return {"pdf_text": text}


def extract_pdf_and_photos(state: AgentState) -> dict:
    """Run page-level OCR and photo classification in parallel.

    OCR is GPU/CPU-bound (docTR / PyTorch); photo classification is I/O
    bound (Ollama HTTP). Both open the PDF independently so they don't
    contend for PyMuPDF state.
    """
    from concurrent.futures import ThreadPoolExecutor

    content = state["pdf_content"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        text_future = pool.submit(_ocr_pdf, content)
        photos_future = pool.submit(extract_property_photos, state)

    text = text_future.result()
    photos_result = photos_future.result()

    return {"pdf_text": text, **photos_result}


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
        "num_keep": 0,
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

    content = state["pdf_content"]
    candidates: list[dict] = []
    seen_xrefs: set[int] = set()
    inline_clips: list[tuple[int, tuple[float, float, float, float], int]] = []
    inline_meta: list[dict] = []

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
                    # Inline image (xref == 0). Rasterising via
                    # page.get_pixmap(clip=...) has segfaulted in mupdf
                    # on wkhtmltopdf-generated PDFs, so we batch the
                    # clips and run them in an isolated subprocess
                    # below — a crash there is contained and we just
                    # lose the inline photos for that PDF.
                    #
                    # The native pixel width (w) is passed through so
                    # the worker can pick a DPI that preserves the
                    # source image's resolution within the bbox —
                    # property-card photos are routinely embedded at
                    # 2000+ px in a 3-4 inch region, and a fixed 150
                    # DPI render would downsample them ~4×, which the
                    # vision classifier then rejects.
                    bbox = info.get("bbox")
                    if not bbox:
                        continue
                    inline_clips.append((page_num, tuple(bbox), w))
                    inline_meta.append({
                        "page": page_num + 1,
                        "width": w,
                        "height": h,
                        "ext": "jpeg",
                    })
    finally:
        doc.close()

    if inline_clips:
        rendered = _clip_pdf_regions(content, inline_clips)
        for meta, jpeg in zip(inline_meta, rendered):
            candidates.append({**meta, "bytes": jpeg})

    logger.info("Found %d candidate image(s), classifying with vision model", len(candidates))

    # Classify candidates in parallel — each call is a single Ollama HTTP
    # round-trip, which is I/O bound. Up to PHOTO_CLASSIFY_CONCURRENCY
    # requests overlap; Ollama serializes GPU work internally so this
    # doesn't help inference time, but it does eliminate the per-request
    # network/scheduling overhead between candidates. Order is preserved
    # via map() so the survivors line up with the original candidate
    # list.
    if candidates:
        from concurrent.futures import ThreadPoolExecutor

        PHOTO_CLASSIFY_CONCURRENCY = min(8, len(candidates))
        with ThreadPoolExecutor(max_workers=PHOTO_CLASSIFY_CONCURRENCY) as pool:
            verdicts = list(pool.map(
                _is_property_photo, (c["bytes"] for c in candidates),
            ))
        photos = [c for c, keep in zip(candidates, verdicts) if keep]
    else:
        photos = []

    logger.info("Kept %d/%d image(s) as property photos", len(photos), len(candidates))
    return {"property_photos": photos}


def _run_extraction_llm(source_text: str, context: str | None = None) -> dict:
    """Run the extraction LLM on a body of text and return coerced property data.

    Streams the response from Ollama and emits each chunk to the DEBUG log as
    it arrives, so you can watch generation progress with --log-cli-level=DEBUG.

    If ``context`` is provided, it is appended to the extraction prompt as
    additional caller-supplied instructions.
    """
    prompt = EXTRACTION_PROMPT.format(
        document_text=source_text,
        chunk_spec=_CHUNK_SPEC,
        chunk_example=_CHUNK_EXAMPLE,
    )
    if context and context.strip():
        prompt = f"{prompt}\n\nAdditional instructions:\n{context.strip()}"
    logger.info("Extracting structured data (model=%s)", CARD_READER_EXTRACTION_MODEL)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise data extraction assistant. Return "
                "ONLY one labelled CSV row per chunk as instructed — "
                "no JSON, no markdown, no code fences, no prose."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    full_text = _stream_llm(messages, label="extraction")
    raw = _parse_chunked_csv(full_text, EXTRACTION_CHUNKS)
    return _coerce_types(raw)


def _reconcile_value_totals(data: dict) -> dict:
    """Enforce landvalue + imprvalue == totalvalue.

    If exactly one of the three is missing but the other two are
    present, compute the missing value arithmetically.

    If all three are present but inconsistent, attempt to identify and
    correct the wrong one. The dominant LLM failure mode is picking a
    sub-row for imprvalue (e.g. a single building's value on a parcel
    with multiple buildings), producing a small impr that fails the
    sum check. Heuristics:

    - When ``land + impr < total``: one of land/impr is a sub-row.
      The smaller component is almost always the mistake — re-derive
      it from ``total - other``.
    - When ``land + impr > total``: total is likely from a different
      (older) assessment year, or one component double-counts an
      outbuilding. Trust the components and recompute total.

    Both cases log a warning so callers know reconciliation happened.
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
            logger.info("Computed missing totalvalue: %d + %d = %d",
                        land, impr, data["totalvalue"])
        elif not have_impr:
            data["imprvalue"] = total - land
            logger.info("Computed missing imprvalue: %d - %d = %d",
                        total, land, data["imprvalue"])
        elif not have_land:
            data["landvalue"] = total - impr
            logger.info("Computed missing landvalue: %d - %d = %d",
                        total, impr, data["landvalue"])
        return data

    if present < 3:
        return data

    if land + impr == total:
        return data

    if land + impr < total:
        # One component is a sub-row. The smaller of the two is the
        # mistake — re-derive from total minus the trusted (larger) one.
        if impr <= land:
            new_impr = total - land
            logger.warning(
                "Value mismatch: land=%d + impr=%d = %d != total=%d; "
                "imprvalue (%d) looks like a sub-row, recomputing as %d",
                land, impr, land + impr, total, impr, new_impr,
            )
            data["imprvalue"] = new_impr
        else:
            new_land = total - impr
            logger.warning(
                "Value mismatch: land=%d + impr=%d = %d != total=%d; "
                "landvalue (%d) looks like a sub-row, recomputing as %d",
                land, impr, land + impr, total, land, new_land,
            )
            data["landvalue"] = new_land
    else:
        # land + impr overshoots total — total is likely stale (wrong
        # assessment year) or land/impr double-counts. Recompute total
        # from the components, which usually come from the current year.
        new_total = land + impr
        logger.warning(
            "Value mismatch: land=%d + impr=%d = %d > total=%d; "
            "totalvalue likely from an older year, recomputing as %d",
            land, impr, land + impr, total, new_total,
        )
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


def _retry_parcelid(data: dict, text: str) -> dict:
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
            parcelid=pid, document_text=text,
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
        payload = {
            "model": CARD_READER_EXTRACTION_MODEL,
            "messages": [{"role": "user", "content": prompt, "images": [b64]}],
            "stream": False,
            "think": False,
            "num_keep": 0
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


# Anchor for a "Heat Fuel" / "Heating Fuel" / "Fuel Type" label only —
# NOT a bare "fuel" (which would also match "B-FIREPLACE GAS" and similar
# fireplace/appliance entries on the card). After matching, the post-pass
# scans a generous window (~500 chars) for any canonical fuel token —
# grid-style cards (Richmond County VA) print the label row and value row
# on separate lines AND docTR's reading-order OCR can interleave values
# from right-side image regions, so the corresponding fuel value can be
# 5-15 lines below the label rather than immediately after it.
_HEATFUEL_LABEL_RE = re.compile(
    r"\b(?:heat\s*fuel|heating\s*fuel|fuel\s*type)\b",
    re.IGNORECASE,
)
_HEATFUEL_VALUE_WINDOW = 500


_CANONICAL_HEATFUEL_VALUES = {
    "GAS", "OIL", "ELECTRIC", "PROPANE", "WOOD", "SOLAR", "COAL", "NONE",
}


def _post_extract_heatfuel(data: dict, text: str) -> dict:
    """Reconcile heatfuel against the explicit 'Heat Fuel' label in the text.

    The card's "Heat Fuel" / "Fuel Type" cell is the ground truth for this
    field. The LLM occasionally:
      - drops the field entirely (silent miss), or
      - emits the WRONG fuel because something like "B-FIREPLACE GAS"
        appears elsewhere on the card and the model conflates a fireplace
        appliance with the heating system.

    Recovery (LLM emitted no heatfuel): scan a generous window after the
    fuel label so grid-style cards that print the label row and value row
    on separate lines (Richmond County VA) can still be read.

    Override (LLM emitted a non-canonical fuel like "ELECTRIC C-AIR"):
    correct it. If the LLM already has a canonical fuel value, leave it
    alone — the LLM has full document context and the wide window risks
    a false positive.
    """
    existing = data.get("heatfuel")
    if isinstance(existing, str) and existing.strip().upper() in _CANONICAL_HEATFUEL_VALUES:
        return data

    for m in _HEATFUEL_LABEL_RE.finditer(text):
        window = text[m.end() : m.end() + _HEATFUEL_VALUE_WINDOW].upper()
        for pattern, canonical in _HEATFUEL_TOKEN_MAP:
            if re.search(rf"\b{pattern}\b", window):
                if existing:
                    logger.warning(
                        "heatfuel overridden: LLM said %r, %s label says %s",
                        existing, m.group(0).strip(), canonical,
                    )
                else:
                    logger.info(
                        "heatfuel recovered from %s label: %s",
                        m.group(0).strip(), canonical,
                    )
                data["heatfuel"] = canonical
                return data
    return data


# Roof-style and roof-cover keywords used by `_post_extract_roof`. These
# are the values county cards print under "Roof Type", "Roof Style",
# "Roof Material", or "Roof Type/Material" labels.
_ROOF_STYLE_TOKENS: tuple[tuple[str, str], ...] = (
    (r"GABLE", "GABLE"),
    (r"HIP", "HIP"),
    (r"GAMBREL", "GAMBREL"),
    (r"MANSARD", "MANSARD"),
    (r"FLAT", "FLAT"),
    (r"SHED", "SHED"),
    (r"DOME", "DOME"),
    (r"SALTBOX", "SALTBOX"),
)
_ROOF_COVER_TOKENS: tuple[tuple[str, str], ...] = (
    (r"COMP\s*SHGLS?", "COMPOSITION SHINGLES"),
    (r"COMP(?:OSITION)?\s*SHINGLES?", "COMPOSITION SHINGLES"),
    (r"ASPHALT\s*SHINGLES?", "ASPHALT SHINGLES"),
    (r"ASPHALT", "ASPHALT"),
    (r"WOOD\s*SHGLS?", "WOOD SHINGLES"),
    (r"WOOD\s*SHINGLES?", "WOOD SHINGLES"),
    (r"METAL", "METAL"),
    (r"SLATE", "SLATE"),
    (r"CLAY\s*TILE", "CLAY TILE"),
    (r"TILE", "TILE"),
    (r"RUBBER", "RUBBER"),
    (r"BUILT\s*UP", "BUILT UP"),
)
_ROOF_LABEL_RE = re.compile(
    r"\bROOF\s*(?:TYPE\s*/\s*MATERIAL|TYPE|MATERIAL|STYLE|COVER)\b",
    re.IGNORECASE,
)


def _looks_like_roof_cover(value: str) -> bool:
    """True when the string value matches a roof-cover material keyword."""
    upper = value.upper()
    return any(re.search(rf"\b{p}\b", upper) for p, _ in _ROOF_COVER_TOKENS)


def _looks_like_roof_style(value: str) -> bool:
    """True when the string value matches a roof-style keyword."""
    upper = value.upper()
    return any(re.search(rf"\b{p}\b", upper) for p, _ in _ROOF_STYLE_TOKENS)


def _post_extract_roof(data: dict, text: str) -> dict:
    """Recover roofstyle / roofcover from a 'Roof Type/Material' label.

    Some assessor cards (e.g. Richmond County VA) print BUILDING
    PROPERTIES as a label row + a value row, with several columns. The
    chunked-CSV extractor frequently mis-aligns when the row contains
    multi-word values ("CRAWL CONCRETE", "GABLE COMP SHGLS"): the LLM
    fragments multi-word values into separate cells, sliding all
    subsequent fields one slot. The result is roofstyle/roofcover
    landing in attic/atticsqft/intwall/extwall, or the two getting
    swapped (a cover material like "COMP SHGLS" landing in roofstyle).

    Recover roofstyle and roofcover deterministically by scanning the
    OCR text for known roof-style and roof-cover keywords inside a
    window after a "Roof" label. If the LLM filled roofstyle with a
    cover material (or roofcover with a style word), correct it.
    """
    m = _ROOF_LABEL_RE.search(text)
    if not m:
        return data
    region = text[m.end() : m.end() + 300].upper()

    existing_style = data.get("roofstyle")
    style_invalid = (
        isinstance(existing_style, str)
        and _looks_like_roof_cover(existing_style)
        and not _looks_like_roof_style(existing_style)
    )
    if not existing_style or style_invalid:
        for pattern, canonical in _ROOF_STYLE_TOKENS:
            if re.search(rf"\b{pattern}\b", region):
                if style_invalid:
                    logger.warning(
                        "roofstyle overridden: LLM said %r (looks like a cover), "
                        "ROOF label region says %s",
                        existing_style, canonical,
                    )
                else:
                    logger.info(
                        "roofstyle recovered from ROOF label: %s", canonical,
                    )
                data["roofstyle"] = canonical
                break

    existing_cover = data.get("roofcover")
    cover_invalid = (
        isinstance(existing_cover, str)
        and _looks_like_roof_style(existing_cover)
        and not _looks_like_roof_cover(existing_cover)
    )
    if not existing_cover or cover_invalid:
        for pattern, canonical in _ROOF_COVER_TOKENS:
            if re.search(rf"\b{pattern}\b", region):
                if cover_invalid:
                    logger.warning(
                        "roofcover overridden: LLM said %r (looks like a style), "
                        "ROOF label region says %s",
                        existing_cover, canonical,
                    )
                else:
                    logger.info(
                        "roofcover recovered from ROOF label: %s", canonical,
                    )
                data["roofcover"] = canonical
                break

    return data


# Anchor for a "Zoning" label in the OCR text. The value can either
# follow inline ("Zoning: A-1") or sit several columns down on the
# value row beneath a multi-column label header ("...UTILITIES ZONING
# CLASS\n...Electric A-1 2"). The post-pass scans a window after the
# label for the first plausible zoning-code token.
_ZONING_LABEL_RE = re.compile(r"\bzoning\b", re.IGNORECASE)
# A real zoning classification: letter prefix + digit (with optional
# dashes / extra letters) OR a known all-letter code. The first-letter
# word boundary keeps us from matching a bare number or a 4-digit zip.
_ZONING_CODE_RE = re.compile(
    r"\b(?:[A-Z]{1,4}-\d[A-Z\d-]*|[A-Z]{1,4}\d[A-Z\d-]*|PUD|PUL[_A-Z\d]*)\b",
    re.IGNORECASE,
)
# Tokens that look like zoning codes but are not — appear adjacent to
# the zoning label on Richmond-style cards as column headers / class
# numbers. Skip these and keep scanning.
_ZONING_BLOCKLIST = {
    "CLASS", "CODE", "DESC", "DESCRIPTION",
}


def _post_extract_zoning(data: dict, text: str) -> dict:
    """Recover zoningcode from an explicit ZONING / Zoning Code label.

    The chunked-CSV extractor occasionally drops zoningcode when the
    LLM is unsure which short token next to the label is the zoning
    code (cards often have a class or use code adjacent to it). When
    that happens, fall back to a label-anchored regex match.

    Search window is wide (~150 chars) because grid-style cards print
    label rows and value rows on separate lines, with the zoning value
    several columns down the value row.
    """
    if data.get("zoningcode"):
        return data
    m = _ZONING_LABEL_RE.search(text)
    if not m:
        return data
    region = text[m.end() : m.end() + 150]
    for cm in _ZONING_CODE_RE.finditer(region):
        candidate = cm.group(0).strip().upper()
        if candidate in _ZONING_BLOCKLIST:
            continue
        data["zoningcode"] = candidate
        logger.info("zoningcode recovered from ZONING label: %s", candidate)
        return data
    return data


def _recover_heatfuel_from_hvac_chunk(data: dict) -> dict:
    """Move a fuel-name value out of heating/cooling and into heatfuel.

    The HVAC chunk (heating, heatfuel, cooling) is the easiest one for the
    LLM to misalign: it's only three cells, and a missed comma puts a fuel
    name in the cooling slot or a delivery-system name in the heatfuel
    slot. Concrete patterns observed on Richmond County VA:
      HVAC:,,ELECTRIC,   (heating empty, heatfuel empty, cooling=ELECTRIC)
      HVAC:ELECTRIC,ELECTRIC,   (LLM duplicated the fuel into heating)

    When heatfuel is missing and another HVAC slot holds a canonical fuel
    name, that's almost certainly a misalignment. Move the value over.
    """
    if data.get("heatfuel"):
        return data
    for src in ("cooling", "heating"):
        v = data.get(src)
        if not isinstance(v, str):
            continue
        canonical = v.strip().upper()
        if canonical in _CANONICAL_HEATFUEL_VALUES:
            data["heatfuel"] = canonical
            del data[src]
            logger.info(
                "heatfuel recovered from misplaced %s value: %s",
                src, canonical,
            )
            return data
    return data


def _post_extract_heating_cooling_codes(data: dict, text: str) -> dict:
    """Set heating=CENTRAL / cooling=CENTRAL AIR from 'C-HEAT' / 'C-AIR' codes.

    Richmond County VA cards encode HVAC in the BUILDING SECTIONS row as
    bare codes (e.g. "C-HEAT" for central heat, "C-AIR" for central air).
    Those codes are unambiguous when present. Override the LLM only when
    its value is missing or is clearly a fuel name accidentally placed
    in the heating/cooling slot (the chunked-CSV extractor occasionally
    puts the fuel where heating or cooling belongs because the columns
    share a row).
    """
    has_c_heat = bool(re.search(r"\bC[-\s]+HEAT\b", text, re.IGNORECASE))
    has_c_air = bool(re.search(r"\bC[-\s]+AIR\b", text, re.IGNORECASE))

    if has_c_heat:
        existing = data.get("heating")
        if not existing or (
            isinstance(existing, str)
            and existing.strip().upper() in _CANONICAL_HEATFUEL_VALUES
        ):
            if existing:
                logger.warning(
                    "heating overridden from C-HEAT signal: %r -> CENTRAL",
                    existing,
                )
            else:
                logger.info("heating recovered from C-HEAT signal: CENTRAL")
            data["heating"] = "CENTRAL"

    if has_c_air:
        existing = data.get("cooling")
        if not existing or (
            isinstance(existing, str)
            and existing.strip().upper() in _CANONICAL_HEATFUEL_VALUES
        ):
            if existing:
                logger.warning(
                    "cooling overridden from C-AIR signal: %r -> CENTRAL AIR",
                    existing,
                )
            else:
                logger.info(
                    "cooling recovered from C-AIR signal: CENTRAL AIR",
                )
            data["cooling"] = "CENTRAL AIR"

    return data


# Values that signal a STRUCT-chunk slot was filled with a value from a
# neighbouring slot — typically because the LLM fragmented a multi-word
# cell ("CRAWL CONCRETE", "GABLE COMP SHGLS") and shifted later cells
# one position to the right.
_NON_ATTIC_VALUES = {
    "GABLE", "HIP", "FLAT", "GAMBREL", "MANSARD", "SHED", "DOME",
    "SALTBOX",
    "CONCRETE", "BRICK", "STONE", "BLOCK", "SLAB", "PIER",
    "COMP", "COMP SHGLS", "COMP SHINGLES", "COMPOSITION SHINGLES",
    "METAL", "ASPHALT", "ASPHALT SHINGLES", "SHINGLES", "TILE",
    "SLATE", "RUBBER",
    "DRY WALL", "DRYWALL", "PLASTER", "PANELING",
}
_NON_EXTWALL_VALUES = {
    "COMP", "COMP SHGLS", "COMP SHINGLES", "COMPOSITION SHINGLES",
    "ASPHALT SHINGLES", "ASPHALT", "METAL SHINGLES", "TILE", "SLATE",
    "RUBBER", "GABLE", "HIP", "FLAT", "GAMBREL", "MANSARD",
}


def _strip_misplaced_struct_values(data: dict) -> dict:
    """Drop STRUCT-chunk values that obviously belong to a different slot.

    The chunked-CSV extractor occasionally fragments multi-word cells in
    BUILDING PROPERTIES rows ("CRAWL CONCRETE", "GABLE COMP SHGLS"),
    sliding values into the wrong slots. The downstream
    `_post_extract_roof` recovers roofstyle/roofcover from text, so the
    contaminated values in attic/extwall are now redundant — drop them
    rather than letting them propagate.
    """
    attic = data.get("attic")
    if isinstance(attic, str) and attic.strip().upper() in _NON_ATTIC_VALUES:
        logger.info(
            "attic value %r looks like a non-attic material; dropping", attic,
        )
        del data["attic"]

    extwall = data.get("extwall")
    if isinstance(extwall, str) and extwall.strip().upper() in _NON_EXTWALL_VALUES:
        logger.info(
            "extwall value %r looks like a roof cover/style; dropping", extwall,
        )
        del data["extwall"]

    return data




def _infer_heating_from_central_air(data: dict, text: str) -> dict:
    """Infer heating from cooling on cards where the only HVAC indicator
    is a "Central Air" signal alongside a heat-fuel label.

    Common on Henry County VA InteractiveGIS cards: the card lists
    "Heat Fuel: ELECT" + "Central Air % 100" with no explicit
    heating-delivery cell. The building has one central forced-air HVAC
    system that does both heating and cooling, so heating mirrors
    cooling in this layout.

    Conditions to fire:
      - heating is empty/missing
      - cooling resolves to a CENTRAL value (CENTRAL / CENTRAL AIR)
      - the OCR/native text actually contains a "Central Air" or
        "Central A/C" token (don't trust a cooling value the LLM might
        have invented out of nowhere)
      - the AC% is NOT explicitly zero. Cards like the Henry mobile-home
        layout show "Central Air % 0", which means the AC option on the
        card line *exists* but the value is none — the building has no
        central air, so we should not infer central-air heating.
    """
    if data.get("heating"):
        return data

    cooling = data.get("cooling")
    if not isinstance(cooling, str):
        return data
    cooling_upper = cooling.strip().upper()
    if "CENTRAL" not in cooling_upper:
        return data

    if not re.search(r"\bcentral\s*(?:air|a\s*/?\s*c)\b", text, re.IGNORECASE):
        return data

    # Suppress when the card explicitly shows AC %/value is zero — e.g.
    # "Central Air % 0", "Central Air % : 0", "AC % 0". The presence of
    # the label doesn't mean AC is installed.
    if re.search(
        r"\bcentral\s*(?:air|a\s*/?\s*c)\s*(?:%|percent)?\s*[:\-]?\s*0\b",
        text,
        re.IGNORECASE,
    ):
        logger.info(
            "Skipping heating-from-cooling inference: Central Air %% is 0",
        )
        return data

    data["heating"] = cooling_upper
    logger.info(
        "Inferred heating=%r from cooling + 'Central Air' signal in source text",
        cooling_upper,
    )
    return data


def _drop_unverified_none(data: dict, text: str) -> dict:
    """Strip "NONE" string values that the OCR text doesn't actually contain.

    The LLM is told to emit "NONE" only when the card explicitly shows that
    literal value, but it occasionally substitutes "NONE" for missing data
    anyway (especially for heatfuel/cooling/basement). If the word "NONE"
    or "None" doesn't appear as a token in the OCR text, the value is a
    fabrication and we drop it.
    """
    if not text:
        return data
    text_has_none = re.search(r"\bnone\b", text, re.IGNORECASE) is not None
    if text_has_none:
        return data
    dropped: list[str] = []
    for k in list(data.keys()):
        v = data[k]
        if isinstance(v, str) and v.strip().upper() == "NONE":
            del data[k]
            dropped.append(k)
    if dropped:
        logger.info(
            "Dropped %d unverified 'NONE' value(s): %s",
            len(dropped), ", ".join(dropped),
        )
    return data


def extract_data(state: AgentState) -> dict:
    """Extract structured property data from the OCR'd PDF text."""
    text = state.get("pdf_text", "") or ""
    if len(text.strip()) < 50:
        logger.info(
            "PDF yielded %d chars of text — skipping extraction",
            len(text.strip()),
        )
        return {"property_data": {}}

    data = _run_extraction_llm(text, state.get("context"))
    data = _reconcile_value_totals(data)
    data = _retry_parcelid(data, text)
    data = _recover_heatfuel_from_hvac_chunk(data)
    data = _post_extract_heatfuel(data, text)
    data = _post_extract_heating_cooling_codes(data, text)
    data = _infer_heating_from_central_air(data, text)
    data = _post_extract_zoning(data, text)
    data = _post_extract_roof(data, text)
    data = _strip_misplaced_struct_values(data)
    data = _drop_unverified_none(data, text)
    logger.info("Extracted %d fields from PDF text", len(data))
    return {"property_data": data}


# Kick off model load in a background daemon thread at import time.
# Why this exists: when many threads / Ray actors call read_property_card
# concurrently very soon after import, they all hit _get_ocr_model() at
# roughly the same moment, all see _OCR_MODEL is None, and pile up on
# the first-call lock. Worse, if the cache miss triggers a download,
# multiple processes (which the in-process lock can't coordinate) race
# on the same .pt path. By starting the load before any caller arrives,
# the download finishes once and all subsequent callers hit the
# already-populated singleton on the lock-free fast path.
#
# Daemon thread (not import-time blocking call) so importing the module
# stays fast for code paths that don't actually need OCR (tests, dry
# runs, etc.). Errors here are swallowed and logged — the lazy lock
# path still works as a fallback if the eager warm fails.
def _eager_warm_ocr_model() -> None:
    try:
        _get_ocr_model()
    except Exception as e:
        logger.warning(
            "Background OCR warm failed (%s); first-call lazy load will retry", e,
        )


# Module-level guard so the warm thread is started exactly once even if
# the module gets re-imported (e.g. via importlib.reload). The truthy
# check is read without a lock; the worst case is two threads briefly,
# both of which serialise on _OCR_MODEL_LOCK inside _get_ocr_model.
if os.environ.get("CARD_READER_NO_EAGER_WARM") != "1":
    threading.Thread(
        target=_eager_warm_ocr_model,
        daemon=True,
        name="docTR-ocr-warm",
    ).start()
