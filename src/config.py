import os
import shutil
import sys
from dotenv import load_dotenv

load_dotenv()

CARD_READER_OLLAMA_HOST = os.getenv("CARD_READER_OLLAMA_HOST", "http://graybase:11434")
CARD_READER_EXTRACTION_MODEL = os.getenv("CARD_READER_EXTRACTION_MODEL", "gemma4:26b-a4b-it-q8_0")
CARD_READER_PHOTO_CLASSIFICATION_MODEL = os.getenv("CARD_READER_PHOTO_CLASSIFICATION_MODEL", "gemma4:e2b")


# OS-level binaries the pipeline shells out to. The pip-installed wrapper
# (pdfkit) is useless without wkhtmltopdf on PATH, so we fail loudly at
# import time rather than producing cryptic subprocess errors deep in the
# pipeline.
_REQUIRED_BINARIES: dict[str, dict[str, str]] = {
    "wkhtmltopdf": {
        "purpose": "rendering HTML property pages to PDF",
        "macos": "brew install --cask wkhtmltopdf",
        "debian": "apt-get install -y wkhtmltopdf",
        "url": "https://wkhtmltopdf.org/downloads.html",
    },
}


def check_os_dependencies() -> None:
    """Raise if any required OS-level binary is missing from PATH.

    Called at package import time. Override by setting
    ``CARD_READER_SKIP_DEP_CHECK=1`` in the environment if you genuinely
    need to import the package without one of the binaries (e.g. running
    only against pre-fetched PDFs in a sandboxed environment).
    """
    if os.getenv("CARD_READER_SKIP_DEP_CHECK"):
        return

    missing: list[tuple[str, dict[str, str]]] = [
        (name, info) for name, info in _REQUIRED_BINARIES.items()
        if shutil.which(name) is None
    ]
    if not missing:
        return

    is_macos = sys.platform == "darwin"
    lines = ["Required OS dependencies are missing from PATH:"]
    for name, info in missing:
        install_cmd = info["macos"] if is_macos else info["debian"]
        lines.append(f"  - {name} ({info['purpose']})")
        lines.append(f"      install: {install_cmd}")
        lines.append(f"      docs:    {info['url']}")
    lines.append(
        "Set CARD_READER_SKIP_DEP_CHECK=1 to bypass this check (not recommended)."
    )
    raise RuntimeError("\n".join(lines))


check_os_dependencies()
