import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://graybase:11434")
EXTRACTION_MODEL = os.getenv("EXTRACTION_MODEL", "gemma4:26b-a4b-it-q8_0")
EXTRACTION_CONTEXT_LENGTH = int(os.getenv("EXTRACTION_CONTEXT_LENGTH", "0")) or None
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
PHOTO_CLASSIFICATION_MODEL = os.getenv("PHOTO_CLASSIFICATION_MODEL", "gemma4:e2b")
CLASSIFICATION_CONTEXT_LENGTH = int(os.getenv("CLASSIFICATION_CONTEXT_LENGTH", "0")) or None
