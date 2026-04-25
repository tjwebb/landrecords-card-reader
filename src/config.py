import os
from dotenv import load_dotenv

load_dotenv()

CARD_READER_OLLAMA_HOST = os.getenv("CARD_READER_OLLAMA_HOST", "http://graybase:11434")
CARD_READER_EXTRACTION_MODEL = os.getenv("CARD_READER_EXTRACTION_MODEL", "gemma4:26b-a4b-it-q8_0")
CARD_READER_PHOTO_CLASSIFICATION_MODEL = os.getenv("CARD_READER_PHOTO_CLASSIFICATION_MODEL", "gemma4:e2b")
