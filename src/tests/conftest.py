"""Pytest config for the fixture-driven integration tests.

Turns on live streaming of the extraction LLM's output so the model's
response is visible token-by-token as it arrives during a test run.
This is decoupled from the log level — only the streaming bytes are
shown, not all DEBUG-level logging.

Requires ``addopts = "-s"`` (set in pyproject.toml's pytest config)
so pytest does not capture stderr at the file-descriptor level —
without it, the streaming bytes are buffered until the test finishes,
defeating the streaming.
"""

import os

os.environ.setdefault("CARD_READER_STREAM_TO_CONSOLE", "1")
