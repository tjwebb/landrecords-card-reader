from typing import Any, TypedDict


class AgentState(TypedDict):
    pdf_url: str
    pdf_bytes: bytes | None
    pdf_content: bytes
    pdf_markdown: str
    property_photos: list[dict[str, Any]]
    property_data: dict[str, Any]
    result: str
