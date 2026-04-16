from langgraph.graph import END, START, StateGraph

from .nodes import (
    download_pdf,
    extract_data,
    extract_pdf_text,
    extract_property_photos,
)
from .state import AgentState


def build_graph():
    """Build and compile the landrecords-card-reader LangGraph.

    Flow:
      download_pdf
        -> extract_property_photos (pull embedded photos -> state.property_photos)
        -> extract_pdf_text        (pull embedded text as markdown)
        -> extract_data            (LLM: markdown -> JSON)
    """
    graph = StateGraph(AgentState)

    graph.add_node("download_pdf", download_pdf)
    #graph.add_node("extract_property_photos", extract_property_photos)
    graph.add_node("extract_pdf_text", extract_pdf_text)
    graph.add_node("extract_data", extract_data)

    graph.add_edge(START, "download_pdf")
    #graph.add_edge("download_pdf", "extract_property_photos")
    graph.add_edge("download_pdf", "extract_pdf_text")
    graph.add_edge("extract_pdf_text", "extract_data")
    graph.add_edge("extract_data", END)

    return graph.compile()


app = build_graph()
