from typing import Optional, Literal, Dict, Any

from fastmcp import FastMCP

# Import the compiled graph and configuration
from ollama_deep_researcher.graph import graph


mcp = FastMCP("local-deep-researcher")


@mcp.tool
def run_research(
    topic: str
) -> Dict[str, Any]:
    """Run Local Deep Researcher on a topic and return a markdown summary.

    Parameters:
        topic: Research topic to investigate

    Returns:
        { "summary_markdown": str }
    """
    result = graph.invoke(
        {"research_topic": topic},
    )

    # Handle result that may be a dict or a dataclass-like object
    if isinstance(result, dict):
        summary = result.get("running_summary")
    else:  # dataclass with attribute
        summary = getattr(result, "running_summary", None)

    return {"summary_markdown": summary}


if __name__ == "__main__":
    mcp.run()