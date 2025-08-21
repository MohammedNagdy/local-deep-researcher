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
        max_web_research_loops: Number of research iterations
        llm_provider: Provider for the LLM (ollama, lmstudio, groq)
        local_llm: Model name to use
        search_api: Web search API to use
        fetch_full_page: Include full page content in results
        strip_thinking_tokens: Remove <think> tokens from model outputs
        use_tool_calling: Use tool calling instead of JSON mode
        ollama_base_url: Ollama base URL
        lmstudio_base_url: LMStudio base URL
        groq_base_url: Groq Open Source Models API base URL
        groq_api_key: API key for Groq Open Source Models

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