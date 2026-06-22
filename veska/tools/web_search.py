"""
Web Search tool for Veska — powered by DuckDuckGo.

Free, no API key needed. Handles search, fetch, and crawl in one tool.
User just adds "web_tools" to their tools list.
"""

from __future__ import annotations

from veska.tools.base import Tool, ToolParameter
from veska.tools.web_helpers import (
    is_url,
    fetch_page,
    crawl_site,
    format_page_result,
    format_crawl_results,
)


def _web_search(query: str, crawl: bool = False, max_pages: int = 10) -> str:
    """
    Search the web, fetch a URL, or crawl a website.

    Args:
        query: A search query or a URL.
        crawl: If True and query is a URL, crawl the entire site (follow internal links).
        max_pages: Maximum pages to crawl (default 10, only used when crawl=True).
    """
    # If it's a URL — fetch or crawl
    if is_url(query):
        if crawl:
            pages = crawl_site(query, max_pages=max_pages)
            return format_crawl_results(pages)
        else:
            page = fetch_page(query)
            return format_page_result(page)

    # It's a search query — use DuckDuckGo
    try:
        from duckduckgo_search import DDGS

        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=5):
                results.append(
                    f"Title: {r['title']}\n"
                    f"URL: {r['href']}\n"
                    f"Snippet: {r['body']}\n"
                )

        if not results:
            return f"No results found for: {query}"

        return f"Search results for: {query}\n\n" + "\n---\n".join(results)

    except ImportError:
        return "Error: duckduckgo-search package not installed. Run: pip install duckduckgo-search"
    except Exception as e:
        return f"Search error: {str(e)}"


def get_web_tools() -> list[Tool]:
    """Get the web search tool (DuckDuckGo — free, no API key)."""
    return [
        Tool(
            name="web_search",
            description="Search the web, fetch a URL, or crawl an entire website",
            when_to_use=(
                "When you need to search the internet for information, "
                "read the content of a webpage, or crawl a full website. "
                "Pass a search query to search, a URL to fetch that page, "
                "or a URL with crawl=true to crawl the entire site."
            ),
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="A search query (e.g. 'latest AI trends') or a URL (e.g. 'https://veska.in')",
                ),
                ToolParameter(
                    name="crawl",
                    type="boolean",
                    description="If true and query is a URL, crawl the entire site following internal links",
                    required=False,
                    default=False,
                ),
                ToolParameter(
                    name="max_pages",
                    type="integer",
                    description="Maximum pages to crawl (only used when crawl is true)",
                    required=False,
                    default=10,
                ),
            ],
            function=_web_search,
        ),
    ]
