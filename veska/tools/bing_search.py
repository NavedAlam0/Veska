"""
Bing Search tool for Veska — powered by Bing Search API.

Requires BING_API_KEY in .env file.
User adds "bing_search" to their tools list.
"""

from __future__ import annotations

import os

import httpx

from veska.tools.base import Tool, ToolParameter
from veska.tools.web_helpers import (
    is_url,
    fetch_page,
    crawl_site,
    format_page_result,
    format_crawl_results,
)


def _bing_search(query: str, crawl: bool = False, max_pages: int = 10) -> str:
    """
    Search with Bing, fetch a URL, or crawl a website.

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

    # It's a search query — use Bing Search API
    api_key = os.environ.get("BING_API_KEY")

    if not api_key:
        return "Error: BING_API_KEY not found in environment. Add it to your .env file."

    try:
        response = httpx.get(
            "https://api.bing.microsoft.com/v7.0/search",
            headers={"Ocp-Apim-Subscription-Key": api_key},
            params={
                "q": query,
                "count": 5,
            },
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()

        items = data.get("webPages", {}).get("value", [])
        if not items:
            return f"No results found for: {query}"

        results = []
        for item in items:
            results.append(
                f"Title: {item.get('name', '')}\n"
                f"URL: {item.get('url', '')}\n"
                f"Snippet: {item.get('snippet', '')}\n"
            )

        return f"Search results for: {query}\n\n" + "\n---\n".join(results)

    except httpx.HTTPStatusError as e:
        return f"Bing API error: HTTP {e.response.status_code}"
    except Exception as e:
        return f"Bing search error: {str(e)}"


def get_bing_search_tools() -> list[Tool]:
    """Get the Bing search tool (requires API key in .env)."""
    return [
        Tool(
            name="bing_search",
            description="Search the web with Bing, fetch a URL, or crawl an entire website",
            when_to_use=(
                "When you need to search the internet using Bing, "
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
            function=_bing_search,
        ),
    ]
