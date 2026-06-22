"""
Shared web helpers for Veska web tools.

Provides: fetch_page, find_links, crawl_site, is_url
Used by web_search.py, google_search.py, and bing_search.py.
"""

from __future__ import annotations

import re
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup


def is_url(text: str) -> bool:
    """Check if text is a URL."""
    text = text.strip()
    return text.startswith("http://") or text.startswith("https://")


def fetch_page(url: str, timeout: int = 15) -> dict:
    """
    Fetch a URL and extract readable text content.

    Returns:
        {"url": str, "title": str, "text": str, "links": list[str], "success": bool, "error": str}
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; Veska/0.1; +https://veska.in)"
        }
        response = httpx.get(url, headers=headers, timeout=timeout, follow_redirects=True)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove non-content elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "iframe"]):
            tag.decompose()

        # Extract title
        title = soup.title.string.strip() if soup.title and soup.title.string else ""

        # Extract readable text
        text = soup.get_text(separator="\n", strip=True)
        # Clean up excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Find links on the same domain
        domain = urlparse(url).netloc
        links = find_links(soup, url, domain)

        return {
            "url": url,
            "title": title,
            "text": text[:10000],  # Cap at 10k chars to avoid huge responses
            "links": links,
            "success": True,
            "error": "",
        }

    except httpx.TimeoutException:
        return {"url": url, "title": "", "text": "", "links": [], "success": False, "error": f"Timeout fetching {url}"}
    except httpx.HTTPStatusError as e:
        return {"url": url, "title": "", "text": "", "links": [], "success": False, "error": f"HTTP {e.response.status_code} for {url}"}
    except Exception as e:
        return {"url": url, "title": "", "text": "", "links": [], "success": False, "error": f"Error fetching {url}: {str(e)}"}


def find_links(soup: BeautifulSoup, base_url: str, domain: str) -> list[str]:
    """
    Find all internal links on the same domain.

    Args:
        soup: Parsed HTML.
        base_url: The page URL (for resolving relative links).
        domain: Only return links on this domain.

    Returns:
        List of unique absolute URLs on the same domain.
    """
    links = set()

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"].strip()

        # Skip anchors, javascript, mailto
        if href.startswith("#") or href.startswith("javascript:") or href.startswith("mailto:"):
            continue

        # Resolve relative URLs
        full_url = urljoin(base_url, href)

        # Only keep links on the same domain
        parsed = urlparse(full_url)
        if parsed.netloc == domain:
            # Remove fragments
            clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            # Remove trailing slash for consistency
            clean_url = clean_url.rstrip("/")
            links.add(clean_url)

    return sorted(links)


def crawl_site(url: str, max_pages: int = 10, timeout: int = 15) -> list[dict]:
    """
    Crawl a website starting from a URL, following internal links.

    Args:
        url: Starting URL.
        max_pages: Maximum number of pages to fetch (default 10).
        timeout: Timeout per page in seconds.

    Returns:
        List of page results from fetch_page.
    """
    visited = set()
    to_visit = [url.rstrip("/")]
    results = []

    while to_visit and len(results) < max_pages:
        current_url = to_visit.pop(0)

        # Normalize URL
        current_url = current_url.rstrip("/")

        if current_url in visited:
            continue

        visited.add(current_url)
        page = fetch_page(current_url, timeout=timeout)
        results.append(page)

        if page["success"]:
            # Add new internal links to the queue
            for link in page["links"]:
                link = link.rstrip("/")
                if link not in visited and link not in to_visit:
                    to_visit.append(link)

    return results


def format_page_result(page: dict) -> str:
    """Format a single page result as readable text."""
    if not page["success"]:
        return f"[Error] {page['url']}: {page['error']}"

    parts = []
    if page["title"]:
        parts.append(f"Title: {page['title']}")
    parts.append(f"URL: {page['url']}")
    parts.append(f"\n{page['text']}")
    return "\n".join(parts)


def format_crawl_results(pages: list[dict]) -> str:
    """Format multiple page results as readable text."""
    parts = [f"Crawled {len(pages)} pages:\n"]
    for i, page in enumerate(pages, 1):
        parts.append(f"--- Page {i} ---")
        parts.append(format_page_result(page))
        parts.append("")
    return "\n".join(parts)
