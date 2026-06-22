"""Tests for web helpers — URL detection, page fetching, link finding, crawling."""

from veska.tools.web_helpers import is_url, find_links, format_page_result, format_crawl_results
from bs4 import BeautifulSoup


def test_is_url_with_http():
    assert is_url("http://example.com") is True


def test_is_url_with_https():
    assert is_url("https://veska.in") is True


def test_is_url_with_search_query():
    assert is_url("latest AI trends") is False


def test_is_url_with_spaces():
    assert is_url("  https://veska.in  ") is True


def test_is_url_with_empty_string():
    assert is_url("") is False


def test_find_links_same_domain():
    html = """
    <html><body>
        <a href="/docs">Docs</a>
        <a href="/about">About</a>
        <a href="https://external.com">External</a>
    </body></html>
    """
    soup = BeautifulSoup(html, "html.parser")
    links = find_links(soup, "https://veska.in", "veska.in")

    assert "https://veska.in/docs" in links
    assert "https://veska.in/about" in links
    # External link should not be included
    assert "https://external.com" not in links


def test_find_links_skips_anchors():
    html = """
    <html><body>
        <a href="#section">Anchor</a>
        <a href="javascript:void(0)">JS</a>
        <a href="mailto:test@test.com">Email</a>
        <a href="/real-page">Real</a>
    </body></html>
    """
    soup = BeautifulSoup(html, "html.parser")
    links = find_links(soup, "https://veska.in", "veska.in")

    assert len(links) == 1
    assert "https://veska.in/real-page" in links


def test_find_links_removes_duplicates():
    html = """
    <html><body>
        <a href="/docs">Link 1</a>
        <a href="/docs">Link 2</a>
        <a href="/docs/">Link 3</a>
    </body></html>
    """
    soup = BeautifulSoup(html, "html.parser")
    links = find_links(soup, "https://veska.in", "veska.in")

    assert links.count("https://veska.in/docs") == 1


def test_format_page_result_success():
    page = {
        "url": "https://veska.in",
        "title": "Veska",
        "text": "A multi-agent framework",
        "links": [],
        "success": True,
        "error": "",
    }
    result = format_page_result(page)

    assert "Veska" in result
    assert "https://veska.in" in result
    assert "multi-agent" in result


def test_format_page_result_error():
    page = {
        "url": "https://bad.com",
        "title": "",
        "text": "",
        "links": [],
        "success": False,
        "error": "Timeout",
    }
    result = format_page_result(page)

    assert "Error" in result
    assert "Timeout" in result


def test_format_crawl_results():
    pages = [
        {"url": "https://veska.in", "title": "Home", "text": "Welcome", "links": [], "success": True, "error": ""},
        {"url": "https://veska.in/docs", "title": "Docs", "text": "Documentation", "links": [], "success": True, "error": ""},
    ]
    result = format_crawl_results(pages)

    assert "Crawled 2 pages" in result
    assert "Home" in result
    assert "Docs" in result
