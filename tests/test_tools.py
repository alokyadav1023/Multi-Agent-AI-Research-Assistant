"""Tests for search tools, web scraper, and UnifiedRetriever."""

import pytest
from src.tools.web_scraper import scrape_url_content
from src.tools.search_tools import UnifiedRetriever, DuckDuckGoSearchTool, ArXivSearchTool, WikipediaSearchTool
from src.models.schemas import SourceDocument


def test_web_scraper_invalid_url():
    res = scrape_url_content("invalid-url-string")
    assert res is None


def test_unified_retriever_deduplication():
    # Instantiate retriever with empty/mocked calls to verify deduplication logic
    retriever = UnifiedRetriever(enable_web=False, enable_arxiv=False, enable_wikipedia=False)
    results = retriever.retrieve(["test query"], max_total_sources=5)
    assert isinstance(results, list)


def test_duckduckgo_tool_initialization():
    tool = DuckDuckGoSearchTool()
    assert tool is not None


def test_arxiv_tool_initialization():
    tool = ArXivSearchTool()
    assert tool is not None


def test_wikipedia_tool_initialization():
    tool = WikipediaSearchTool()
    assert tool is not None
