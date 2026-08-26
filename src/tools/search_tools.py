"""Search tools integration: DuckDuckGo, ArXiv, Wikipedia, and Tavily."""

import logging
import os
import warnings
from typing import List, Dict, Any, Optional
from src.models.schemas import SourceDocument
from src.tools.web_scraper import scrape_url_content

# Filter warnings from duckduckgo rename notice
warnings.filterwarnings("ignore", category=RuntimeWarning, module="duckduckgo_search")

logger = logging.getLogger(__name__)


class DuckDuckGoSearchTool:
    """Performs web search using DuckDuckGo (no API key required)."""

    def search(self, query: str, max_results: int = 4) -> List[Dict[str, Any]]:
        results = []
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                ddg_results = list(ddgs.text(query, max_results=max_results))
                for item in ddg_results:
                    results.append({
                        "title": item.get("title", "Web Result"),
                        "url": item.get("href", ""),
                        "snippet": item.get("body", ""),
                        "source_type": "web",
                        "authors": []
                    })
        except Exception as e:
            logger.debug(f"DuckDuckGo search exception for query '{query}': {e}")
        return results


class ArXivSearchTool:
    """Searches scientific papers and preprints on ArXiv."""

    def search(self, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        results = []
        try:
            import arxiv
            client = arxiv.Client()
            # Clean query for ArXiv API
            clean_query = query.replace(":", " ").replace("-", " ")[:120]
            search = arxiv.Search(
                query=clean_query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            for paper in client.results(search):
                authors = [author.name for author in paper.authors][:4]
                pub_date = paper.published.strftime("%Y-%m-%d") if paper.published else None
                results.append({
                    "title": f"[ArXiv] {paper.title}",
                    "url": paper.entry_id,
                    "snippet": paper.summary[:600].replace("\n", " ") + "...",
                    "source_type": "arxiv",
                    "authors": authors,
                    "published_date": pub_date
                })
        except Exception as e:
            logger.debug(f"ArXiv search exception for query '{query}': {e}")
        return results


class WikipediaSearchTool:
    """Searches Wikipedia for authoritative background and definitions."""

    def search(self, query: str, max_results: int = 2) -> List[Dict[str, Any]]:
        results = []
        try:
            import wikipedia
            wikipedia.set_lang("en")
            # Extract key concept words for Wikipedia
            keywords = " ".join([w for w in query.split() if len(w) > 2][:4])
            titles = wikipedia.search(keywords, results=max_results)
            for title in titles:
                try:
                    page = wikipedia.page(title, auto_suggest=False)
                    summary = page.summary[:700].replace("\n", " ") + "..."
                    results.append({
                        "title": f"[Wikipedia] {page.title}",
                        "url": page.url,
                        "snippet": summary,
                        "source_type": "wikipedia",
                        "authors": ["Wikipedia Contributors"]
                    })
                except Exception:
                    continue
        except Exception as e:
            logger.debug(f"Wikipedia search exception for query '{query}': {e}")
        return results


class TavilySearchTool:
    """Searches using Tavily AI Search API if an API key is provided."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")

    def search(self, query: str, max_results: int = 4) -> List[Dict[str, Any]]:
        if not self.api_key:
            return []
        results = []
        try:
            import requests
            response = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": self.api_key,
                    "query": query,
                    "search_depth": "basic",
                    "include_answer": False,
                    "max_results": max_results
                },
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                for item in data.get("results", []):
                    results.append({
                        "title": item.get("title", "Web Article"),
                        "url": item.get("url", ""),
                        "snippet": item.get("content", ""),
                        "source_type": "tavily",
                        "authors": []
                    })
        except Exception as e:
            logger.debug(f"Tavily search exception for '{query}': {e}")
        return results


class UnifiedRetriever:
    """Orchestrates retrieval across enabled search engines and enriches sources."""

    def __init__(
        self,
        enable_web: bool = True,
        enable_arxiv: bool = True,
        enable_wikipedia: bool = True,
        tavily_api_key: Optional[str] = None,
        scrape_full_content: bool = True,
        max_scrape_count: int = 4
    ):
        self.ddg = DuckDuckGoSearchTool() if enable_web else None
        self.arxiv = ArXivSearchTool() if enable_arxiv else None
        self.wiki = WikipediaSearchTool() if enable_wikipedia else None
        self.tavily = TavilySearchTool(api_key=tavily_api_key) if (enable_web and (tavily_api_key or os.getenv("TAVILY_API_KEY"))) else None
        self.scrape_full_content = scrape_full_content
        self.max_scrape_count = max_scrape_count

    def retrieve(self, queries: List[str], max_total_sources: int = 12) -> List[SourceDocument]:
        """Executes queries across all available tools, deduplicates, and formats sources."""
        all_raw_results = []
        seen_urls = set()

        for q in queries:
            # Tavily AI Search (if key is set)
            if self.tavily:
                tavily_res = self.tavily.search(q, max_results=3)
                all_raw_results.extend(tavily_res)

            # DuckDuckGo search
            if self.ddg:
                ddg_res = self.ddg.search(q, max_results=3)
                all_raw_results.extend(ddg_res)

            # Wikipedia search
            if self.wiki and len(all_raw_results) < max_total_sources:
                wiki_res = self.wiki.search(q, max_results=2)
                all_raw_results.extend(wiki_res)

            # ArXiv search
            if self.arxiv and len(all_raw_results) < max_total_sources:
                arxiv_res = self.arxiv.search(q, max_results=2)
                all_raw_results.extend(arxiv_res)

        # Deduplicate results by normalized URL and title
        unique_results = []
        for item in all_raw_results:
            url = item.get("url", "").strip().rstrip("/")
            title = item.get("title", "").strip()
            key = url if url else title

            if key and key not in seen_urls:
                seen_urls.add(key)
                unique_results.append(item)
                if len(unique_results) >= max_total_sources:
                    break

        # Convert to SourceDocument models with 1-based indexing
        source_docs: List[SourceDocument] = []
        scraped_count = 0

        for idx, item in enumerate(unique_results, start=1):
            url = item.get("url", "")
            content = None

            # Attempt full content scraping for the first few web sources
            if self.scrape_full_content and scraped_count < self.max_scrape_count:
                if item.get("source_type") in ["web", "tavily"] and url:
                    content = scrape_url_content(url, max_chars=3000)
                    if content:
                        scraped_count += 1

            doc = SourceDocument(
                id=idx,
                title=item.get("title", f"Source {idx}"),
                url=url,
                snippet=item.get("snippet", ""),
                content=content,
                source_type=item.get("source_type", "web"),
                authors=item.get("authors", []),
                published_date=item.get("published_date"),
                relevance_score=0.85
            )
            source_docs.append(doc)

        return source_docs
