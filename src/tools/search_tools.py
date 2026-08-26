"""High-speed parallel search tools integration: DuckDuckGo, ArXiv, Wikipedia, and Tavily."""

import logging
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
from src.models.schemas import SourceDocument
from src.tools.web_scraper import scrape_url_content

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
            clean_query = query.replace(":", " ").replace("-", " ")[:100]
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
            keywords = " ".join([w for w in query.split() if len(w) > 2][:3])
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
                timeout=8
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
    """Orchestrates high-speed concurrent retrieval across enabled search engines."""

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

    def _query_worker(self, query: str) -> List[Dict[str, Any]]:
        results = []
        if self.tavily:
            results.extend(self.tavily.search(query, max_results=3))
        if self.ddg:
            results.extend(self.ddg.search(query, max_results=3))
        if self.wiki:
            results.extend(self.wiki.search(query, max_results=2))
        if self.arxiv:
            results.extend(self.arxiv.search(query, max_results=2))
        return results

    def retrieve(self, queries: List[str], max_total_sources: int = 12) -> List[SourceDocument]:
        """Executes search queries in parallel threads for maximum speed."""
        all_raw_results = []
        seen_urls = set()

        # Parallel query execution
        with ThreadPoolExecutor(max_workers=min(len(queries) or 1, 6)) as executor:
            futures = [executor.submit(self._query_worker, q) for q in queries]
            for future in as_completed(futures):
                try:
                    res = future.result()
                    all_raw_results.extend(res)
                except Exception as e:
                    logger.debug(f"Query worker error: {e}")

        # Deduplicate
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

        # Parallel content scraping for top sources
        sources_to_scrape = [
            (idx, item) for idx, item in enumerate(unique_results[:self.max_scrape_count], start=1)
            if item.get("source_type") in ["web", "tavily"] and item.get("url")
        ]

        scraped_contents: Dict[int, Optional[str]] = {}
        if self.scrape_full_content and sources_to_scrape:
            with ThreadPoolExecutor(max_workers=len(sources_to_scrape)) as scraper_exec:
                scrape_futures = {
                    scraper_exec.submit(scrape_url_content, item["url"], 2500, 5): idx
                    for idx, item in sources_to_scrape
                }
                for fut in as_completed(scrape_futures):
                    idx = scrape_futures[fut]
                    try:
                        scraped_contents[idx] = fut.result()
                    except Exception:
                        scraped_contents[idx] = None

        # Build SourceDocuments
        source_docs: List[SourceDocument] = []
        for idx, item in enumerate(unique_results, start=1):
            content = scraped_contents.get(idx)
            doc = SourceDocument(
                id=idx,
                title=item.get("title", f"Source {idx}"),
                url=item.get("url", ""),
                snippet=item.get("snippet", ""),
                content=content,
                source_type=item.get("source_type", "web"),
                authors=item.get("authors", []),
                published_date=item.get("published_date"),
                relevance_score=0.9
            )
            source_docs.append(doc)

        return source_docs
