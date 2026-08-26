"""Information Retrieval Agent for multi-source search and evidence collection."""

import logging
from typing import List, Optional
from src.models.schemas import ResearchPlan, SourceDocument, ReviewFeedback
from src.tools.search_tools import UnifiedRetriever

logger = logging.getLogger(__name__)


class RetrieverAgent:
    """Retrieval Agent coordinating web, academic, and encyclopedia searches."""

    def __init__(
        self,
        enable_web: bool = True,
        enable_arxiv: bool = True,
        enable_wikipedia: bool = True,
        tavily_api_key: Optional[str] = None
    ):
        self.retriever = UnifiedRetriever(
            enable_web=enable_web,
            enable_arxiv=enable_arxiv,
            enable_wikipedia=enable_wikipedia,
            tavily_api_key=tavily_api_key,
            scrape_full_content=True,
            max_scrape_count=4
        )

    def execute_retrieval(
        self,
        plan: ResearchPlan,
        existing_sources: Optional[List[SourceDocument]] = None,
        feedback: Optional[ReviewFeedback] = None,
        max_sources: int = 15
    ) -> List[SourceDocument]:
        """Executes queries from plan and any revision feedback."""
        queries = []

        # Extract queries from subtopics
        for sub in plan.subtopics:
            queries.extend(sub.search_queries)

        # Include targeted refinement queries if reviewer suggested missing aspects
        if feedback and feedback.missing_aspects:
            for gap in feedback.missing_aspects:
                queries.append(f"{plan.topic} {gap}")

        # Fallback if no queries
        if not queries:
            queries = [plan.topic, f"{plan.topic} overview", f"{plan.topic} research"]

        logger.info(f"RetrieverAgent running {len(queries)} search queries...")
        new_sources = self.retriever.retrieve(queries, max_total_sources=max_sources)

        # Merge with existing sources if this is a refinement loop
        if existing_sources:
            seen_urls = {s.url.rstrip("/") for s in existing_sources if s.url}
            next_id = len(existing_sources) + 1
            merged = list(existing_sources)
            for src in new_sources:
                url_clean = src.url.rstrip("/") if src.url else ""
                if url_clean and url_clean not in seen_urls:
                    seen_urls.add(url_clean)
                    src.id = next_id
                    next_id += 1
                    merged.append(src)
            return merged[:max_sources]

        return new_sources
