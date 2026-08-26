"""State definition for LangGraph Multi-Agent Research Assistant."""

from typing import TypedDict, List, Optional, Dict, Any
from src.models.schemas import (
    ResearchPlan,
    SourceDocument,
    SectionDraft,
    ReviewFeedback,
    FinalReport
)


class ResearchState(TypedDict):
    """Typed state container passed across nodes in the research graph."""
    topic: str
    depth: str  # "brief", "standard", "deep"
    enabled_sources: List[str]  # e.g. ["web", "arxiv", "wikipedia"]
    user_guidance: Optional[str]
    plan: Optional[ResearchPlan]
    sources: List[SourceDocument]
    draft_sections: List[SectionDraft]
    review_feedback: Optional[ReviewFeedback]
    final_report: Optional[FinalReport]
    iteration_count: int
    max_iterations: int
    timeline_events: List[Dict[str, Any]]
    error: Optional[str]
