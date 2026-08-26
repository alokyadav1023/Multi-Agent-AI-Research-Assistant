"""Tests for Pydantic schemas and serialization."""

import pytest
from src.models.schemas import (
    SubTopic,
    ResearchPlan,
    SourceDocument,
    SectionDraft,
    ReviewFeedback,
    FinalReport,
    QAResponse
)


def test_research_plan_creation():
    sub = SubTopic(
        title="State Machine Graphs",
        description="Analysis of cyclical graphs in multi-agent workflows",
        search_queries=["LangGraph state machine"],
        key_questions=["How does cyclic routing work?"]
    )
    plan = ResearchPlan(
        topic="Multi-Agent Orchestration",
        overview="Comprehensive analysis",
        target_audience="Researchers",
        subtopics=[sub],
        expected_outcomes=["Detailed blueprint"]
    )

    assert plan.topic == "Multi-Agent Orchestration"
    assert len(plan.subtopics) == 1
    assert plan.subtopics[0].title == "State Machine Graphs"


def test_source_document_model():
    doc = SourceDocument(
        id=1,
        title="LangGraph Documentation",
        url="https://langchain-ai.github.io/langgraph/",
        snippet="Cyclical graph orchestration for agentic workflows.",
        source_type="web"
    )
    assert doc.id == 1
    assert doc.source_type == "web"
    assert doc.relevance_score == 0.85 or doc.relevance_score == 0.8


def test_review_feedback_model():
    feedback = ReviewFeedback(
        is_approved=True,
        overall_score=9.0,
        critique="Excellently grounded.",
        factual_accuracy_notes="Citations are consistent.",
        missing_aspects=[],
        suggested_refinements=["Format TOC"],
        revision_needed=False
    )
    assert feedback.is_approved is True
    assert feedback.overall_score == 9.0
    assert feedback.revision_needed is False


def test_final_report_model():
    report = FinalReport(
        title="State of AI Agents",
        topic="AI Agents",
        executive_summary="Executive summary text.",
        key_takeaways=["Key insight 1"],
        table_of_contents=["Section 1"],
        sections=[
            SectionDraft(
                section_title="Section 1",
                content="Section content with citation [1].",
                citations_used=[1],
                key_findings=["Finding 1"]
            )
        ],
        future_outlook="Promising future.",
        methodology_notes="Multi-agent synthesis.",
        sources=[
            SourceDocument(
                id=1,
                title="Source 1",
                url="https://example.com",
                snippet="Snippet 1"
            )
        ],
        generation_timestamp="2026-08-25 12:00:00",
        total_words=450
    )
    assert len(report.sections) == 1
    assert report.total_words == 450
    assert report.sources[0].id == 1
