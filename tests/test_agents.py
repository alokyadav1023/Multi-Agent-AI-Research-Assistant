"""Tests for individual research agents using MockLLM."""

import pytest
from src.utils.llm_factory import MockLLM
from src.agents.planner import PlannerAgent
from src.agents.retriever import RetrieverAgent
from src.agents.writer import WriterAgent
from src.agents.reviewer import ReviewerAgent
from src.agents.qa_agent import QAAgent
from src.models.schemas import SourceDocument, ResearchPlan, SubTopic, SectionDraft, FinalReport


@pytest.fixture
def mock_llm():
    return MockLLM()


def test_planner_agent(mock_llm):
    planner = PlannerAgent(mock_llm)
    plan = planner.generate_plan(topic="Multi-Agent AI Systems", depth="standard")

    assert isinstance(plan, ResearchPlan)
    assert len(plan.subtopics) >= 1
    assert all(isinstance(sub, SubTopic) for sub in plan.subtopics)


def test_writer_agent(mock_llm):
    writer = WriterAgent(mock_llm)
    sub = SubTopic(
        title="State Machine Graphs",
        description="Analysis of cyclical graphs",
        search_queries=["LangGraph state machine"],
        key_questions=["How does cyclic routing work?"]
    )
    sources = [
        SourceDocument(
            id=1,
            title="LangGraph Docs",
            url="https://langchain.com",
            snippet="Cyclic execution graphs"
        )
    ]
    draft = writer.write_section(sub, "Multi-Agent Systems", sources)

    assert isinstance(draft, SectionDraft)
    assert isinstance(draft.section_title, str)
    assert len(draft.section_title) > 0
    assert len(draft.content) > 20
    assert isinstance(draft.citations_used, list)


def test_reviewer_agent(mock_llm):
    reviewer = ReviewerAgent(mock_llm)
    plan = ResearchPlan(
        topic="Multi-Agent AI Systems",
        overview="Overview",
        target_audience="Engineers",
        subtopics=[],
        expected_outcomes=["Outcome 1"]
    )
    feedback = reviewer.review_draft(
        plan=plan,
        sources=[],
        draft_sections=[],
        current_iteration=1,
        max_iterations=2
    )

    assert feedback.overall_score >= 1.0
    assert isinstance(feedback.is_approved, bool)


def test_qa_agent(mock_llm):
    qa = QAAgent(mock_llm)
    report = FinalReport(
        title="Research on Agents",
        topic="Multi-Agent Systems",
        executive_summary="Executive summary text.",
        key_takeaways=["Autonomous agents excel at complex workflows."],
        table_of_contents=["Section 1"],
        sections=[],
        future_outlook="",
        methodology_notes="",
        sources=[
            SourceDocument(id=1, title="Doc 1", url="https://example.com", snippet="Snippet")
        ],
        generation_timestamp="2026-08-25 12:00:00",
        total_words=200
    )
    qa_res = qa.answer_question("How do multi-agent systems work?", report)

    assert qa_res.answer is not None
    assert len(qa_res.answer) > 10
    assert 0.0 <= qa_res.confidence_score <= 1.0
