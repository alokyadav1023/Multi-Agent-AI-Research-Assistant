"""End-to-end tests for LangGraph state machine execution."""

import pytest
from src.utils.llm_factory import MockLLM
from src.agents.retriever import RetrieverAgent
from src.graph.workflow import create_agent_graph
from src.models.schemas import FinalReport, ResearchPlan


def test_workflow_execution_with_mock():
    llm = MockLLM()
    retriever_agent = RetrieverAgent(
        enable_web=False,
        enable_arxiv=False,
        enable_wikipedia=False
    )
    graph = create_agent_graph(llm=llm, retriever_agent=retriever_agent)

    initial_state = {
        "topic": "Graph Orchestration in Multi-Agent AI Systems",
        "depth": "brief",
        "enabled_sources": ["web"],
        "user_guidance": None,
        "plan": None,
        "sources": [],
        "draft_sections": [],
        "review_feedback": None,
        "final_report": None,
        "iteration_count": 1,
        "max_iterations": 2,
        "timeline_events": [],
        "error": None
    }

    final_state = graph.invoke(initial_state)

    assert final_state is not None
    assert final_state["plan"] is not None
    assert isinstance(final_state["plan"], ResearchPlan)
    assert final_state["final_report"] is not None
    assert isinstance(final_state["final_report"], FinalReport)
    assert len(final_state["timeline_events"]) >= 4
