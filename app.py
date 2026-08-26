"""Multi-Agent AI Research Assistant - Streamlit Application Entry Point."""

import streamlit as st
import logging
from typing import Dict, Any

from src.ui.styles import apply_custom_styles
from src.ui.components import (
    render_header,
    render_sidebar,
    render_workflow_diagram,
    render_timeline,
    render_report_view,
    render_plan_view,
    render_sources_view,
    render_review_view,
    render_qa_chat
)
from src.utils.llm_factory import get_llm
from src.agents.retriever import RetrieverAgent
from src.graph.workflow import create_agent_graph
from src.models.schemas import FinalReport, ResearchPlan, ReviewFeedback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit Page Configuration
st.set_page_config(
    page_title="Multi-Agent AI Research Assistant",
    page_icon="??",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply UI CSS Styles
apply_custom_styles()


def init_session_state():
    """Initializes Streamlit session state variables."""
    if "final_report" not in st.session_state:
        st.session_state.final_report = None
    if "research_plan" not in st.session_state:
        st.session_state.research_plan = None
    if "sources" not in st.session_state:
        st.session_state.sources = []
    if "review_feedback" not in st.session_state:
        st.session_state.review_feedback = None
    if "timeline_events" not in st.session_state:
        st.session_state.timeline_events = []
    if "is_researching" not in st.session_state:
        st.session_state.is_researching = False
    if "active_llm" not in st.session_state:
        st.session_state.active_llm = None


def main():
    init_session_state()
    render_header()

    # Sidebar configuration
    config = render_sidebar()

    # Workflow architecture diagram
    render_workflow_diagram()

    # Example topics for quick start
    example_topics = [
        "Emerging Architectures in Multi-Agent Reasoning & LangGraph Orchestration",
        "Quantum Computing Breakthroughs in Molecular Drug Discovery",
        "Autonomous AI Agents in Real-Time Cybersecurity Threat Mitigation",
        "Neuromorphic Computing Hardware and Spiking Neural Networks"
    ]

    st.markdown("### ?? Define Research Scope")
    
    col_input, col_preset = st.columns([3, 1])
    with col_preset:
        preset_choice = st.selectbox("Or select an example topic:", options=["-- Custom Topic --"] + example_topics)

    default_topic = preset_choice if preset_choice != "-- Custom Topic --" else ""

    with col_input:
        topic_input = st.text_input(
            "Research Topic or Question:",
            value=default_topic,
            placeholder="e.g., Multi-Agent Systems in Healthcare Diagnostics",
            help="Enter the technical or scientific domain you want the multi-agent squad to investigate."
        )

    with st.expander("?? Additional Research Angles & Focus Directives (Optional)", expanded=False):
        user_guidance = st.text_area(
            "Special focus areas, hypotheses, or constraints:",
            placeholder="e.g. Compare LangGraph with CrewAI and AutoGen; include empirical latency and token cost benchmarks."
        )

    start_btn = st.button("?? Launch Multi-Agent Research Squad", type="primary", use_container_width=True)

    if start_btn:
        if not topic_input.strip():
            st.error("Please enter a valid research topic to begin.")
            return

        st.session_state.is_researching = True
        st.session_state.timeline_events = []
        st.session_state.final_report = None
        st.session_state.chat_history = []

        # Instantiate LLM and Retriever
        llm = get_llm(
            provider=config["provider"],
            model_name=config["model_name"],
            api_key=config["api_key"]
        )
        st.session_state.active_llm = llm

        retriever_agent = RetrieverAgent(
            enable_web=config["enable_web"],
            enable_arxiv=config["enable_arxiv"],
            enable_wikipedia=config["enable_wikipedia"],
            tavily_api_key=config["tavily_key"]
        )

        app_graph = create_agent_graph(llm=llm, retriever_agent=retriever_agent)

        initial_state = {
            "topic": topic_input.strip(),
            "depth": config["depth"],
            "enabled_sources": [
                s for s, enabled in [
                    ("web", config["enable_web"]),
                    ("arxiv", config["enable_arxiv"]),
                    ("wikipedia", config["enable_wikipedia"])
                ] if enabled
            ],
            "user_guidance": user_guidance.strip() if user_guidance else None,
            "plan": None,
            "sources": [],
            "draft_sections": [],
            "review_feedback": None,
            "final_report": None,
            "iteration_count": 1,
            "max_iterations": config["max_iterations"],
            "timeline_events": [],
            "error": None
        }

        # Progress tracking container
        progress_container = st.empty()
        status_box = st.status("?? Research Squad Active: Coordinating Agents...", expanded=True)

        try:
            with status_box:
                st.write("?? **Step 1/5:** Lead Planner decomposing topic into analytical pillars...")
                # Invoke the LangGraph workflow
                final_state = app_graph.invoke(initial_state)

                st.write("?? **Step 2/5:** Information Retriever gathering & verifying multi-source evidence...")
                st.write("?? **Step 3/5:** Research Writer synthesizing technical sections with citations...")
                st.write("??? **Step 4/5:** Fact-Checker & Reviewer performing quality audit...")
                st.write("?? **Step 5/5:** Executive Formatter packaging complete report...")

                status_box.update(label="? Multi-Agent Investigation Complete!", state="complete", expanded=False)

            st.session_state.final_report = final_state.get("final_report")
            st.session_state.research_plan = final_state.get("plan")
            st.session_state.sources = final_state.get("sources", [])
            st.session_state.review_feedback = final_state.get("review_feedback")
            st.session_state.timeline_events = final_state.get("timeline_events", [])
            st.session_state.is_researching = False

        except Exception as e:
            logger.error(f"Error executing multi-agent graph: {e}", exc_info=True)
            status_box.update(label="? Error encountered during execution", state="error")
            st.error(f"Execution Error: {e}")
            st.session_state.is_researching = False

    # Display results if available
    if st.session_state.final_report:
        st.divider()

        # Display timeline above or in tabs
        with st.expander("? Live Agent Activity Timeline & Thought Log", expanded=False):
            render_timeline(st.session_state.timeline_events)

        # Tab navigation for comprehensive research results
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "?? Comprehensive Report",
            "??? Research Strategy & Plan",
            "?? Evidence Catalog",
            "??? Quality & Fact-Check Audit",
            "?? Interactive Consultation (Q&A)"
        ])

        with tab1:
            render_report_view(st.session_state.final_report)

        with tab2:
            if st.session_state.research_plan:
                render_plan_view(st.session_state.research_plan)

        with tab3:
            render_sources_view(st.session_state.sources)

        with tab4:
            render_review_view(
                st.session_state.review_feedback,
                iterations=len(st.session_state.timeline_events)
            )

        with tab5:
            llm_for_qa = st.session_state.active_llm or get_llm("mock")
            render_qa_chat(st.session_state.final_report, llm_for_qa)


if __name__ == "__main__":
    main()
