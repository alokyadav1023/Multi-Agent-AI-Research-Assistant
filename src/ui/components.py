"""Streamlit UI components for Multi-Agent AI Research Assistant."""

import os
import streamlit as st
from typing import Dict, Any, List, Optional
from src.models.schemas import FinalReport, ResearchPlan, SourceDocument, ReviewFeedback, QAResponse
from src.utils.exporter import export_to_markdown, export_to_html, export_to_pdf, export_to_json
from src.agents.qa_agent import QAAgent
from src.ui.styles import CUSTOM_CSS


def apply_custom_styles():
    """Injects custom CSS into the Streamlit app."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_header():
    """Renders the top banner header."""
    st.markdown("""
    <div class="main-header">
        <h1>🧠 Multi-Agent AI Research Assistant</h1>
        <p>⚡ Ultra-Fast Autonomous Research Squad powered by NVIDIA NIM & LangGraph</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar() -> Dict[str, Any]:
    """Renders the sidebar configuration options and returns settings dict."""
    with st.sidebar:
        st.header("⚙️ System Configuration")

        # Provider Selection
        provider = st.selectbox(
            "LLM Provider",
            options=["NVIDIA NIM (Fast)", "Google Gemini", "OpenAI", "Groq", "Anthropic", "Ollama", "Offline / Mock Mode"],
            index=0,
            help="Choose the LLM engine for agent reasoning."
        )

        provider_key_map = {
            "NVIDIA NIM (Fast)": "nvidia",
            "Google Gemini": "google",
            "OpenAI": "openai",
            "Groq": "groq",
            "Anthropic": "anthropic",
            "Ollama": "ollama",
            "Offline / Mock Mode": "mock"
        }
        selected_provider = provider_key_map[provider]

        # Model Name Selection
        if selected_provider == "nvidia":
            model_display_map = {
                "⚡ nvidia/nemotron-3-nano-30b-a3b (Ultra-Fast 0.4s)": "nvidia/nemotron-3-nano-30b-a3b",
                "⚡ meta/llama-3.2-11b-vision-instruct (Fast 0.6s)": "meta/llama-3.2-11b-vision-instruct",
                "🧠 nvidia/nemotron-3-super-120b-a12b (Deep Reasoning 120B)": "nvidia/nemotron-3-super-120b-a12b",
                "nvidia/nemotron-3.5-lightning-30b-a3b": "nvidia/nemotron-3.5-lightning-30b-a3b"
            }
            selected_display = st.selectbox(
                "NVIDIA NIM Model",
                options=list(model_display_map.keys()),
                index=0,
                help="High-speed reasoning and synthesis models hosted on NVIDIA NIM."
            )
            model_name = model_display_map[selected_display]
            env_nvidia_key = os.getenv("NVIDIA_API_KEY", "")
            api_key = st.text_input("NVIDIA NIM API Key", value=env_nvidia_key, type="password", help="API key from integrate.api.nvidia.com.")
        elif selected_provider == "google":
            model_name = st.selectbox(
                "Model Name",
                options=["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash", "gemini-1.5-flash"],
                index=0
            )
            api_key = st.text_input("Google Gemini API Key", type="password", help="Leave blank to use GOOGLE_API_KEY from environment.")
        elif selected_provider == "openai":
            model_name = st.selectbox(
                "Model Name",
                options=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
                index=0
            )
            api_key = st.text_input("OpenAI API Key", type="password", help="Leave blank to use OPENAI_API_KEY from environment.")
        elif selected_provider == "groq":
            model_name = st.selectbox(
                "Model Name",
                options=["llama-3.3-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"],
                index=0
            )
            api_key = st.text_input("Groq API Key", type="password", help="Leave blank to use GROQ_API_KEY from environment.")
        elif selected_provider == "anthropic":
            model_name = st.selectbox(
                "Model Name",
                options=["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"],
                index=0
            )
            api_key = st.text_input("Anthropic API Key", type="password", help="Leave blank to use ANTHROPIC_API_KEY from environment.")
        elif selected_provider == "ollama":
            model_name = st.text_input("Ollama Model Name", value="llama3")
            api_key = ""
        else:  # Mock
            model_name = "mock-research-model"
            api_key = ""
            st.info("💡 Mock Mode runs without any external API keys for demonstration and testing.")

        st.divider()

        # Research Depth
        st.subheader("🎯 Research Parameters")
        depth_option = st.radio(
            "Research Depth",
            options=["Brief Overview (Fast)", "Standard Research", "Deep Comprehensive Investigation"],
            index=0,
            help="Controls the number of subtopics and research breadth."
        )
        depth_map = {
            "Brief Overview (Fast)": "brief",
            "Standard Research": "standard",
            "Deep Comprehensive Investigation": "deep"
        }
        depth = depth_map[depth_option]

        # Information Sources
        st.subheader("🌐 Information Sources")
        enable_web = st.checkbox("Web Search (DuckDuckGo)", value=True)
        enable_arxiv = st.checkbox("Academic Papers (ArXiv)", value=True)
        enable_wikipedia = st.checkbox("Encyclopedia (Wikipedia)", value=True)
        tavily_key = st.text_input("Tavily API Key (Optional)", type="password", help="Enable AI search if key is provided.")

        # Refinement loop
        max_iterations = st.slider(
            "Max Refinement Iterations",
            min_value=1,
            max_value=3,
            value=1,
            help="Maximum loops allowed for fact-checker self-correction."
        )

        st.divider()
        st.caption("Multi-Agent AI Research Assistant v1.2 • Optimized for NVIDIA NIM")

        return {
            "provider": selected_provider,
            "model_name": model_name,
            "api_key": api_key,
            "depth": depth,
            "enable_web": enable_web,
            "enable_arxiv": enable_arxiv,
            "enable_wikipedia": enable_wikipedia,
            "tavily_key": tavily_key,
            "max_iterations": max_iterations
        }


def render_workflow_diagram():
    """Displays the multi-agent orchestration architecture diagram."""
    with st.expander("🗺️ Multi-Agent Architecture & Graph Workflow", expanded=False):
        st.markdown("""
```mermaid
flowchart LR
    UserQuery([User Research Topic]) --> Planner[1. Lead Planner (NVIDIA Nemotron)]
    Planner -->|Subtopics & Parallel Queries| Retriever[2. Concurrent Retriever]
    Retriever -->|Verified Sources| Writer[3. Parallel Writer (NVIDIA Nemotron)]
    Writer -->|Draft Sections| Reviewer[4. Fact-Checker & Reviewer]
    
    Reviewer -->|Iterate / Needs Refinement| Retriever
    Reviewer -->|Approved| Formatter[5. Executive Formatter]
    Formatter --> FinalReport([Comprehensive Report & Exports])
```
        """)


def render_timeline(events: List[Dict[str, Any]]):
    """Renders live agent execution events."""
    st.subheader("⚡ Agent Activity Timeline")
    badge_classes = {
        "Lead Planner": "badge-planner",
        "Information Retriever": "badge-retriever",
        "Research Writer": "badge-writer",
        "Quality Reviewer": "badge-reviewer",
        "Executive Formatter": "badge-formatter"
    }

    for ev in events:
        agent_name = ev.get("agent", "Agent")
        badge_cls = badge_classes.get(agent_name, "badge-planner")
        step = ev.get("step", "")
        msg = ev.get("message", "")
        ts = ev.get("timestamp", "")

        st.markdown(f"""
        <div class="agent-card">
            <div class="agent-title">
                <span class="agent-badge {badge_cls}">{agent_name}</span>
                <span>{step}</span>
                <span style="font-size:0.8rem; color:#94a3b8; margin-left:auto;">{ts}</span>
            </div>
            <div style="font-size:0.92rem; margin-top:6px; color:#334155;">
                {msg}
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_report_view(report: FinalReport):
    """Renders the complete research report with download actions."""
    st.markdown(f"""
    <div style="margin-bottom: 16px;">
        <span class="metric-pill">📊 Words: ~{report.total_words}</span>
        <span class="metric-pill">📚 Sources Cited: {len(report.sources)}</span>
        <span class="metric-pill">📑 Sections: {len(report.sections)}</span>
        <span class="metric-pill">⏱️ Generated: {report.generation_timestamp}</span>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    md_data = export_to_markdown(report)
    html_data = export_to_html(report)
    json_data = export_to_json(report)

    with c1:
        st.download_button(
            "📥 Download Markdown",
            data=md_data,
            file_name=f"research_report_{report.topic.lower().replace(' ', '_')[:30]}.md",
            mime="text/markdown",
            use_container_width=True
        )
    with c2:
        try:
            pdf_bytes = export_to_pdf(report)
            st.download_button(
                "📄 Download PDF",
                data=pdf_bytes,
                file_name=f"research_report_{report.topic.lower().replace(' ', '_')[:30]}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        except Exception:
            st.button("PDF Export (Unavailable)", disabled=True, use_container_width=True)

    with c3:
        st.download_button(
            "🌐 Download HTML",
            data=html_data,
            file_name=f"research_report_{report.topic.lower().replace(' ', '_')[:30]}.html",
            mime="text/html",
            use_container_width=True
        )
    with c4:
        st.download_button(
            "📦 Download JSON",
            data=json_data,
            file_name=f"research_report_{report.topic.lower().replace(' ', '_')[:30]}.json",
            mime="application/json",
            use_container_width=True
        )

    st.divider()

    st.info(f"**Executive Summary**\n\n{report.executive_summary}")

    if report.key_takeaways:
        st.subheader("💡 Key Strategic Takeaways")
        cols = st.columns(min(len(report.key_takeaways), 3))
        for i, point in enumerate(report.key_takeaways):
            with cols[i % len(cols)]:
                st.success(f"**Insight {i+1}**\n\n{point}")

    st.divider()

    for i, section in enumerate(report.sections, start=1):
        st.subheader(f"{i}. {section.section_title}")
        st.markdown(section.content)

        if section.key_findings:
            with st.expander(f"🔍 Core Findings ({section.section_title})", expanded=False):
                for f in section.key_findings:
                    st.markdown(f"- {f}")
        st.markdown("")

    if report.future_outlook:
        st.subheader("🔭 Future Outlook & Emerging Trajectory")
        st.markdown(report.future_outlook)

    if report.methodology_notes:
        with st.expander("🔬 Research Methodology & Agent Verification", expanded=False):
            st.markdown(report.methodology_notes)


def render_plan_view(plan: ResearchPlan):
    """Renders the structured research plan and task decomposition."""
    st.markdown(f"### 📋 Research Strategy: {plan.topic}")
    st.write(f"**Scope Overview:** {plan.overview}")
    st.write(f"**Target Audience:** {plan.target_audience}")

    st.subheader("Analytical Pillars & Decomposed Subtasks")
    for i, sub in enumerate(plan.subtopics, start=1):
        with st.container():
            st.markdown(f"#### Pillar {i}: {sub.title}")
            st.write(f"*{sub.description}*")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Targeted Search Queries:**")
                for q in sub.search_queries:
                    st.code(q, language="text")
            with c2:
                st.markdown("**Key Analytical Questions:**")
                for k in sub.key_questions:
                    st.markdown(f"- {k}")
            st.divider()


def render_sources_view(sources: List[SourceDocument]):
    """Renders the verified sources catalog with inspection cards."""
    st.subheader(f"📚 Evidence Catalog ({len(sources)} Sources)")
    
    type_filter = st.selectbox("Filter by Source Type", options=["All", "Web", "ArXiv", "Wikipedia"])
    
    for s in sources:
        if type_filter != "All" and s.source_type.lower() != type_filter.lower():
            continue

        with st.container():
            type_label = s.source_type.upper()
            author_text = f" • By {', '.join(s.authors)}" if s.authors else ""
            date_text = f" • {s.published_date}" if s.published_date else ""

            st.markdown(f"""
            <div class="source-card">
                <div>
                    <span class="agent-badge badge-retriever">[{s.id}] {type_label}</span>
                    <a href="{s.url}" target="_blank">{s.title}</a>
                    <span style="font-size:0.8rem; color:#64748b;">{author_text}{date_text}</span>
                </div>
                <div class="source-snippet">{s.snippet}</div>
            </div>
            """, unsafe_allow_html=True)

            if s.content:
                with st.expander(f"📄 Full Extracted Text Preview ([{s.id}] {s.title[:40]}...)", expanded=False):
                    st.text(s.content[:1500] + ("..." if len(s.content) > 1500 else ""))


def render_review_view(feedback: Optional[ReviewFeedback], iterations: int):
    """Renders the quality audit and review notes."""
    if not feedback:
        st.info("Review data will be available once the Reviewer Agent completes its audit.")
        return

    st.subheader("🛡️ Quality Audit & Fact-Check Review")

    score = feedback.overall_score
    col1, col2, col3 = st.columns(3)
    col1.metric("Quality Score", f"{score}/10.0")
    col2.metric("Review Status", "Approved" if feedback.is_approved else "Needs Refinement")
    col3.metric("Iterations Completed", f"{iterations}")

    st.markdown("### Detailed Critique")
    st.write(feedback.critique)

    st.markdown("### Factual Grounding Assessment")
    st.write(feedback.factual_accuracy_notes)

    if feedback.missing_aspects:
        st.markdown("### Identified Gaps & Additional Angles")
        for gap in feedback.missing_aspects:
            st.markdown(f"- {gap}")

    if feedback.suggested_refinements:
        st.markdown("### Recommended Refinements")
        for ref in feedback.suggested_refinements:
            st.markdown(f"- {ref}")


def render_qa_chat(report: FinalReport, llm: Any):
    """Interactive Q&A grounded in the research report."""
    st.subheader("💬 Interactive Research Consultation")
    st.caption("Ask questions grounded strictly in the generated report and verified sources.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                st.caption(f"Sources referenced: {msg['sources']}")

    user_q = st.chat_input("Ask a specific question about this research...")
    if user_q:
        st.session_state.chat_history.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        with st.chat_message("assistant"):
            with st.spinner("Grounded Research Consultant analyzing with NVIDIA NIM..."):
                qa_agent = QAAgent(llm)
                qa_res: QAResponse = qa_agent.answer_question(user_q, report)

                ans_text = qa_res.answer
                st.markdown(ans_text)
                if qa_res.sources_referenced:
                    src_str = ", ".join([f"[{sid}]" for sid in qa_res.sources_referenced])
                    st.caption(f"Referenced Sources: {src_str} (Confidence: {int(qa_res.confidence_score*100)}%)")

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": ans_text,
                    "sources": qa_res.sources_referenced
                })
