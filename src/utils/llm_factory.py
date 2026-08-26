"""LLM Factory for initializing multi-provider LLMs with structured outputs and offline mock support."""

import os
import json
import re
import logging
from typing import Type, TypeVar, Optional, Any, Dict, List
from pydantic import BaseModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class MockLLM(BaseChatModel):
    """Mock LLM for offline testing, demos, or keyless evaluation."""

    model_name: str = "mock-research-model"

    @property
    def _llm_type(self) -> str:
        return "mock-research-assistant"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any
    ) -> ChatResult:
        last_msg = messages[-1].content if messages else ""

        if "ResearchPlan" in last_msg or "subtopics" in last_msg or "Task Decomposition" in last_msg:
            content = json.dumps({
                "topic": "Multi-Agent AI Systems & Orchestration",
                "overview": "Comprehensive investigation into multi-agent AI architectures, coordination mechanisms, memory sharing, and practical applications in automated research.",
                "target_audience": "AI Engineers, Researchers, and System Architects",
                "subtopics": [
                    {
                        "title": "Architectural Foundations & Graph Orchestration",
                        "description": "State-space representations, cyclical workflows, and execution DAGs in modern agent systems like LangGraph.",
                        "search_queries": ["LangGraph multi agent cyclical architecture", "multi agent graph orchestration state transitions"],
                        "key_questions": ["How do cyclical graphs outperform linear chains?", "What role does shared state play?"]
                    },
                    {
                        "title": "Information Retrieval & Multi-Source Evidence Aggregation",
                        "description": "Techniques for combining web searches, academic preprints, and knowledge bases with deduplication.",
                        "search_queries": ["multi agent retrieval augmented generation arxiv", "agentic search information synthesis"],
                        "key_questions": ["How do specialized retrieval agents minimize hallucinations?"]
                    },
                    {
                        "title": "Iterative Refinement & Fact-Checking Loops",
                        "description": "Self-correction, critique mechanisms, and confidence scoring in agent teams.",
                        "search_queries": ["agentic self reflection review loop", "fact checking LLM output multi-agent"],
                        "key_questions": ["How is convergence guaranteed during feedback loops?"]
                    }
                ],
                "expected_outcomes": [
                    "Detailed breakdown of multi-agent patterns",
                    "Empirical comparison of orchestration frameworks",
                    "Production deployment guidelines"
                ]
            })
        elif "ReviewFeedback" in last_msg or "factual_accuracy_notes" in last_msg or "Quality Auditor" in last_msg:
            content = json.dumps({
                "is_approved": True,
                "overall_score": 9.2,
                "critique": "The synthesized research report is well-structured, technically rigorous, and appropriately grounded in the retrieved sources.",
                "factual_accuracy_notes": "All claims are supported with consistent [1], [2] citations. Key technical terms are accurately defined.",
                "missing_aspects": ["Could expand on edge-device local agent execution in future work."],
                "suggested_refinements": ["Add summary comparison table for orchestration frameworks."],
                "revision_needed": False
            })
        elif "SectionDraft" in last_msg or "AVAILABLE SOURCE CATALOG" in last_msg:
            content = json.dumps({
                "section_title": "Architectural Foundations of Multi-Agent Systems",
                "content": "Multi-agent systems represent a fundamental shift from monolithic prompt engineering to distributed cognitive architectures [1]. By decomposing complex goals into granular subtasks handled by autonomous agents, systems achieve higher reliability and specialized reasoning [2]. Frameworks like LangGraph utilize cyclical state graphs to model loops, human-in-the-loop interactions, and self-correction [3].",
                "citations_used": [1, 2, 3],
                "key_findings": [
                    "Cyclical state management enables self-healing reasoning loops.",
                    "Specialized agents reduce prompt congestion and hallucination rates."
                ]
            })
        elif "FinalReport" in last_msg:
            content = json.dumps({
                "title": "State-of-the-Art in Multi-Agent AI Research & System Architectures",
                "topic": "Multi-Agent AI Systems",
                "executive_summary": "This comprehensive report analyzes the evolution of Multi-Agent Systems (MAS) powered by modern Large Language Models. We examine stateful graph orchestration, multi-source information retrieval, iterative critique loops, and practical industry implementations.",
                "key_takeaways": [
                    "Graph-based orchestration provides superior resilience over sequential chains.",
                    "Specialized roles (Planner, Retriever, Writer, Critic) dramatically enhance research depth.",
                    "Grounding outputs with dynamic search reduces hallucination risk by over 80%."
                ],
                "table_of_contents": [
                    "Executive Summary",
                    "Architectural Foundations & Graph Orchestration",
                    "Information Retrieval & Evidence Aggregation",
                    "Iterative Refinement & Fact-Checking",
                    "Strategic Implications & Future Outlook"
                ],
                "sections": [
                    {
                        "section_title": "Architectural Foundations & Graph Orchestration",
                        "content": "Traditional LLM pipelines often struggle with multi-step reasoning due to context degradation. Multi-agent architectures solve this by delegating tasks across specialized agents [1]. LangGraph represents agent interactions as state machines, allowing conditional branching and cyclic feedback [2].",
                        "citations_used": [1, 2],
                        "key_findings": ["Cyclical graphs enable autonomous error recovery."]
                    },
                    {
                        "section_title": "Information Retrieval & Evidence Aggregation",
                        "content": "Modern research assistants integrate multi-channel search across web endpoints, ArXiv preprints, and structured knowledge bases [3]. Automated content scraping extracts core prose while filtering noise [4].",
                        "citations_used": [3, 4],
                        "key_findings": ["Multi-modal source integration ensures comprehensive coverage."]
                    }
                ],
                "future_outlook": "The future of multi-agent systems lies in multimodal reasoning, verifiable tool execution, and local-first edge deployments.",
                "methodology_notes": "Synthesized across academic literature and real-time search queries utilizing a 4-agent LangGraph workflow.",
                "sources": [],
                "generation_timestamp": "2026-08-25T12:00:00Z",
                "total_words": 1250,
                "review_summary": "Verified and approved with a quality score of 9.2/10."
            })
        elif "QAResponse" in last_msg or "Consultation" in last_msg or "User Question" in last_msg:
            content = json.dumps({
                "answer": "Based on the research findings, multi-agent architectures achieve superior accuracy and reliability by delegating subtasks to specialized agents and utilizing cyclical state graphs for iterative self-correction and fact-checking [1][2].",
                "sources_referenced": [1, 2],
                "confidence_score": 0.95
            })
        else:
            content = "Multi-Agent AI architectures allow coordinated specialization, stateful memory management, and robust task decomposition across complex research workflows."

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])


def get_llm(
    provider: str = "nvidia",
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.2
) -> BaseChatModel:
    """Factory function to instantiate chat model for specified provider."""
    provider = (provider or "").lower().strip()

    if provider in ["mock", "offline", "demo"]:
        return MockLLM()

    elif provider in ["nvidia", "nim", "nemotron"]:
        key = api_key or os.getenv("NVIDIA_API_KEY")
        if not key:
            logger.warning("No NVIDIA API key provided; falling back to MockLLM.")
            return MockLLM()
        from langchain_openai import ChatOpenAI
        m = model_name or os.getenv("DEFAULT_NVIDIA_MODEL", "nvidia/nemotron-3-super-120b-a12b")
        base_url = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
        return ChatOpenAI(
            model=m,
            api_key=key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=8192
        )

    elif provider in ["google", "gemini"]:
        key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not key:
            logger.warning("No Google API key provided; falling back to MockLLM.")
            return MockLLM()
        from langchain_google_genai import ChatGoogleGenerativeAI
        m = model_name or "gemini-2.5-flash"
        return ChatGoogleGenerativeAI(
            model=m,
            google_api_key=key,
            temperature=temperature
        )

    elif provider in ["openai", "gpt"]:
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            logger.warning("No OpenAI API key provided; falling back to MockLLM.")
            return MockLLM()
        from langchain_openai import ChatOpenAI
        m = model_name or "gpt-4o-mini"
        return ChatOpenAI(
            model=m,
            api_key=key,
            temperature=temperature
        )

    elif provider in ["groq"]:
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            logger.warning("No Groq API key provided; falling back to MockLLM.")
            return MockLLM()
        from langchain_groq import ChatGroq
        m = model_name or "llama-3.3-70b-versatile"
        return ChatGroq(
            model=m,
            groq_api_key=key,
            temperature=temperature
        )

    elif provider in ["anthropic", "claude"]:
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            logger.warning("No Anthropic API key provided; falling back to MockLLM.")
            return MockLLM()
        from langchain_anthropic import ChatAnthropic
        m = model_name or "claude-3-5-sonnet-20241022"
        return ChatAnthropic(
            model=m,
            api_key=key,
            temperature=temperature
        )

    elif provider in ["ollama"]:
        from langchain_community.chat_models import ChatOllama
        m = model_name or "llama3"
        return ChatOllama(model=m, temperature=temperature)

    logger.warning(f"Unrecognized provider '{provider}'; defaulting to MockLLM.")
    return MockLLM()


def extract_json_payload(text: str) -> Optional[Dict[str, Any]]:
    """Robustly extracts JSON dictionary from model output text."""
    if not text:
        return None

    try:
        data = json.loads(text.strip())
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    json_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_block:
        try:
            data = json.loads(json_block.group(1).strip())
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidate = text[first_brace : last_brace + 1]
        try:
            data = json.loads(candidate.strip())
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    return None
