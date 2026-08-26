"""Lead Planner & Task Decomposition Agent."""

import logging
from typing import Dict, Any, List
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from src.models.schemas import ResearchPlan, SubTopic
from src.utils.llm_factory import extract_json_payload

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = """You are the Lead Research Strategist & Task Decomposition Agent in a multi-agent system.
Your objective is to decompose a broad or specialized research topic into a rigorous, well-structured investigation plan.

Guidelines:
1. Break down the topic into logical, distinct analytical pillars (subtopics).
2. For each subtopic, define clear research objectives, formulate 2-3 targeted search queries (optimizing for academic/web search engines), and outline key questions to investigate.
3. Tailor the scope based on the requested research depth:
   - 'brief': 2-3 focused subtopics.
   - 'standard': 3-4 comprehensive subtopics.
   - 'deep': 4-6 exhaustive subtopics covering fundamentals, advanced mechanisms, real-world case studies, tradeoffs, and future trajectory.
4. Output your response as a valid JSON object matching this schema:
{
  "topic": "string",
  "overview": "string",
  "target_audience": "string",
  "subtopics": [
    {
      "title": "string",
      "description": "string",
      "search_queries": ["query 1", "query 2"],
      "key_questions": ["question 1", "question 2"]
    }
  ],
  "expected_outcomes": ["outcome 1", "outcome 2"]
}
"""


class PlannerAgent:
    """Planner Agent that decomposes research tasks into structured execution plans."""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def generate_plan(self, topic: str, depth: str = "standard", user_guidance: str = "") -> ResearchPlan:
        """Generates a structured ResearchPlan from the research topic and depth setting."""
        user_prompt = f"Research Topic: {topic}\\nResearch Depth: {depth}"
        if user_guidance:
            user_prompt += f"\\nSpecial User Guidance/Focus Areas: {user_guidance}"

        messages = [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]

        try:
            # Try structured output first if supported
            if hasattr(self.llm, "with_structured_output"):
                try:
                    structured_llm = self.llm.with_structured_output(ResearchPlan)
                    res = structured_llm.invoke(messages)
                    if isinstance(res, ResearchPlan):
                        return res
                except Exception as e:
                    logger.debug(f"with_structured_output failed in PlannerAgent: {e}")

            # Fallback to standard invocation and JSON extraction
            response = self.llm.invoke(messages)
            text = response.content if hasattr(response, "content") else str(response)
            data = extract_json_payload(text)

            if data:
                return ResearchPlan.model_validate(data)

        except Exception as e:
            logger.error(f"PlannerAgent execution failed: {e}")

        # Robust default fallback
        return ResearchPlan(
            topic=topic,
            overview=f"Systematic analysis and evaluation of {topic}.",
            target_audience="Researchers and Technical Practitioners",
            subtopics=[
                SubTopic(
                    title="Foundational Principles & Architecture",
                    description=f"Core concepts, theoretical underpinnings, and key terminology of {topic}.",
                    search_queries=[f"{topic} architecture fundamentals", f"{topic} overview principles"],
                    key_questions=[f"What are the primary building blocks of {topic}?", "What core challenges does it address?"]
                ),
                SubTopic(
                    title="State of the Art & Methodologies",
                    description=f"Contemporary methodologies, frameworks, and leading approaches in {topic}.",
                    search_queries=[f"{topic} state of the art methods", f"{topic} recent advancements"],
                    key_questions=["What are the dominant modern techniques?", "How do different implementations compare?"]
                ),
                SubTopic(
                    title="Trade-offs, Challenges & Future Horizons",
                    description=f"Critical bottlenecks, limitations, and future outlook for {topic}.",
                    search_queries=[f"{topic} challenges limitations future trends", f"{topic} emerging research"],
                    key_questions=["What are the major open problems?", "Where is the field headed in the next 3-5 years?"]
                )
            ],
            expected_outcomes=[
                f"Comprehensive understanding of {topic}",
                "Actionable technical synthesis and comparative insights",
                "Strategic future projections"
            ]
        )
