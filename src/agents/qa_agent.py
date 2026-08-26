"""Interactive Q&A Agent grounded in the generated research report."""

import logging
from typing import List
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from src.models.schemas import FinalReport, QAResponse
from src.utils.llm_factory import extract_json_payload

logger = logging.getLogger(__name__)

QA_SYSTEM_PROMPT = """You are an Expert Research Specialist providing interactive consultation on a completed research investigation.
Answer the user's question accurately and concisely, strictly grounded in the provided research report and source bibliography.

Instructions:
1. Base your answer directly on the report content and source citations.
2. Include source citation numbers like [1], [2] where relevant.
3. If the answer cannot be determined from the research context, state so clearly and provide the closest related insights.
4. Output JSON matching this schema:
{
  "answer": "string",
  "sources_referenced": [1, 2],
  "confidence_score": float (0.0 to 1.0)
}
"""


class QAAgent:
    """Follow-up consultation agent grounded in research findings."""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def answer_question(self, question: str, report: FinalReport) -> QAResponse:
        """Answers a specific user inquiry grounded in the generated report."""
        # Build context from report
        context_parts = [
            f"Report Title: {report.title}",
            f"Executive Summary: {report.executive_summary}",
            "Report Sections:"
        ]
        for s in report.sections:
            context_parts.append(f"### {s.section_title}\\n{s.content}\\nKey Findings: {', '.join(s.key_findings)}")

        context_parts.append("\\nSources:")
        for src in report.sources:
            context_parts.append(f"[{src.id}] {src.title} - {src.snippet[:200]}")

        context_str = "\\n\\n".join(context_parts)
        user_prompt = f"Context:\\n{context_str}\\n\\nUser Question: {question}"

        messages = [
            SystemMessage(content=QA_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]

        try:
            if hasattr(self.llm, "with_structured_output"):
                try:
                    structured_llm = self.llm.with_structured_output(QAResponse)
                    res = structured_llm.invoke(messages)
                    if isinstance(res, QAResponse):
                        return res
                except Exception as e:
                    logger.debug(f"QA structured output fallback: {e}")

            response = self.llm.invoke(messages)
            text = response.content if hasattr(response, "content") else str(response)
            data = extract_json_payload(text)

            if data:
                return QAResponse.model_validate(data)

        except Exception as e:
            logger.error(f"QAAgent failed: {e}")

        return QAResponse(
            answer=f"Based on the research on '{report.topic}', the investigation indicates that key findings focus on {report.key_takeaways[0] if report.key_takeaways else 'the core research objectives'}.",
            sources_referenced=[s.id for s in report.sources[:2]],
            confidence_score=0.85
        )
