"""Report exporter utilities for Markdown, HTML, PDF, and JSON formats."""

import io
import json
import logging
from typing import Optional
from src.models.schemas import FinalReport

logger = logging.getLogger(__name__)


def export_to_markdown(report: FinalReport) -> str:
    """Converts a FinalReport object to a complete Markdown document."""
    lines = []
    lines.append(f"# {report.title}")
    lines.append(f"**Research Topic:** {report.topic}  ")
    lines.append(f"**Date Generated:** {report.generation_timestamp}  ")
    lines.append(f"**Word Count:** ~{report.total_words} words  ")
    if report.review_summary:
        lines.append(f"**Quality Verification:** {report.review_summary}  ")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Executive Summary
    lines.append("## Executive Summary")
    lines.append(report.executive_summary)
    lines.append("")

    # Key Takeaways
    if report.key_takeaways:
        lines.append("### Key Strategic Takeaways")
        for point in report.key_takeaways:
            lines.append(f"- {point}")
        lines.append("")

    # Table of Contents
    if report.table_of_contents:
        lines.append("## Table of Contents")
        for item in report.table_of_contents:
            slug = item.lower().replace(" ", "-").replace("&", "").replace("/", "")
            lines.append(f"- [{item}](#{slug})")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Body Sections
    for section in report.sections:
        lines.append(f"## {section.section_title}")
        lines.append(section.content)
        lines.append("")
        if section.key_findings:
            lines.append("**Core Findings:**")
            for f in section.key_findings:
                lines.append(f"- {f}")
            lines.append("")

    # Future Outlook
    if report.future_outlook:
        lines.append("## Future Outlook & Emerging Directions")
        lines.append(report.future_outlook)
        lines.append("")

    # Methodology
    if report.methodology_notes:
        lines.append("## Methodology & Synthesis Process")
        lines.append(report.methodology_notes)
        lines.append("")

    # References
    lines.append("## References & Sources")
    if report.sources:
        for src in report.sources:
            author_str = f" by {', '.join(src.authors)}" if src.authors else ""
            date_str = f" ({src.published_date})" if src.published_date else ""
            type_badge = f"[{src.source_type.upper()}]"
            lines.append(f"{src.id}. {type_badge} **[{src.title}]({src.url})**{author_str}{date_str}")
            if src.snippet:
                lines.append(f"   > *{src.snippet[:200]}...*")
    else:
        lines.append("*No external sources recorded.*\n")

    return "\\n".join(lines)


def export_to_html(report: FinalReport) -> str:
    """Renders the report as a styled HTML webpage."""
    import markdown
    md_content = export_to_markdown(report)
    html_body = markdown.markdown(md_content, extensions=["tables", "fenced_code"])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report.title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: #24292e;
            max-width: 900px;
            margin: 40px auto;
            padding: 0 20px;
            background-color: #ffffff;
        }}
        h1 {{
            color: #1a202c;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 8px;
        }}
        h2 {{
            color: #2b6cb0;
            margin-top: 32px;
            border-bottom: 1px solid #edf2f7;
            padding-bottom: 6px;
        }}
        h3 {{
            color: #4a5568;
        }}
        blockquote {{
            border-left: 4px solid #4299e1;
            margin: 1.5em 0;
            padding: 0.5em 1em;
            background-color: #ebf8ff;
            color: #2d3748;
            border-radius: 4px;
        }}
        code {{
            background-color: #f7fafc;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.9em;
            color: #c53030;
        }}
        hr {{
            border: 0;
            height: 1px;
            background: #e2e8f0;
            margin: 24px 0;
        }}
        ul {{
            padding-left: 24px;
        }}
        li {{
            margin-bottom: 6px;
        }}
        a {{
            color: #3182ce;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    {html_body}
</body>
</html>"""
    return html


def export_to_pdf(report: FinalReport) -> bytes:
    """Generates a PDF document from the report using ReportLab."""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'ReportTitle',
        parent=styles['Title'],
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#1a365d"),
        alignment=0,
        spaceAfter=12
    )

    h1_style = ParagraphStyle(
        'ReportH1',
        parent=styles['Heading1'],
        fontSize=15,
        leading=19,
        textColor=colors.HexColor("#2b6cb0"),
        spaceBefore=14,
        spaceAfter=6
    )

    h2_style = ParagraphStyle(
        'ReportH2',
        parent=styles['Heading2'],
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#2d3748"),
        spaceBefore=10,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        'ReportBody',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#2d3748"),
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        'ReportBullet',
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=3
    )

    meta_style = ParagraphStyle(
        'ReportMeta',
        parent=body_style,
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#718096")
    )

    story = []

    # Title & Metadata
    story.append(Paragraph(report.title, title_style))
    story.append(Paragraph(f"<b>Topic:</b> {report.topic} | <b>Date:</b> {report.generation_timestamp}", meta_style))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#cbd5e0"), spaceAfter=14))

    # Executive Summary
    story.append(Paragraph("Executive Summary", h1_style))
    clean_summary = report.executive_summary.replace("\\n", "<br/>")
    story.append(Paragraph(clean_summary, body_style))
    story.append(Spacer(1, 10))

    # Key Takeaways
    if report.key_takeaways:
        story.append(Paragraph("Key Takeaways", h2_style))
        for point in report.key_takeaways:
            story.append(Paragraph(f"&bull; {point}", bullet_style))
        story.append(Spacer(1, 10))

    # Sections
    for section in report.sections:
        story.append(Paragraph(section.section_title, h1_style))
        clean_content = section.content.replace("\\n", "<br/>")
        # Clean any raw markdown bolding for reportlab
        clean_content = clean_content.replace("**", "<b>").replace("**", "</b>")
        story.append(Paragraph(clean_content, body_style))

        if section.key_findings:
            story.append(Spacer(1, 4))
            for kf in section.key_findings:
                story.append(Paragraph(f"&bull; {kf}", bullet_style))
        story.append(Spacer(1, 10))

    # Future Outlook
    if report.future_outlook:
        story.append(Paragraph("Future Outlook", h1_style))
        story.append(Paragraph(report.future_outlook.replace("\\n", "<br/>"), body_style))
        story.append(Spacer(1, 10))

    # References
    if report.sources:
        story.append(Paragraph("References & Sources", h1_style))
        for src in report.sources:
            ref_line = f"[{src.id}] <b>{src.title}</b> ({src.source_type.upper()})<br/>&nbsp;&nbsp;&nbsp;&nbsp;{src.url}"
            story.append(Paragraph(ref_line, meta_style))
            story.append(Spacer(1, 4))

    doc.build(story)
    pdf_data = buffer.getvalue()
    buffer.close()
    return pdf_data


def export_to_json(report: FinalReport) -> str:
    """Serializes report into formatted JSON string."""
    return report.model_dump_json(indent=2)
