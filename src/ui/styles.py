"""Custom CSS styles for Multi-Agent AI Research Assistant Streamlit UI."""

import streamlit as st

CUSTOM_CSS = """
<style>
/* Main app layout */
.main-header {
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%);
    padding: 24px;
    border-radius: 12px;
    color: white;
    margin-bottom: 24px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
}

.main-header h1 {
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0;
    color: #ffffff !important;
}

.main-header p {
    font-size: 1.05rem;
    margin-top: 6px;
    opacity: 0.92;
    color: #f0fdf4 !important;
}

/* Agent Stage Card */
.agent-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #3b82f6;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 12px;
    transition: all 0.2s ease;
}

.agent-card:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    border-left-color: #2563eb;
}

.agent-title {
    font-weight: 700;
    font-size: 1rem;
    color: #1e293b;
    display: flex;
    align-items: center;
    gap: 8px;
}

.agent-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
}

.badge-planner { background-color: #e0e7ff; color: #3730a3; }
.badge-retriever { background-color: #dbeafe; color: #1e40af; }
.badge-writer { background-color: #fef3c7; color: #92400e; }
.badge-reviewer { background-color: #dcfce7; color: #166534; }
.badge-formatter { background-color: #fae8ff; color: #86198f; }

/* Source Card */
.source-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 10px;
}

.source-card a {
    color: #2563eb;
    text-decoration: none;
    font-weight: 600;
}

.source-card a:hover {
    text-decoration: underline;
}

.source-snippet {
    font-size: 0.88rem;
    color: #475569;
    margin-top: 4px;
    line-height: 1.4;
}

/* Metric Pill */
.metric-pill {
    background: #f1f5f9;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.85rem;
    font-weight: 600;
    color: #334155;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    margin-right: 8px;
}

/* Dark mode adjustments */
@media (prefers-color-scheme: dark) {
    .agent-card {
        background: #1e293b;
        border-color: #334155;
    }
    .agent-title {
        color: #f1f5f9;
    }
    .source-card {
        background: #1e293b;
        border-color: #334155;
    }
    .source-snippet {
        color: #94a3b8;
    }
    .metric-pill {
        background: #334155;
        color: #f1f5f9;
    }
}
</style>
"""


def apply_custom_styles():
    """Injects custom CSS into the Streamlit app."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
