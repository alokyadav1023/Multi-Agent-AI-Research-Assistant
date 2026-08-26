# 🧠 Multi-Agent AI Research Assistant

An autonomous, multi-agent AI research and information synthesis system built with **LangGraph**, **Streamlit**, **Python**, and high-speed **NVIDIA NIM** models (`nvidia/nemotron-3-nano-30b-a3b`, `nvidia/nemotron-3-super-120b-a12b`), along with multi-provider LLM support (**Google Gemini**, **OpenAI**, **Groq**, **Anthropic**, **Ollama**, and offline **Mock Mode**).

---

## 🌟 Overview

The **Multi-Agent AI Research Assistant** coordinates specialized autonomous agents to execute end-to-end research workflows:
1. **Task Decomposition & Strategic Planning**: Breaks down complex research topics into analytical pillars, hypotheses, and targeted search queries.
2. **Multi-Source Information Retrieval**: Concurrently gathers and cross-references evidence from **DuckDuckGo**, **ArXiv** scientific preprints, **Wikipedia**, and optional **Tavily AI Search**, scraping full text from target web pages.
3. **Structured Technical Synthesis**: Drafts rigorous, in-depth sections in parallel with inline numeric citations [1], [2] linked to the verified source catalog.
4. **Fact-Checking & Reflection Loops**: Audits the draft for factual fidelity, grounding, and completeness, executing targeted refinement loops if gaps exist.
5. **Executive Formatting & Export**: Packages publication-ready reports with Executive Summaries, Table of Contents, BibTeX/References, and one-click export to **PDF**, **Markdown**, **HTML**, and **JSON**.
6. **Interactive Consultation (Grounded Q&A)**: Enables follow-up questions answered strictly from the research context.

---

## 🏗️ Architecture & Multi-Agent Graph

```mermaid
flowchart TD
    User([User Research Topic & Depth]) --> UI[Streamlit Interactive Frontend]
    UI --> Orchestrator[LangGraph State Orchestrator]

    subgraph MultiAgentSquad ["LangGraph Multi-Agent Workflow (NVIDIA NIM Accelerated)"]
        Orchestrator --> Planner["1. Lead Planner Agent<br/>(Task Decomposition & Subtopics)"]
        Planner --> Retriever["2. Concurrent Retriever Agent<br/>(DuckDuckGo, ArXiv, Wikipedia, Tavily)"]
        Retriever --> Scraper["Web Content Extractor"]
        Scraper --> Writer["3. Parallel Writer Agent<br/>(Multi-Threaded Synthesis & Citations)"]
        Writer --> Reviewer["4. Fact-Checker & Reviewer Agent<br/>(Factual Grounding & Quality Audit)"]
        
        Reviewer -->|Needs Refinement & Iterations < Max| Retriever
        Reviewer -->|Approved| Formatter["5. Executive Formatter Agent<br/>(TOC, Metrics, Key Takeaways)"]
    end

    Formatter --> Export[Export Hub: PDF / Markdown / HTML / JSON]
    Formatter --> InteractiveQA[Grounded Research Q&A Consultant]
    Export --> UI
    InteractiveQA --> UI
```

---

## 🚀 Key Features

- **Autonomous Agent Squad**:
  - `PlannerAgent`: Formulates structured decomposition using Pydantic models.
  - `RetrieverAgent`: Multi-channel search with deduplication and relevance scoring.
  - `WriterAgent`: Section-by-section concurrent synthesis with strict citation anchors.
  - `ReviewerAgent`: Factual consistency check and reflection loop routing.
  - `QAAgent`: Grounded interactive follow-up agent.
- **Ultra-Fast NVIDIA NIM Engine**:
  - `⚡ nvidia/nemotron-3-nano-30b-a3b`: Sub-second latency (~0.47s).
  - `🧠 nvidia/nemotron-3-super-120b-a12b`: 120B parameter deep reasoning flagship.
- **Multi-Source Evidence Gathering**: DuckDuckGo (no key required), ArXiv (academic preprints), Wikipedia (encyclopedic foundation), and Tavily (deep search).
- **Multi-Provider LLM Engine**: Seamless switching between NVIDIA NIM, Google Gemini, OpenAI, Groq (Llama 3.3 70B), Anthropic Claude, Ollama local models, and offline Mock Mode.
- **Interactive Streamlit Interface**:
  - Real-time agent activity timeline and thought trace.
  - Interactive tabs for Final Report, Research Plan, Source Inspector, Quality Audit, and Follow-up Q&A.
  - One-click document export to **PDF** (via ReportLab), **Markdown**, **HTML**, and **JSON**.

---

## 💻 Commands to Run on Localhost

### Method 1: One-Line Command (Fastest)
In your terminal inside the project directory:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### Method 2: Standard Terminal Flow
```powershell
# 1. Activate the virtual environment
.\.venv\Scripts\activate

# 2. Start the application
streamlit run app.py
```

### Method 3: 1-Click Batch Launcher (Windows)
Double-click `run_app.bat` or run:
```powershell
.\run_app.bat
```

Once started, open your browser and go to:
👉 **`http://localhost:8501`**

---

## 📦 Setting Up Fresh from Scratch (New Machine)

```bash
# 1. Clone the repo
git clone https://github.com/alokyadav1023/Multi-Agent-AI-Research-Assistant.git
cd Multi-Agent-AI-Research-Assistant

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
# Windows:
.\.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Configure API Key (Optional)
cp .env.example .env

# 6. Run the app
streamlit run app.py
```

---

## 🧪 Running Automated Tests

```bash
pytest -v
```

---

## 📂 Project Structure

```
Multi-Agent AI Research Assistant/
├── app.py                     # Streamlit application entry point
├── requirements.txt           # Project dependencies
├── run_app.bat                # Windows launcher batch script
├── .env.example               # Environment variables template
├── README.md                  # Documentation
├── src/
│   ├── config.py              # Configuration settings
│   ├── agents/
│   │   ├── planner.py         # Lead Planner Agent
│   │   ├── retriever.py       # Information Retrieval Agent
│   │   ├── writer.py          # Parallel Synthesis & Writer Agent
│   │   ├── reviewer.py        # Quality Reviewer & Fact-Checker
│   │   └── qa_agent.py        # Grounded Q&A Consultation Agent
│   ├── graph/
│   │   ├── state.py           # LangGraph TypedDict state schema
│   │   └── workflow.py        # StateGraph builder and conditional routing
│   ├── models/
│   │   └── schemas.py         # Pydantic structured output models
│   ├── tools/
│   │   ├── search_tools.py    # DuckDuckGo, ArXiv, Wikipedia, Tavily
│   │   └── web_scraper.py     # HTML text extraction utility
│   ├── utils/
│   │   ├── llm_factory.py     # Multi-provider LLM initializers & Mock LLM
│   │   └── exporter.py        # PDF, Markdown, HTML, and JSON generators
│   └── ui/
│       ├── styles.py          # Custom CSS and theming
│       └── components.py      # Modular Streamlit UI components
└── tests/
    ├── test_models.py         # Pydantic schema validation tests
    ├── test_tools.py          # Search and scraping tests
    ├── test_agents.py         # Agent node tests
    ├── test_nvidia.py         # NVIDIA NIM provider tests
    └── test_workflow.py       # LangGraph end-to-end execution tests
```
