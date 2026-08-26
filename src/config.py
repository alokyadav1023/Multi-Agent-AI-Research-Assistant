"""Configuration module for Multi-Agent AI Research Assistant."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Default LLM configurations
DEFAULT_PROVIDER = os.getenv("DEFAULT_PROVIDER", "nvidia")
DEFAULT_NVIDIA_MODEL = os.getenv("DEFAULT_NVIDIA_MODEL", "nvidia/nemotron-3-super-120b-a12b")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

DEFAULT_GOOGLE_MODEL = os.getenv("DEFAULT_GOOGLE_MODEL", "gemini-2.5-flash")
DEFAULT_OPENAI_MODEL = os.getenv("DEFAULT_OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_GROQ_MODEL = os.getenv("DEFAULT_GROQ_MODEL", "llama-3.3-70b-versatile")
DEFAULT_ANTHROPIC_MODEL = os.getenv("DEFAULT_ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
DEFAULT_OLLAMA_MODEL = os.getenv("DEFAULT_OLLAMA_MODEL", "llama3")

# Search and retrieval limits
MAX_SEARCH_RESULTS_PER_QUERY = int(os.getenv("MAX_SEARCH_RESULTS_PER_QUERY", "5"))
MAX_TOTAL_SOURCES = int(os.getenv("MAX_TOTAL_SOURCES", "15"))
MAX_SCRAPED_CHARS = int(os.getenv("MAX_SCRAPED_CHARS", "4000"))

# Graph Workflow settings
DEFAULT_MAX_ITERATIONS = int(os.getenv("DEFAULT_MAX_ITERATIONS", "2"))
DEFAULT_RESEARCH_DEPTH = "standard"  # "brief", "standard", "deep"
