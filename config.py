"""
config.py — Centralised configuration for virtual_delphi pipeline.
Set OPENROUTER_API_KEY in a .env file in the repo root.
Optional: DATA_BASE_DIR, RESULTS_DIR, OUT_DIR
"""
import os
from pathlib import Path
from dotenv import load_dotenv

_repo_root = Path(__file__).parent
load_dotenv(_repo_root / ".env")
load_dotenv(Path.home() / "Desktop/openalex/.env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"

_default_base = Path.home() / "Desktop/openalex"
DATA_BASE_DIR  = Path(os.getenv("DATA_BASE_DIR", str(_default_base)))

FULLTEXT_BASE       = DATA_BASE_DIR / "cluster_fulltexts"
NEWS_BASE           = DATA_BASE_DIR / "cluster_news"
RESULTS_BASE        = Path(os.getenv("RESULTS_DIR", str(DATA_BASE_DIR / "ce_delphi_v3_results")))
OUT_DIR             = Path(os.getenv("OUT_DIR", str(DATA_BASE_DIR / "ce_retrieval_results")))
ABSTRACTS_PATH      = DATA_BASE_DIR / "ce_abstracts.json"
BRIDGE_TERMS_PATH   = DATA_BASE_DIR / "bridge_terms.json"
BRIDGE_AUTHORS_PATH = DATA_BASE_DIR / "bridge_authors.json"
WEAK_SIGNALS_PATH   = DATA_BASE_DIR / "weak_signals.json"
FULL_CORPUS_DIR     = DATA_BASE_DIR / "results_circular_economy/full_corpus"
PAPER_TOPICS_PATH   = FULL_CORPUS_DIR / "paper_topics_clean.csv"
TOPICS_PATH         = FULL_CORPUS_DIR / "semantic_topics.csv"
SIJ_PATH            = FULL_CORPUS_DIR / "circular_economy_aws_v2_pairwise.csv"

def check_api_key():
    if not OPENROUTER_API_KEY:
        raise EnvironmentError("OPENROUTER_API_KEY not set. Add to .env: OPENROUTER_API_KEY=your_key")
