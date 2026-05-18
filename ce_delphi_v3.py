#!/usr/bin/env python3
"""
CE-Delphi-RAG Pipeline v3 — Three-Condition Ablation Design
============================================================

Three experimental conditions, each running 3 Delphi rounds independently.
The unit of analysis is the cluster (n=11), not macroareas.

CONDITIONS:
  A  No-RAG            Parametric knowledge only, no external context
  B  RAG-Lit           Scientific literature + specialist role
  C  RAG-Full          Literature + news/grey-lit + specialist role
                       + weak signals / wild cards prompt stimulus

DELPHI OUTPUT (per condition, per round):
  - Per-cluster scores: P10/P50/P90 for "research growth potential"
  - Qualitative signals: list of emerging terms/technologies per cluster
  - Wild cards (Condition C only): low-probability / high-impact events

VALIDATION:
  - Convergence: CV, IQR, interval width across rounds (methodological)
  - Signal recall: overlap between panel-identified terms and terms that
    actually emerged in paper_topics_clean.csv post-2020 (empirical)
  - Spearman rho on cluster ranking (n=11, critical rho=0.618 for p<0.05)

USAGE:
  conda activate aws
  python ce_delphi_v3.py --condition A
  python ce_delphi_v3.py --condition B
  python ce_delphi_v3.py --condition C   # requires news corpus
  python ce_delphi_v3.py --condition B --analysis-only
  python ce_delphi_v3.py --build-news    # build news corpus (run once)

REFERENCES:
  - AI-Delphi: Nobrega et al. (2025), AI 6(6):294
  - LLM forecasting failure modes: Karkar & Chopra (2025)
  - Percentile elicitation: Lichtendahl & Winkler (2007)
  - Weak signals in foresight: Ansoff (1975), Hiltunen (2008)
  - Wild cards: Petersen (1999), Glenn & Gordon (2009)
"""

import json
import os
import random
import time
import csv
import math
import argparse
import threading
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import requests

load_dotenv(Path.home() / "Desktop/openalex/.env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"


# ---------------------------------------------------------------------------
# CLUSTER DEFINITIONS — 11 clean clusters from paper_topics_clean.csv
# ---------------------------------------------------------------------------

CLUSTERS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14]

CLUSTER_LABELS = {
    2:  "Energy, LCA & Decarbonization in Circular Economy",
    3:  "CE Governance, Urban Metabolism & Regional Development",
    4:  "CE Business Models, Green Finance & Innovation",
    5:  "CE Policy, Sustainability Transitions & Political Economy",
    6:  "CE Frameworks, Digital Passports & Conceptual Foundations",
    7:  "Fashion, Textiles & Consumer Circularity",
    8:  "Industry 4.0, Supply Chain & Circular Manufacturing",
    9:  "Construction Waste, Built Environment & Urban Planning",
    10: "Advanced Plastics, Biocomposites & Circular Materials",
    11: "Recycled Construction Materials & Geotechnical Applications",
    14: "Industrial Waste, Medical Devices & Material Recovery",
}

# Ground truth: observed growth 2016-2020 -> 2021-2024
# Source: paper_topics_clean.csv
OBSERVED_GROWTH = {
    2:  262.8,
    3:  209.6,
    4:  365.4,
    5:  308.0,
    6:  190.4,
    7:  333.5,
    8:  218.2,
    9:  326.8,
    10: 520.0,
    11: 556.6,
    14: 281.5,
}

# Thematic groupings (for RAG specialization — not used as prediction units)
CLUSTER_GROUPS = {
    "energy_materials":  [2, 14],
    "policy_governance": [3, 5, 6],
    "business_industry": [4, 7, 8],
    "built_materials":   [9, 10, 11],
}

DISCIPLINES = ["environmental_science", "economics", "engineering", "policy", "sociology"]
REGIONS     = ["Europe", "Asia", "Americas", "Africa_ME"]
SENIORITY   = ["mid", "senior"]
APPROACHES  = ["quantitative", "qualitative", "mixed"]

MODELS = [
    "meta-llama/llama-3.1-70b-instruct",
    "google/gemma-3-27b-it",
    "mistralai/mistral-large-2411",
    "anthropic/claude-haiku-4-5",
    "qwen/qwen3-30b-a3b",
    "microsoft/phi-4",
]

N_EXPERTS_PER_MODEL = 10
N_ROUNDS            = 3
ABSTRACTS_SPEC      = 5
ABSTRACTS_CROSS     = 1
NEWS_PER_CLUSTER    = 1
MAX_WORKERS         = 3
TEMPORAL_CUTOFF     = 2020

FULLTEXT_BASE = Path.home() / "Desktop/openalex/cluster_fulltexts"
NEWS_BASE     = Path.home() / "Desktop/openalex/cluster_news"

_progress_lock = threading.Lock()


# ---------------------------------------------------------------------------
# OUTPUT DIRECTORY
# ---------------------------------------------------------------------------

def get_output_dir(condition: str) -> Path:
    base = Path.home() / "Desktop/openalex/ce_delphi_v3_results" / f"condition_{condition}"
    base.mkdir(parents=True, exist_ok=True)
    return base


# ---------------------------------------------------------------------------
# MODULE 1 — CORPUS LOADERS
# ---------------------------------------------------------------------------

def load_scientific_corpus() -> dict:
    """Load scientific papers per cluster, filtered to <= TEMPORAL_CUTOFF."""
    corpus = defaultdict(list)
    counts = defaultdict(int)

    for cluster_dir in FULLTEXT_BASE.glob("cluster_C*"):
        try:
            cluster_id = int(cluster_dir.name.replace("cluster_C", ""))
        except ValueError:
            continue
        if cluster_id not in CLUSTERS:
            continue

        for meta_f in cluster_dir.glob("*.meta.json"):
            txt_f = meta_f.with_suffix("").with_suffix(".txt")
            if not txt_f.exists():
                continue
            try:
                meta = json.loads(meta_f.read_text())
                year = meta.get("year")
                if year and str(year).lower() != "nan":
                    if int(float(year)) > TEMPORAL_CUTOFF:
                        continue
                text = txt_f.read_text(encoding="utf-8", errors="ignore").strip()
                if len(text) < 100:
                    continue
                corpus[cluster_id].append({
                    "doi":         meta.get("doi", ""),
                    "title":       meta.get("title", ""),
                    "year":        year,
                    "cluster":     cluster_id,
                    "source_type": "scientific",
                    "is_fulltext": meta.get("is_fulltext", False),
                    "text":        text[:3000],
                })
                counts[cluster_id] += 1
            except Exception:
                continue

    print("Scientific corpus:")
    for c in CLUSTERS:
        print(f"  C{c} ({CLUSTER_LABELS[c][:40]}): {counts[c]} docs")
    missing = [c for c in CLUSTERS if counts[c] == 0]
    if missing:
        print(f"  WARNING: clusters with 0 docs: {missing}")
    return corpus


def load_news_corpus() -> dict:
    """
    Load news / grey literature per cluster (Condition C only).

    Expected format: NEWS_BASE/cluster_C{N}/*.meta.json + *.txt
    Each document: title, year (2016-2020), source, short text.
    Build with --build-news flag.
    """
    corpus = defaultdict(list)
    if not NEWS_BASE.exists():
        print(f"  WARNING: news corpus not found at {NEWS_BASE}")
        print(f"  Run with --build-news to create it.")
        return corpus

    for cluster_dir in NEWS_BASE.glob("cluster_C*"):
        try:
            cluster_id = int(cluster_dir.name.replace("cluster_C", ""))
        except ValueError:
            continue
        if cluster_id not in CLUSTERS:
            continue

        for meta_f in cluster_dir.glob("*.meta.json"):
            txt_f = meta_f.with_suffix("").with_suffix(".txt")
            if not txt_f.exists():
                continue
            try:
                meta = json.loads(meta_f.read_text())
                year = meta.get("year")
                if year and str(year).lower() != "nan":
                    if int(float(year)) > TEMPORAL_CUTOFF:
                        continue
                text = txt_f.read_text(encoding="utf-8", errors="ignore").strip()
                if len(text) < 50:
                    continue
                corpus[cluster_id].append({
                    "title":       meta.get("title", ""),
                    "year":        year,
                    "source":      meta.get("source", ""),
                    "cluster":     cluster_id,
                    "source_type": "news",
                    "text":        text[:2000],
                })
            except Exception:
                continue

    total = sum(len(v) for v in corpus.values())
    print(f"News corpus: {total} documents across {len(corpus)} clusters")
    return corpus


# ---------------------------------------------------------------------------
# MODULE 2 — NEWS CORPUS BUILDER (--build-news)
# ---------------------------------------------------------------------------

def build_news_corpus():
    """
    Build news corpus using LLM to generate realistic news summaries
    for each cluster, 2016-2020, simulating grey literature / media coverage.

    This is clearly documented as LLM-generated synthetic news for
    methodological purposes (not real scraped news).
    Each cluster gets NEWS_PER_CLUSTER documents per year window.

    For a real paper, replace with actual scraped news (GDELT, NewsAPI).
    """
    print("Building synthetic news corpus (LLM-generated)...")
    print("NOTE: For publication, replace with real scraped news (GDELT/NewsAPI).")

    NEWS_BASE.mkdir(parents=True, exist_ok=True)

    for cluster_id in CLUSTERS:
        cluster_dir = NEWS_BASE / f"cluster_C{cluster_id}"
        cluster_dir.mkdir(exist_ok=True)
        label = CLUSTER_LABELS[cluster_id]

        for doc_idx in range(NEWS_PER_CLUSTER):
            meta_f = cluster_dir / f"news_{doc_idx:03d}.meta.json"
            txt_f  = cluster_dir / f"news_{doc_idx:03d}.txt"
            if meta_f.exists() and txt_f.exists():
                print(f"  C{cluster_id} doc {doc_idx}: already exists, skipping")
                continue

            year = 2016 + (doc_idx % 5)  # spread across 2016-2020
            prompt = (
                f"Write a short news article (150-200 words) from {year} about "
                f"emerging developments in the field of '{label}' within the circular "
                f"economy. Focus on: (1) a specific technology, policy, or practice "
                f"that was newly emerging at that time, (2) a concrete real-world "
                f"application or pilot project, (3) any regulatory or market signal. "
                f"Write in a factual, journalistic style. Do NOT mention events after "
                f"{TEMPORAL_CUTOFF}. Output ONLY the article text, no title."
            )

            messages = [
                {"role": "system", "content": "You are a science journalist specializing in sustainability and circular economy. Write factual, grounded news articles."},
                {"role": "user", "content": prompt},
            ]

            # Use a single model for news generation
            model  = "mistralai/mistral-large-2411"
            text   = _call_llm_raw(model, messages, temperature=0.6, max_tokens=400)

            if not text:
                print(f"  C{cluster_id} doc {doc_idx}: generation failed")
                continue

            meta = {
                "title":   f"Emerging developments in {label} ({year})",
                "year":    year,
                "source":  "synthetic_news_LLM",
                "cluster": cluster_id,
            }
            meta_f.write_text(json.dumps(meta, indent=2))
            txt_f.write_text(text)
            print(f"  C{cluster_id} doc {doc_idx} ({year}): generated ({len(text)} chars)")
            time.sleep(0.5)

    print(f"\nNews corpus built at {NEWS_BASE}")
    print("Review generated documents before running Condition C.")


def _call_llm_raw(model, messages, temperature=0.7, max_tokens=600):
    """Raw LLM call without parsing."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":       model,
        "messages":    messages,
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }
    for attempt in range(3):
        try:
            r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=90)
            if r.status_code == 429:
                time.sleep(2 ** (attempt + 2))
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt == 2:
                print(f"    [FAIL] {model}: {e}")
            else:
                time.sleep(3)
    return None


# ---------------------------------------------------------------------------
# MODULE 3 — EXPERT POPULATION
# ---------------------------------------------------------------------------

def generate_expert_profile(model_idx: int, expert_idx: int) -> dict:
    """Deterministic expert profile. Seed is fixed at pipeline level."""
    discipline = DISCIPLINES[(model_idx + expert_idx) % len(DISCIPLINES)]
    region     = REGIONS[(model_idx * 3 + expert_idx) % len(REGIONS)]
    seniority  = SENIORITY[expert_idx % len(SENIORITY)]
    approach   = APPROACHES[(model_idx + expert_idx * 2) % len(APPROACHES)]
    years_exp  = 8 + (expert_idx % 20) if seniority == "mid" else 18 + (expert_idx % 15)

    # Assign cluster specialization (not MA — direct cluster)
    group_keys = list(CLUSTER_GROUPS.keys())
    group_key  = group_keys[(model_idx + expert_idx) % len(group_keys)]
    spec_clusters = CLUSTER_GROUPS[group_key]

    return {
        "expert_id":         f"M{model_idx:02d}_E{expert_idx:02d}",
        "model":             MODELS[model_idx],
        "discipline":        discipline,
        "region":            region,
        "seniority":         seniority,
        "approach":          approach,
        "spec_group":        group_key,
        "spec_clusters":     spec_clusters,
        "years_experience":  years_exp,
    }


def build_expert_rag(expert: dict, sci_corpus: dict, news_corpus: dict,
                     condition: str, rng: random.Random) -> dict:
    """
    Build personalized RAG context for one expert given condition.

    Returns dict with keys 'scientific' and 'news' (list of docs each).
    """
    spec_clusters = expert["spec_clusters"]
    sci_docs = []
    news_docs = []

    # Scientific RAG by condition
    for c in CLUSTERS:
        pool = sci_corpus.get(c, [])
        if not pool:
            continue
        if condition == "C":
            if c in spec_clusters:
                sci_docs.extend(rng.choices(pool, k=ABSTRACTS_SPEC))
        elif condition in ("E", "F", "G", "H"):
            if c in spec_clusters:
                sci_docs.extend(rng.choices(pool, k=1))  # 1 abstract per spec cluster
        else:
            k = ABSTRACTS_SPEC if c in spec_clusters else ABSTRACTS_CROSS
            sci_docs.extend(rng.choices(pool, k=k))

    # News for C and E
    if condition in ("C", "E"):
        for c in CLUSTERS:
            pool = news_corpus.get(c, [])
            if pool:
                # E: 1 news doc per spec cluster only; C: NEWS_PER_CLUSTER for all
                if condition == "E":
                    if c in spec_clusters:
                        news_docs.append(rng.choice(pool))
                else:
                    news_docs.extend(rng.choices(pool, k=min(NEWS_PER_CLUSTER, len(pool))))

    return {"scientific": sci_docs, "news": news_docs}


def format_scientific_context(docs: list, max_chars: int = 1500) -> str:
    if not docs:
        return "(No scientific publications available.)"
    lines = []
    label = "[ABSTRACT]" if max_chars <= 300 else "[FULL TEXT]"
    for i, doc in enumerate(docs, 1):
        c = doc.get("cluster")
        lines.append(f"--- Publication {i} {label} ---")
        lines.append(f"Title: {doc['title']}")
        if doc.get("year") and str(doc["year"]).lower() != "nan":
            lines.append(f"Year: {int(float(doc['year']))}")
        if c:
            lines.append(f"Research area: {CLUSTER_LABELS.get(c, '')}")
        lines.append(doc["text"][:max_chars])
        lines.append("")
    return "\n".join(lines)


def format_news_context(docs: list, max_chars: int = 800) -> str:
    if not docs:
        return "(No news documents available.)"
    lines = []
    for i, doc in enumerate(docs, 1):
        lines.append(f"--- News/Policy Document {i} ---")
        lines.append(f"Title: {doc['title']}")
        if doc.get("year"):
            lines.append(f"Year: {doc['year']}")
        if doc.get("source"):
            lines.append(f"Source: {doc['source']}")
        lines.append(doc["text"][:max_chars])
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# MODULE 4 — PROMPTS (three conditions)
# ---------------------------------------------------------------------------

CLUSTER_LIST_STR = "\n".join(
    f"  C{c}: {CLUSTER_LABELS[c]}" for c in CLUSTERS
)


def _specialist_role(expert: dict) -> str:
    spec_labels = [CLUSTER_LABELS[c] for c in expert["spec_clusters"]]
    return (
        f"You are a senior circular economy researcher with {expert['years_experience']} "
        f"years of experience in {expert['discipline'].replace('_', ' ')} "
        f"({expert['region']}). "
        f"Your primary specialization covers: {', '.join(spec_labels)}. "
        f"Your methodological approach is {expert['approach']}."
    )


def build_prompt_condition_A(expert: dict, round_num: int,
                              facilitator_summary: str | None) -> list:
    """
    Condition A — No RAG.
    Expert reasons purely from parametric knowledge.
    Task: assess growth potential and identify emerging terms per cluster.
    """
    system = (
        "You are a circular economy researcher participating in a structured "
        "Delphi study. Use your general knowledge of circular economy research "
        "up to 2020 to answer. Respond ONLY with valid JSON — no extra text."
    )

    feedback_block = ""
    if facilitator_summary and round_num > 1:
        feedback_block = f"""
The anonymized panel aggregate from the previous round:
{facilitator_summary}

Review the panel feedback. You may revise your assessments if the collective
evidence is stronger than your individual knowledge, but justify any changes.
"""

    user = f"""You are a circular economy researcher.

TASK (Round {round_num} of {N_ROUNDS}):

Below are 11 active research clusters in circular economy. For each cluster:
1. Assess its growth potential for the period 2021-2025 (score 0-1, where
   1 = highest growth potential relative to the other clusters).
2. List 3-5 specific terms, technologies, or sub-topics you expect to EMERGE
   or GROW SIGNIFICANTLY within that cluster after 2020.

Research clusters:
{CLUSTER_LIST_STR}

{feedback_block}

Respond ONLY with this JSON:
{{
  "clusters": {{
    "C2":  {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>,
             "emerging_terms": ["term1", "term2", "term3"]}},
    "C3":  {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>,
             "emerging_terms": ["term1", "term2", "term3"]}},
    "C4":  {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>,
             "emerging_terms": ["term1", "term2", "term3"]}},
    "C5":  {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>,
             "emerging_terms": ["term1", "term2", "term3"]}},
    "C6":  {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>,
             "emerging_terms": ["term1", "term2", "term3"]}},
    "C7":  {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>,
             "emerging_terms": ["term1", "term2", "term3"]}},
    "C8":  {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>,
             "emerging_terms": ["term1", "term2", "term3"]}},
    "C9":  {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>,
             "emerging_terms": ["term1", "term2", "term3"]}},
    "C10": {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>,
             "emerging_terms": ["term1", "term2", "term3"]}},
    "C11": {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>,
             "emerging_terms": ["term1", "term2", "term3"]}},
    "C14": {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>,
             "emerging_terms": ["term1", "term2", "term3"]}}
  }},
  "rationale": "<2-3 sentences on your overall reasoning>"
}}

Constraints:
- growth_p10 <= growth_p50 <= growth_p90, all in [0, 1]
- growth_p50 values should reflect relative differences between clusters
"""
    return [{"role": "system", "content": system},
            {"role": "user",   "content": user}]


def build_prompt_condition_B(expert: dict, rag_docs: dict, round_num: int,
                              facilitator_summary: str | None,
                              abstract_only: bool = False) -> list:
    """
    Condition B — RAG literature + specialist role.
    Expert reasons from scientific publications and declared specialization.
    """
    system = (
        f"{_specialist_role(expert)} "
        "You are participating in a structured Delphi study. "
        "Reason exclusively from the publication record provided. "
        "Respond ONLY with valid JSON — no extra text."
    )

    _max_chars  = 300 if abstract_only else 1500
    sci_context = format_scientific_context(rag_docs["scientific"], max_chars=_max_chars)

    feedback_block = ""
    if facilitator_summary and round_num > 1:
        feedback_block = f"""
Anonymized panel aggregate from previous round:
{facilitator_summary}

Review the panel feedback and revise where the collective evidence is stronger
than your individual record. Justify any changes.
"""

    user = f"""Your publication record (all papers published up to {TEMPORAL_CUTOFF}):

{sci_context}
---
TASK (Round {round_num} of {N_ROUNDS}):

Based on the publication record above and your expertise, assess each of the
11 circular economy research clusters:

1. Growth potential score (P10/P50/P90) for the 2021-2025 period.
2. Emerging terms: 3-5 specific terms, technologies, or sub-topics you expect
   to emerge or accelerate within that cluster after 2020, based on signals
   visible in the literature up to {TEMPORAL_CUTOFF}.

Signals to look for in the literature:
- Topics described as "nascent", "emerging", or "underexplored"
- New methods appearing in recent papers (2018-2020)
- Cross-disciplinary uptake of concepts from other fields
- Calls for more research / identified gaps

Research clusters:
{CLUSTER_LIST_STR}

{feedback_block}

Respond ONLY with this JSON:
{{
  "clusters": {{
    "C2":  {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>,
             "emerging_terms": ["term1", "term2", "term3"]}},
    "C3":  {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>,
             "emerging_terms": ["term1", "term2", "term3"]}},
    "C4":  {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>,
             "emerging_terms": ["term1", "term2", "term3"]}},
    "C5":  {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>,
             "emerging_terms": ["term1", "term2", "term3"]}},
    "C6":  {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>,
             "emerging_terms": ["term1", "term2", "term3"]}},
    "C7":  {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>,
             "emerging_terms": ["term1", "term2", "term3"]}},
    "C8":  {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>,
             "emerging_terms": ["term1", "term2", "term3"]}},
    "C9":  {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>,
             "emerging_terms": ["term1", "term2", "term3"]}},
    "C10": {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>,
             "emerging_terms": ["term1", "term2", "term3"]}},
    "C11": {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>,
             "emerging_terms": ["term1", "term2", "term3"]}},
    "C14": {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>,
             "emerging_terms": ["term1", "term2", "term3"]}}
  }},
  "rationale": "<2-3 sentences citing specific signals from the publications>"
}}
"""
    return [{"role": "system", "content": system},
            {"role": "user",   "content": user}]


def build_prompt_condition_C(expert: dict, rag_docs: dict, round_num: int,
                              facilitator_summary: str | None) -> list:
    """Condition C: 5 spec abstracts + news/policy. Ultra-compact for small context models."""
    system = (
        "JSON ONLY. Start your response with { and end with }. "
        "No thinking tags, no prose, no explanation. "
        "If you write anything before { the experiment fails."
    )

    sci_context  = format_scientific_context(rag_docs["scientific"], max_chars=300)
    news_context = format_news_context(rag_docs["news"], max_chars=400)

    feedback_block = ""
    if facilitator_summary and round_num > 1:
        feedback_block = f"Panel round {round_num-1} summary:\n{facilitator_summary}\n---\n"

    user = f"""{feedback_block}Scientific abstracts (pre-2020):
{sci_context}
News/policy (pre-2020):
{news_context}
Score 2021-2025 growth for each CE cluster. JSON only:
{{
  "clusters": {{
    "C2":  {{"p10":<f>,"p50":<f>,"p90":<f>,"terms":["t1","t2","t3"]}},
    "C3":  {{"p10":<f>,"p50":<f>,"p90":<f>,"terms":["t1","t2","t3"]}},
    "C4":  {{"p10":<f>,"p50":<f>,"p90":<f>,"terms":["t1","t2","t3"]}},
    "C5":  {{"p10":<f>,"p50":<f>,"p90":<f>,"terms":["t1","t2","t3"]}},
    "C6":  {{"p10":<f>,"p50":<f>,"p90":<f>,"terms":["t1","t2","t3"]}},
    "C7":  {{"p10":<f>,"p50":<f>,"p90":<f>,"terms":["t1","t2","t3"]}},
    "C8":  {{"p10":<f>,"p50":<f>,"p90":<f>,"terms":["t1","t2","t3"]}},
    "C9":  {{"p10":<f>,"p50":<f>,"p90":<f>,"terms":["t1","t2","t3"]}},
    "C10": {{"p10":<f>,"p50":<f>,"p90":<f>,"terms":["t1","t2","t3"]}},
    "C11": {{"p10":<f>,"p50":<f>,"p90":<f>,"terms":["t1","t2","t3"]}},
    "C14": {{"p10":<f>,"p50":<f>,"p90":<f>,"terms":["t1","t2","t3"]}}
  }}
}}
"""
    return [{"role": "system", "content": system},
            {"role": "user",   "content": user}]



def build_prompt_condition_E(expert: dict, rag_docs: dict, round_num: int,
                              facilitator_summary: str | None) -> list:
    """Condition E: 1 spec abstract + 1 spec news. Minimal RAG for context efficiency."""
    system = (
        f"{_specialist_role(expert)} "
        "You are participating in a structured Delphi study. "
        "Respond ONLY with valid JSON — no extra text."
    )

    sci_context  = format_scientific_context(rag_docs["scientific"], max_chars=300)
    news_context = format_news_context(rag_docs["news"], max_chars=400)

    feedback_block = ""
    if facilitator_summary and round_num > 1:
        feedback_block = f"""
Anonymized panel aggregate from previous round:
{facilitator_summary}
Review the panel signals. Revise where warranted.
"""

    user = f"""Your publication record (pre-{TEMPORAL_CUTOFF}):

{sci_context}

---
Policy signal (pre-{TEMPORAL_CUTOFF}):

{news_context}

---
TASK (Round {round_num} of {N_ROUNDS}):

Assess each of the 11 CE research clusters for 2021-2025 growth potential.
1. Growth potential score (P10/P50/P90) in [0,1].
2. Emerging terms: 3 specific terms expected to emerge post-2020.

Clusters:
{CLUSTER_LIST_STR}

{feedback_block}
Respond ONLY with this JSON:
{{
  "clusters": {{
    "C2":  {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>, "emerging_terms": ["t1","t2","t3"]}},
    "C3":  {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>, "emerging_terms": ["t1","t2","t3"]}},
    "C4":  {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>, "emerging_terms": ["t1","t2","t3"]}},
    "C5":  {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>, "emerging_terms": ["t1","t2","t3"]}},
    "C6":  {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>, "emerging_terms": ["t1","t2","t3"]}},
    "C7":  {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>, "emerging_terms": ["t1","t2","t3"]}},
    "C8":  {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>, "emerging_terms": ["t1","t2","t3"]}},
    "C9":  {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>, "emerging_terms": ["t1","t2","t3"]}},
    "C10": {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>, "emerging_terms": ["t1","t2","t3"]}},
    "C11": {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>, "emerging_terms": ["t1","t2","t3"]}},
    "C14": {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>, "emerging_terms": ["t1","t2","t3"]}}
  }},
  "rationale": "<1-2 sentences>"
}}
"""
    return [{"role": "system", "content": system},
            {"role": "user",   "content": user}]



def build_prompt_condition_F(expert: dict, rag_docs: dict, round_num: int,
                              facilitator_summary: str | None,
                              weak_signals: dict) -> list:
    """Condition F: 1 spec abstract + pre-extracted weak signals + drivers per cluster."""
    system = (
        f"{_specialist_role(expert)} "
        "You are participating in a structured Delphi study. "
        "Respond ONLY with valid JSON — no extra text."
    )

    sci_context = format_scientific_context(rag_docs["scientific"], max_chars=300)

    # Build weak signals context from pre-extracted JSON
    ws_lines = ["Foresight signals extracted from policy documents (pre-2020):"]
    for ckey in [f"C{c}" for c in CLUSTERS]:
        ws = weak_signals.get(ckey, {})
        if ws.get("weak_signal") and ws["weak_signal"] != "Extraction failed.":
            label = CLUSTER_LABELS.get(int(ckey[1:]), ckey)
            ws_lines.append(f"\n{ckey} ({label[:40]}):")
            ws_lines.append(f"  Weak signal: {ws['weak_signal'][:150]}")
            ws_lines.append(f"  Driver: {ws['driver'][:150]}")
    ws_context = "\n".join(ws_lines)

    feedback_block = ""
    if facilitator_summary and round_num > 1:
        feedback_block = f"""
Anonymized panel aggregate from previous round:
{facilitator_summary}
Review the panel signals. Revise where warranted.
"""

    user = f"""Your publication record (pre-{TEMPORAL_CUTOFF}):

{sci_context}

---
{ws_context}

---
TASK (Round {round_num} of {N_ROUNDS}):

Using your expertise, your publication record, and the foresight signals above,
assess each of the 11 CE research clusters for 2021-2025 growth potential.

1. Growth potential score (P10/P50/P90) in [0,1].
2. Emerging terms: 3 specific terms expected to emerge post-2020.

Clusters:
{CLUSTER_LIST_STR}

{feedback_block}
Respond ONLY with this JSON:
{{
  "clusters": {{
    "C2":  {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>, "emerging_terms": ["t1","t2","t3"]}},
    "C3":  {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>, "emerging_terms": ["t1","t2","t3"]}},
    "C4":  {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>, "emerging_terms": ["t1","t2","t3"]}},
    "C5":  {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>, "emerging_terms": ["t1","t2","t3"]}},
    "C6":  {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>, "emerging_terms": ["t1","t2","t3"]}},
    "C7":  {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>, "emerging_terms": ["t1","t2","t3"]}},
    "C8":  {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>, "emerging_terms": ["t1","t2","t3"]}},
    "C9":  {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>, "emerging_terms": ["t1","t2","t3"]}},
    "C10": {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>, "emerging_terms": ["t1","t2","t3"]}},
    "C11": {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>, "emerging_terms": ["t1","t2","t3"]}},
    "C14": {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>, "emerging_terms": ["t1","t2","t3"]}}
  }},
  "rationale": "<1-2 sentences>"
}}
"""
    return [{"role": "system", "content": system},
            {"role": "user",   "content": user}]



def build_prompt_condition_G(expert: dict, rag_docs: dict, round_num: int,
                              facilitator_summary: str | None,
                              bridge_terms: dict) -> list:
    """
    Condition G: E + bridge term injection for AWS-isolated clusters only.
    C10, C11, C14 receive their distinctive terms + bridge terms to neighbours.
    All other clusters: identical to E.
    """
    system = (
        f"{_specialist_role(expert)} "
        "You are participating in a structured Delphi study. "
        "Respond ONLY with valid JSON — no extra text."
    )

    sci_context = format_scientific_context(rag_docs["scientific"], max_chars=300)

    # Build bridge context — only for isolated clusters
    ISOLATED = {"C10", "C11", "C14"}
    bridge_lines = []
    for ckey, info in bridge_terms.items():
        if ckey not in ISOLATED:
            continue
        label = info["label"]
        dist  = info.get("distinctive_terms", [])[:5]
        neighbours = info.get("neighbours", [])[:2]

        bridge_lines.append(f"\n{ckey} ({label}) — vocabulary bridge context:")
        if dist:
            bridge_lines.append(f"  Distinctive technical terms in this cluster: {', '.join(dist)}")
        for n in neighbours:
            shared = n.get("bridge_terms", [])[:3]
            neigh_top = n.get("neigh_top5", [])[:3]
            bridge_lines.append(
                f"  Connects to {n['cluster']} ({n['label'][:30]}) via: {', '.join(shared)}"
            )
            bridge_lines.append(
                f"  That cluster uses: {', '.join(neigh_top)}"
            )

    bridge_context = ""
    if bridge_lines:
        bridge_context = (
            "\n---\nVocabulary bridge context for structurally isolated clusters\n"
            "(derived from citation network analysis — use to inform growth estimates):\n"
            + "\n".join(bridge_lines)
            + "\n"
        )

    feedback_block = ""
    if facilitator_summary and round_num > 1:
        feedback_block = f"""
Anonymized panel aggregate from previous round:
{facilitator_summary}
Review the panel signals. Revise where warranted.
"""

    user = f"""Your publication record (pre-{TEMPORAL_CUTOFF}):

{sci_context}
{bridge_context}
---
TASK (Round {round_num} of {N_ROUNDS}):

Assess each of the 11 CE research clusters for 2021-2025 growth potential.
For clusters C10, C11, C14: use the bridge context above to inform your
estimates — these clusters are terminologically isolated but structurally
connected to the CE corpus.

1. Growth potential score (P10/P50/P90) in [0,1].
2. Emerging terms: 3 specific terms expected to emerge post-2020.
   For C10, C11, C14: include at least 1 term from the bridge context.

Clusters:
{CLUSTER_LIST_STR}

{feedback_block}
Respond ONLY with this JSON:
{{
  "clusters": {{
    "C2":  {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>, "emerging_terms": ["t1","t2","t3"]}},
    "C3":  {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>, "emerging_terms": ["t1","t2","t3"]}},
    "C4":  {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>, "emerging_terms": ["t1","t2","t3"]}},
    "C5":  {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>, "emerging_terms": ["t1","t2","t3"]}},
    "C6":  {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>, "emerging_terms": ["t1","t2","t3"]}},
    "C7":  {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>, "emerging_terms": ["t1","t2","t3"]}},
    "C8":  {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>, "emerging_terms": ["t1","t2","t3"]}},
    "C9":  {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>, "emerging_terms": ["t1","t2","t3"]}},
    "C10": {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>, "emerging_terms": ["t1","t2","t3"]}},
    "C11": {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>, "emerging_terms": ["t1","t2","t3"]}},
    "C14": {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>, "emerging_terms": ["t1","t2","t3"]}}
  }},
  "rationale": "<1-2 sentences>"
}}
"""
    return [{"role": "system", "content": system},
            {"role": "user",   "content": user}]



def build_prompt_condition_H(expert: dict, rag_docs: dict, round_num: int,
                              facilitator_summary: str | None,
                              bridge_authors: dict) -> list:
    """
    Condition H: Graph RAG — bridge author vocabulary injection.
    For isolated clusters (C10, C11, C14): injects vocabulary used by
    authors who publish in BOTH the isolated cluster AND CE clusters.
    These are real human translators between terminological worlds.
    All other clusters: identical to E.
    """
    system = (
        f"{_specialist_role(expert)} "
        "You are participating in a structured Delphi study. "
        "Respond ONLY with valid JSON — no extra text."
    )

    sci_context = format_scientific_context(rag_docs["scientific"], max_chars=300)

    # Build graph RAG context — only for isolated clusters
    ISOLATED = {"10", "11", "14"}
    graph_lines = []
    for ckey, info in bridge_authors.items():
        if ckey not in ISOLATED:
            continue
        label       = info["label"]
        n_authors   = info["bridge_authors_n"]
        iso_vocab   = info.get("iso_vocab", [])[:5]
        translation = info.get("translation_terms", [])[:6]
        ce_clusters = info.get("author_ce_clusters", [])[:3]

        if not translation and not iso_vocab:
            continue

        graph_lines.append(f"\nC{ckey} ({label}):")
        graph_lines.append(
            f"  {n_authors} researchers publish in both this cluster and CE literature."
        )
        if ce_clusters:
            graph_lines.append(f"  Their CE work spans: {', '.join(ce_clusters)}")
        if iso_vocab:
            graph_lines.append(
                f"  Cluster-specific terms they use: {', '.join(iso_vocab)}"
            )
        if translation:
            graph_lines.append(
                f"  Bridge vocabulary (used in both worlds): {', '.join(translation)}"
            )

    graph_context = ""
    if graph_lines:
        graph_context = (
            "\n---\nGraph RAG — bridge researcher vocabulary\n"
            "(researchers who publish across isolated and CE clusters; "
            "use their vocabulary to inform growth estimates for C10, C11, C14):\n"
            + "\n".join(graph_lines)
            + "\n"
        )

    feedback_block = ""
    if facilitator_summary and round_num > 1:
        feedback_block = f"""
Anonymized panel aggregate from previous round:
{facilitator_summary}
Review the panel signals. Revise where warranted.
"""

    user = f"""Your publication record (pre-{TEMPORAL_CUTOFF}):

{sci_context}
{graph_context}
---
TASK (Round {round_num} of {N_ROUNDS}):

Assess each of the 11 CE research clusters for 2021-2025 growth potential.
For clusters C10, C11, C14: use the bridge researcher vocabulary above to
inform your emerging term predictions — these researchers span both worlds.

1. Growth potential score (P10/P50/P90) in [0,1].
2. Emerging terms: 3 specific terms expected to emerge post-2020.
   For C10, C11, C14: draw from the bridge vocabulary provided.

Clusters:
{CLUSTER_LIST_STR}

{feedback_block}
Respond ONLY with this JSON:
{{
  "clusters": {{
    "C2":  {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>, "emerging_terms": ["t1","t2","t3"]}},
    "C3":  {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>, "emerging_terms": ["t1","t2","t3"]}},
    "C4":  {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>, "emerging_terms": ["t1","t2","t3"]}},
    "C5":  {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>, "emerging_terms": ["t1","t2","t3"]}},
    "C6":  {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>, "emerging_terms": ["t1","t2","t3"]}},
    "C7":  {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>, "emerging_terms": ["t1","t2","t3"]}},
    "C8":  {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>, "emerging_terms": ["t1","t2","t3"]}},
    "C9":  {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>, "emerging_terms": ["t1","t2","t3"]}},
    "C10": {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>, "emerging_terms": ["t1","t2","t3"]}},
    "C11": {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>, "emerging_terms": ["t1","t2","t3"]}},
    "C14": {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>, "emerging_terms": ["t1","t2","t3"]}}
  }},
  "rationale": "<1-2 sentences>"
}}
"""
    return [{"role": "system", "content": system},
            {"role": "user",   "content": user}]

def parse_response(text: str, condition: str) -> dict | None:
    """
    Parse LLM response for all three conditions.
    Returns dict with 'clusters' key or None on failure.
    """
    if not text:
        return None
    try:
        # Strip reasoning blocks
        if "<think>" in text:
            end = text.find("</think>")
            if end != -1:
                text = text[end + 8:].strip()

        start = text.find("{")
        end   = text.rfind("}") + 1
        if start == -1 or end == 0:
            return None

        data = json.loads(text[start:end])
        if "clusters" not in data:
            return None

        parsed = {}
        for c in CLUSTERS:
            key = f"C{c}"
            if key not in data["clusters"]:
                return None
            entry = data["clusters"][key]

            p10 = float(entry.get("growth_p10", entry.get("p10", 0)))
            p50 = float(entry.get("growth_p50", entry.get("p50", 0)))
            p90 = float(entry.get("growth_p90", entry.get("p90", 0)))

            if not (0.0 <= p10 <= p50 <= p90 <= 1.0):
                return None

            terms = entry.get("emerging_terms", entry.get("terms", []))
            if not isinstance(terms, list):
                terms = []

            cluster_data = {
                "growth_p10":     p10,
                "growth_p50":     p50,
                "growth_p90":     p90,
                "emerging_terms": [str(t).strip().lower() for t in terms[:6]],
            }

            if condition == "C":
                cluster_data["weak_signals"] = entry.get("weak_signals", [])
                cluster_data["wild_card"]    = entry.get("wild_card", None)

            parsed[c] = cluster_data

        return {
            "clusters":  parsed,
            "rationale": data.get("rationale", ""),
        }

    except Exception:
        return None


# ---------------------------------------------------------------------------
# MODULE 6 — DELPHI FACILITATOR
# ---------------------------------------------------------------------------

def build_facilitator_summary(round_results: list) -> tuple[str, dict]:
    """
    Build anonymized aggregate for next round feedback.
    Returns (summary_text, stats_dict).
    """
    cluster_stats = {}
    for c in CLUSTERS:
        p50_vals = []
        widths   = []
        all_terms = []
        for r in round_results:
            if r and "clusters" in r and c in r["clusters"]:
                entry = r["clusters"][c]
                p50_vals.append(entry.get("growth_p50", entry.get("p50", 0)))
                widths.append(entry.get("growth_p90", entry.get("p90", 0)) - entry.get("growth_p10", entry.get("p10", 0)))
                all_terms.extend(entry.get("emerging_terms", entry.get("terms", [])))
        if p50_vals:
            mean_p50 = sum(p50_vals) / len(p50_vals)
            std_p50  = (sum((v-mean_p50)**2 for v in p50_vals) / len(p50_vals)) ** 0.5
            mean_w   = sum(widths) / len(widths)
            # Top 5 most mentioned terms
            from collections import Counter
            top_terms = [t for t, _ in Counter(all_terms).most_common(5)]
            cluster_stats[c] = {
                "mean_p50":  round(mean_p50, 3),
                "std_p50":   round(std_p50, 3),
                "mean_width": round(mean_w, 3),
                "top_terms": top_terms,
            }

    lines = ["Panel aggregate (anonymized) — growth_p50 mean ± std | top emerging terms:"]
    for c in CLUSTERS:
        if c in cluster_stats:
            s = cluster_stats[c]
            lines.append(
                f"  C{c} ({CLUSTER_LABELS[c][:35]}): "
                f"p50={s['mean_p50']:.3f} ± {s['std_p50']:.3f} | "
                f"top terms: {', '.join(s['top_terms'][:3])}"
            )

    return "\n".join(lines), cluster_stats


def compute_convergence(all_round_results: list) -> list:
    """CV, IQR, mean interval width per cluster per round."""
    import statistics
    metrics = []
    for round_idx, round_results in enumerate(all_round_results):
        valid = [r for r in round_results if r and "clusters" in r]
        if not valid:
            continue
        row = {"round": round_idx + 1}
        for c in CLUSTERS:
            p50_vals = [r["clusters"][c].get("growth_p50", r["clusters"][c].get("p50", 0))
                        for r in valid if c in r["clusters"]]
            widths   = [r["clusters"][c].get("growth_p90", r["clusters"][c].get("p90",0)) - r["clusters"][c].get("growth_p10", r["clusters"][c].get("p10",0))
                        for r in valid if c in r["clusters"]]
            if len(p50_vals) < 2:
                continue
            mean = sum(p50_vals) / len(p50_vals)
            std  = statistics.stdev(p50_vals)
            cv   = std / mean if mean > 0 else float("inf")
            sp   = sorted(p50_vals)
            q1   = sp[len(sp)//4]
            q3   = sp[3*len(sp)//4]
            row[f"C{c}_cv"]    = round(cv, 4)
            row[f"C{c}_iqr"]   = round(q3 - q1, 4)
            row[f"C{c}_width"] = round(sum(widths)/len(widths), 4)
        metrics.append(row)
    return metrics


# ---------------------------------------------------------------------------
# MODULE 7 — VALIDATION
# ---------------------------------------------------------------------------

def compute_spearman_validation(all_results: list) -> dict:
    """
    Spearman rho between panel's mean growth_p50 and observed growth (n=11).
    Critical value: rho >= 0.618 for p < 0.05 (two-tailed, n=11).
    """
    from scipy.stats import spearmanr

    # Aggregate mean p50 per cluster across all experts
    cluster_p50 = {}
    for c in CLUSTERS:
        vals = []
        for r in all_results:
            if r and "clusters" in r and c in r["clusters"]:
                vals.append(r["clusters"][c].get("growth_p50", r["clusters"][c].get("p50", 0)))
        if vals:
            cluster_p50[c] = sum(vals) / len(vals)

    pred = [cluster_p50.get(c, 0) for c in CLUSTERS]
    obs  = [OBSERVED_GROWTH[c] for c in CLUSTERS]

    rho, p = spearmanr(pred, obs)

    ranked_pred = sorted(CLUSTERS, key=lambda c: -cluster_p50.get(c, 0))
    ranked_obs  = sorted(CLUSTERS, key=lambda c: -OBSERVED_GROWTH[c])

    return {
        "spearman_rho":   round(rho, 4),
        "p_value":        round(p, 4),
        "significant":    int(p < 0.05),
        "ranked_pred":    [f"C{c}" for c in ranked_pred],
        "ranked_obs":     [f"C{c}" for c in ranked_obs],
        "cluster_p50":    {f"C{c}": round(v, 4) for c, v in cluster_p50.items()},
        "critical_rho_05": 0.618,
        "note": "n=11; critical rho=0.618 for p<0.05 two-tailed (Zar 1984)"
    }


def compute_signal_recall(all_results: list, ground_truth_terms: dict) -> dict:
    """
    Recall of emerging terms: what fraction of ground-truth emerging terms
    were mentioned by the panel?

    ground_truth_terms: {cluster_id: [term1, term2, ...]} from paper_topics_clean
    Uses simple lexical overlap (exact + substring match).
    """
    results = {}
    for c in CLUSTERS:
        gt = set(t.lower() for t in ground_truth_terms.get(c, []))
        if not gt:
            continue

        panel_terms = set()
        for r in all_results:
            if r and "clusters" in r and c in r["clusters"]:
                for t in r["clusters"][c].get("emerging_terms", []):
                    panel_terms.add(t.lower())

        # Exact match
        exact_hits = gt & panel_terms

        # Substring match (panel term appears in ground truth term or vice versa)
        substr_hits = set()
        for pt in panel_terms:
            for gt_t in gt:
                if pt in gt_t or gt_t in pt:
                    substr_hits.add(gt_t)

        all_hits  = exact_hits | substr_hits
        recall    = len(all_hits) / len(gt) if gt else 0
        precision = len(all_hits) / len(panel_terms) if panel_terms else 0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0)

        results[c] = {
            "gt_terms":     list(gt),
            "panel_terms":  list(panel_terms)[:20],
            "hits":         list(all_hits),
            "recall":       round(recall, 3),
            "precision":    round(precision, 3),
            "f1":           round(f1, 3),
        }

    macro_recall = sum(v["recall"] for v in results.values()) / len(results)
    macro_f1     = sum(v["f1"]     for v in results.values()) / len(results)

    return {
        "per_cluster":    results,
        "macro_recall":   round(macro_recall, 3),
        "macro_f1":       round(macro_f1, 3),
    }


# ---------------------------------------------------------------------------
# MODULE 8 — PARALLEL EXPERT PROCESSING
# ---------------------------------------------------------------------------

def process_one_expert(args):
    (i, expert, rag_docs, round_num, facilitator_summary,
     condition, weak_signals, bridge_terms, bridge_authors) = args

    if condition == "A":
        messages = build_prompt_condition_A(expert, round_num, facilitator_summary)
    elif condition in ("B", "D"):
        messages = build_prompt_condition_B(expert, rag_docs, round_num,
                                            facilitator_summary,
                                            abstract_only=(condition == "D"))
    elif condition == "E":
        messages = build_prompt_condition_E(expert, rag_docs, round_num,
                                            facilitator_summary)
    elif condition == "F":
        messages = build_prompt_condition_F(expert, rag_docs, round_num,
                                            facilitator_summary, weak_signals)
    elif condition == "G":
        messages = build_prompt_condition_G(expert, rag_docs, round_num,
                                            facilitator_summary, bridge_terms)
    elif condition == "H":
        messages = build_prompt_condition_H(expert, rag_docs, round_num,
                                            facilitator_summary, bridge_authors)
    else:
        messages = build_prompt_condition_C(expert, rag_docs, round_num,
                                            facilitator_summary)

    max_tok  = 3000 if condition == "C" else 1800
    raw      = _call_llm_raw(expert["model"], messages,
                              temperature=0.7, max_tokens=max_tok)
    parsed   = parse_response(raw, condition)
    if parsed:
        parsed["expert_id"] = expert["expert_id"]
        parsed["model"]     = expert["model"]
        parsed["round"]     = round_num
    return i, parsed


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def run_pipeline(condition: str, analysis_only: bool, n_rounds: int):
    output_dir = get_output_dir(condition)

    print("=" * 60)
    print(f"CE-Delphi v3 — Condition {condition} | {n_rounds} rounds")
    print("=" * 60)

    # Load corpora
    sci_corpus  = load_scientific_corpus() if condition in ("B", "C", "D", "E", "F", "G", "H") else {}
    news_corpus = load_news_corpus()        if condition in ("C", "E") else {}

    # Load pre-extracted weak signals for condition F
    weak_signals = {}
    if condition == "F":
        ws_path = Path.home() / "Desktop/openalex/weak_signals.json"
        if ws_path.exists():
            weak_signals = json.loads(ws_path.read_text())
            print(f"Weak signals loaded: {len(weak_signals)} clusters")
        else:
            print("WARNING: weak_signals.json not found — run extract_weak_signals.py first")

    # Load bridge terms for condition G (only for isolated clusters C10, C11, C14)
    bridge_terms = {}
    if condition == "G":
        bt_path = Path.home() / "Desktop/openalex/bridge_terms.json"
        if bt_path.exists():
            bridge_terms = json.loads(bt_path.read_text())
            print(f"Bridge terms loaded for isolated clusters: {list(bridge_terms.keys())}")
        else:
            print("WARNING: bridge_terms.json not found — run condition_g_bridge_terms.py first")

    # Load bridge authors for condition H (Graph RAG)
    bridge_authors = {}
    if condition == "H":
        ba_path = Path.home() / "Desktop/openalex/bridge_authors.json"
        if ba_path.exists():
            bridge_authors = json.loads(ba_path.read_text())
            print(f"Bridge authors loaded for isolated clusters: {list(bridge_authors.keys())}")
        else:
            print("WARNING: bridge_authors.json not found — run condition_h_graph_rag.py first")

    if condition in ("B", "C", "D", "E", "F", "G", "H") and not any(sci_corpus.values()):
        print("ERROR: scientific corpus empty. Check FULLTEXT_BASE path.")
        return

    # Generate experts
    rng     = random.Random(42)
    experts = []
    for model_idx, model in enumerate(MODELS):
        for expert_idx in range(N_EXPERTS_PER_MODEL):
            e = generate_expert_profile(model_idx, expert_idx)
            experts.append(e)

    n_experts = len(experts)
    print(f"Experts: {n_experts} ({len(MODELS)} models × {N_EXPERTS_PER_MODEL})")

    # Build RAG corpora per expert
    rag_corpora = [
        build_expert_rag(e, sci_corpus, news_corpus, condition, rng)
        for e in experts
    ]

    # Resume logic
    all_round_results = []
    facilitator_summary = None
    start_round = 1

    for r in range(1, n_rounds + 1):
        f = output_dir / f"round_{r}_results.json"
        if f.exists():
            loaded = json.loads(f.read_text())
            all_round_results.append(loaded)
            valid = [x for x in loaded if x]
            if valid:
                facilitator_summary, _ = build_facilitator_summary(valid)
            start_round = r + 1
            print(f"[RESUME] Round {r}: {len(valid)} valid responses")

    if analysis_only:
        if not all_round_results:
            print("ERROR: No saved rounds found.")
            return
        print(f"[ANALYSIS-ONLY] {len(all_round_results)} rounds loaded")
    else:
        for round_num in range(start_round, n_rounds + 1):
            print(f"\n{'='*50}\nROUND {round_num}/{n_rounds}\n{'='*50}")

            results_map = {}
            counter     = {"ok": 0, "fail": 0}
            task_args   = [
                (i, experts[i], rag_corpora[i], round_num,
                 facilitator_summary, condition, weak_signals, bridge_terms, bridge_authors)
                for i in range(n_experts)
            ]

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {executor.submit(process_one_expert, a): a[0]
                           for a in task_args}
                for future in as_completed(futures):
                    i, parsed = future.result()
                    results_map[i] = parsed
                    with _progress_lock:
                        if parsed:
                            counter["ok"] += 1
                        else:
                            counter["fail"] += 1
                        done = counter["ok"] + counter["fail"]
                        if done % 10 == 0 or done == n_experts:
                            print(f"  {done}/{n_experts} | ok={counter['ok']} "
                                  f"fail={counter['fail']}")

            round_results = [results_map.get(i) for i in range(n_experts)]
            all_round_results.append(round_results)

            (output_dir / f"round_{round_num}_results.json").write_text(
                json.dumps(round_results, indent=2)
            )

            valid = [r for r in round_results if r]
            if valid:
                facilitator_summary, stats = build_facilitator_summary(valid)
                print(f"\nRound {round_num} ({len(valid)}/{n_experts} valid):")
                top3 = sorted(CLUSTERS,
                              key=lambda c: -stats.get(c, {}).get("mean_p50", 0))[:3]
                for c in top3:
                    s = stats[c]
                    print(f"  C{c}: p50={s['mean_p50']:.3f} ± {s['std_p50']:.3f} "
                          f"| {s['top_terms'][:3]}")

    # -----------------------------------------------------------------------
    # ANALYSIS
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}\nANALYSIS — CONDITION {condition}\n{'='*60}")

    valid_final = [r for r in all_round_results[-1] if r]

    # Convergence
    convergence = compute_convergence(all_round_results)
    print("\n--- CONVERGENCE ---")
    for row in convergence:
        cvs = [row.get(f"C{c}_cv", float("nan")) for c in CLUSTERS
               if f"C{c}_cv" in row]
        widths = [row.get(f"C{c}_width", float("nan")) for c in CLUSTERS
                  if f"C{c}_width" in row]
        mean_cv = sum(cvs)/len(cvs) if cvs else float("nan")
        mean_w  = sum(widths)/len(widths) if widths else float("nan")
        print(f"  Round {row['round']}: mean_CV={mean_cv:.4f}, "
              f"mean_width={mean_w:.4f}")

    # Spearman
    spearman = compute_spearman_validation(valid_final)
    print(f"\n--- SPEARMAN (n=11) ---")
    print(f"  rho={spearman['spearman_rho']}, p={spearman['p_value']}, "
          f"significant={spearman['significant']}")
    print(f"  Predicted: {spearman['ranked_pred']}")
    print(f"  Observed:  {spearman['ranked_obs']}")

    # Signal recall (requires ground truth — built from paper_topics_clean)
    gt_path = Path.home() / "Desktop/openalex/results_circular_economy/full_corpus/paper_topics_clean.csv"
    signal_recall = {}
    if gt_path.exists():
        try:
            import pandas as pd
            from collections import Counter
            df = pd.read_csv(gt_path)
            ground_truth = {}
            for c in CLUSTERS:
                sub  = df[df["cluster"] == c]
                post = sub[sub["year"].between(2021, 2024)]
                pre  = sub[sub["year"].between(2016, 2020)]
                post_terms = Counter()
                pre_terms  = Counter()
                for _, row in post.iterrows():
                    if pd.notna(row["topic_label"]):
                        for t in str(row["topic_label"]).split(","):
                            t = t.strip().lower()
                            if len(t) > 3:
                                post_terms[t] += 1
                for _, row in pre.iterrows():
                    if pd.notna(row["topic_label"]):
                        for t in str(row["topic_label"]).split(","):
                            t = t.strip().lower()
                            if len(t) > 3:
                                pre_terms[t] += 1
                # Emerging: post_count >= 3 AND ratio > 2
                emerging = [t for t in post_terms
                            if post_terms[t] >= 3
                            and post_terms[t] / max(pre_terms.get(t, 0), 1) > 2.0]
                ground_truth[c] = emerging[:15]

            signal_recall = compute_signal_recall(valid_final, ground_truth)
            print(f"\n--- SIGNAL RECALL ---")
            print(f"  Macro recall: {signal_recall['macro_recall']}")
            print(f"  Macro F1:     {signal_recall['macro_f1']}")
            for c in CLUSTERS:
                r = signal_recall["per_cluster"].get(c, {})
                print(f"  C{c}: recall={r.get('recall','?')}, "
                      f"f1={r.get('f1','?')}, "
                      f"hits={r.get('hits',[])[:3]}")
        except Exception as e:
            print(f"  Signal recall skipped: {e}")
    else:
        print(f"  Signal recall skipped: ground truth not found at {gt_path}")

    # -----------------------------------------------------------------------
    # SAVE
    # -----------------------------------------------------------------------
    summary = {
        "condition":      condition,
        "n_experts":      len(experts),
        "n_rounds":       len(all_round_results),
        "n_valid_final":  len(valid_final),
        "convergence":    convergence,
        "spearman":       spearman,
        "signal_recall":  signal_recall,
    }
    (output_dir / "pipeline_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    # Convergence CSV
    if convergence:
        with open(output_dir / "convergence.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=convergence[0].keys())
            writer.writeheader()
            writer.writerows(convergence)

    # All expert responses CSV
    with open(output_dir / "expert_responses.csv", "w", newline="") as f:
        fields = (["round", "expert_id", "model", "discipline", "region",
                   "spec_group"]
                  + [f"C{c}_p50" for c in CLUSTERS]
                  + [f"C{c}_terms" for c in CLUSTERS]
                  + ["rationale"])
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for round_idx, round_results in enumerate(all_round_results):
            for j, result in enumerate(round_results):
                if result is None:
                    continue
                e = experts[j] if j < len(experts) else {}
                row = {
                    "round":      round_idx + 1,
                    "expert_id":  result.get("expert_id", ""),
                    "model":      result.get("model", ""),
                    "discipline": e.get("discipline", ""),
                    "region":     e.get("region", ""),
                    "spec_group": e.get("spec_group", ""),
                    "rationale":  result.get("rationale", ""),
                }
                for c in CLUSTERS:
                    entry = result.get("clusters", {}).get(c, {})
                    row[f"C{c}_p50"]   = entry.get("growth_p50", "")
                    row[f"C{c}_terms"] = "|".join(
                        entry.get("emerging_terms", []))
                writer.writerow(row)

    print(f"\nSaved to {output_dir}/")
    print(f"  pipeline_summary.json")
    print(f"  convergence.csv")
    print(f"  expert_responses.csv")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CE-Delphi v3: three-condition ablation on 11 CE clusters"
    )
    parser.add_argument(
        "--condition", choices=["A", "B", "C", "D", "E", "F", "G", "H"], default="B",
        help="A=no-RAG | B=RAG-lit+role | C=RAG-lit+news+role+weak-signals"
    )
    parser.add_argument(
        "--analysis-only", action="store_true",
        help="Recompute analysis from saved rounds without new API calls"
    )
    parser.add_argument(
        "--rounds", type=int, default=N_ROUNDS, choices=[1, 2, 3],
        help="Number of Delphi rounds (default: 3)"
    )
    parser.add_argument(
        "--build-news", action="store_true",
        help="Build synthetic news corpus for Condition C (run once)"
    )
    args = parser.parse_args()

    if args.build_news:
        if not OPENROUTER_API_KEY:
            print("ERROR: OPENROUTER_API_KEY not set.")
            return
        build_news_corpus()
        return

    if not args.analysis_only and not OPENROUTER_API_KEY:
        print("ERROR: OPENROUTER_API_KEY not set. Check .env file.")
        return

    run_pipeline(args.condition, args.analysis_only, args.rounds)


if __name__ == "__main__":
    main()