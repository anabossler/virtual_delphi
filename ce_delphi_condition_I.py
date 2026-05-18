#!/usr/bin/env python3
"""
CE-Delphi Condition I — Prospective Horizon 2030-2040
======================================================

Extends ce_delphi_v3.py with a new condition I for long-horizon foresight.

KEY DIFFERENCES from condition G:
  - Horizon: 2030-2040 (not 2021-2025)
  - Prior context: Round 3 consensus from condition G injected as "known past"
  - Models: Claude Haiku 4.5 + Gemma-3-27B only (most consistent in G)
  - Same bridge terms as G for isolated clusters
  - Output: emerging_terms_2030 per cluster + wild_card per cluster
  - No ground truth validation (prospective — not verifiable yet)
  - Convergence only metric

USAGE:
  conda activate aws
  # Copy this file next to ce_delphi_v3.py, then:
  python ce_delphi_condition_I.py

OUTPUT:
  ~/Desktop/openalex/ce_delphi_v3_results/condition_I/
    pipeline_summary.json
    convergence.csv
    expert_responses.csv
    round_1_results.json
    round_2_results.json
    round_3_results.json
"""

import json
import os
import random
import time
import csv
import argparse
import threading
from pathlib import Path
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import requests
import statistics

load_dotenv(Path.home() / "Desktop/openalex/.env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"

# ── Only the two most consistent models from condition G ──────────────────
MODELS_I = [
    "anthropic/claude-haiku-4-5",   # mean=0.781, std=0.039
    "google/gemma-3-27b-it",         # mean=0.776, std=0.015
]

N_EXPERTS_PER_MODEL = 10   # 20 total — sufficient for convergence
N_ROUNDS            = 3
MAX_WORKERS         = 3
TEMPORAL_CUTOFF     = 2020
HORIZON             = "2030-2040"

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

FULLTEXT_BASE = Path.home() / "Desktop/openalex/cluster_fulltexts"
NEWS_BASE     = Path.home() / "Desktop/openalex/cluster_news"
RESULTS_BASE  = Path.home() / "Desktop/openalex/ce_delphi_v3_results"

_progress_lock = threading.Lock()

CLUSTER_LIST_STR = "\n".join(
    f"  C{c}: {CLUSTER_LABELS[c]}" for c in CLUSTERS
)


# ── Load Round 3 consensus from condition G as prior ──────────────────────

def load_condition_G_prior() -> str:
    """
    Load condition G Round 3 summary as prior context for condition I.
    Returns a formatted string of confirmed 2021-2024 trends per cluster.
    """
    summary_path = RESULTS_BASE / "condition_G" / "pipeline_summary.json"
    if not summary_path.exists():
        print("WARNING: condition_G pipeline_summary.json not found.")
        return "(No prior data available.)"

    with open(summary_path) as f:
        summary = json.load(f)

    sr = summary.get("signal_recall", {}).get("per_cluster", {})
    sp = summary.get("spearman", {})
    cluster_p50 = sp.get("cluster_p50", {})

    lines = [
        "CONFIRMED EMERGING TRENDS 2021-2024 (from validated panel, condition G):",
        "These trends have been empirically confirmed against OpenAlex data.",
        "Use them as the baseline from which to project 2030-2040 trajectories.",
        ""
    ]

    for c in CLUSTERS:
        ckey = f"C{c}"
        label = CLUSTER_LABELS[c]
        cluster_data = sr.get(str(c), {})
        hits = cluster_data.get("hits", [])
        panel_terms = cluster_data.get("panel_terms", [])[:5]
        p50 = cluster_p50.get(ckey, "N/A")

        lines.append(f"C{c} — {label}:")
        lines.append(f"  Confirmed 2021-2024 trends: {', '.join(hits) if hits else 'none confirmed'}")
        lines.append(f"  Additional panel signals: {', '.join(panel_terms[:3])}")
        lines.append(f"  Growth score 2021-2024: {p50}")
        lines.append("")

    return "\n".join(lines)


# ── Corpus loaders (same as v3) ────────────────────────────────────────────

def load_scientific_corpus() -> dict:
    corpus = defaultdict(list)
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
                    "doi":   meta.get("doi", ""),
                    "title": meta.get("title", ""),
                    "year":  year,
                    "cluster": cluster_id,
                    "text":  text[:3000],
                })
            except Exception:
                continue
    return corpus


def load_bridge_terms() -> dict:
    bt_path = Path.home() / "Desktop/openalex/bridge_terms.json"
    if bt_path.exists():
        with open(bt_path) as f:
            return json.load(f)
    print("WARNING: bridge_terms.json not found.")
    return {}


# ── Expert generation ──────────────────────────────────────────────────────

def generate_expert_profile(model_idx: int, expert_idx: int) -> dict:
    discipline = DISCIPLINES[(model_idx + expert_idx) % len(DISCIPLINES)]
    region     = REGIONS[(model_idx * 3 + expert_idx) % len(REGIONS)]
    seniority  = SENIORITY[expert_idx % len(SENIORITY)]
    approach   = APPROACHES[(model_idx + expert_idx * 2) % len(APPROACHES)]
    years_exp  = 8 + (expert_idx % 20) if seniority == "mid" else 18 + (expert_idx % 15)

    group_keys    = list(CLUSTER_GROUPS.keys())
    group_key     = group_keys[(model_idx + expert_idx) % len(group_keys)]
    spec_clusters = CLUSTER_GROUPS[group_key]

    return {
        "expert_id":        f"I_M{model_idx:02d}_E{expert_idx:02d}",
        "model":            MODELS_I[model_idx],
        "discipline":       discipline,
        "region":           region,
        "seniority":        seniority,
        "approach":         approach,
        "spec_group":       group_key,
        "spec_clusters":    spec_clusters,
        "years_experience": years_exp,
    }


def build_expert_rag(expert: dict, sci_corpus: dict, rng: random.Random) -> dict:
    """1 abstract per assigned cluster — same as condition G/E."""
    spec_clusters = expert["spec_clusters"]
    sci_docs = []
    for c in CLUSTERS:
        pool = sci_corpus.get(c, [])
        if pool and c in spec_clusters:
            sci_docs.extend(rng.choices(pool, k=1))
    return {"scientific": sci_docs}


def format_scientific_context(docs: list) -> str:
    if not docs:
        return "(No scientific publications available.)"
    lines = []
    for i, doc in enumerate(docs, 1):
        c = doc.get("cluster")
        lines.append(f"--- Publication {i} [ABSTRACT] ---")
        lines.append(f"Title: {doc['title']}")
        if doc.get("year") and str(doc["year"]).lower() != "nan":
            lines.append(f"Year: {int(float(doc['year']))}")
        if c:
            lines.append(f"Research area: {CLUSTER_LABELS.get(c, '')}")
        lines.append(doc["text"][:300])
        lines.append("")
    return "\n".join(lines)


# ── Prompt for condition I ─────────────────────────────────────────────────

def build_prompt_condition_I(expert: dict, rag_docs: dict, round_num: int,
                              facilitator_summary: str | None,
                              g_prior: str,
                              bridge_terms: dict) -> list:
    """
    Condition I: long-horizon prospective Delphi 2030-2040.

    Context layers:
    1. Specialist role
    2. Scientific abstracts (pre-2020, same as G)
    3. Confirmed 2021-2024 trends from condition G (prior)
    4. Bridge terms for isolated clusters (same as G)
    5. Facilitator summary from previous rounds
    """
    system = (
        f"You are a senior circular economy researcher with "
        f"{expert['years_experience']} years of experience in "
        f"{expert['discipline'].replace('_', ' ')} ({expert['region']}). "
        f"You are participating in a structured Delphi foresight exercise "
        f"focused on the LONG-TERM horizon 2030-2040. "
        f"Your methodological approach is {expert['approach']}. "
        f"Respond ONLY with valid JSON — no extra text."
    )

    sci_context = format_scientific_context(rag_docs["scientific"])

    # Bridge context for isolated clusters
    ISOLATED = {"C10", "C11", "C14"}
    bridge_lines = []
    for ckey, info in bridge_terms.items():
        if ckey not in ISOLATED:
            continue
        label = info["label"]
        dist  = info.get("distinctive_terms", [])[:5]
        neighbours = info.get("neighbours", [])[:2]
        bridge_lines.append(f"\n{ckey} ({label}) — vocabulary bridge:")
        if dist:
            bridge_lines.append(f"  Technical terms: {', '.join(dist)}")
        for n in neighbours:
            shared = n.get("bridge_terms", [])[:3]
            bridge_lines.append(
                f"  Connects to {n['cluster']} via: {', '.join(shared)}"
            )

    bridge_context = ""
    if bridge_lines:
        bridge_context = (
            "\n---\nVocabulary bridge context for structurally isolated clusters:\n"
            + "\n".join(bridge_lines) + "\n"
        )

    feedback_block = ""
    if facilitator_summary and round_num > 1:
        feedback_block = f"""
---
Panel aggregate from Round {round_num - 1}:
{facilitator_summary}
Review the panel signals. Revise where warranted, especially where
your 2030-2040 trajectories differ from the panel consensus.
"""

    user = f"""Your publication record (pre-{TEMPORAL_CUTOFF}):

{sci_context}

---
{g_prior}

{bridge_context}
{feedback_block}
---
TASK (Round {round_num} of {N_ROUNDS}): LONG-HORIZON FORESIGHT 2030-2040

The confirmed 2021-2024 trends above are the STARTING POINT.
Your task is to forecast what comes NEXT — the research directions
that will DOMINATE each CE subdomain in the period 2030-2040,
given the trajectory already visible in 2021-2024.

For each of the 11 CE research clusters:

1. GROWTH SCORE (P10/P50/P90 in [0,1]):
   Relative growth potential for 2030-2040 compared to other clusters.
   Consider: EU Green Deal 2050 net-zero targets, material scarcity,
   demographic shifts, AI/automation convergence, geopolitical supply
   chain restructuring.

2. EMERGING TERMS 2030-2040 (3-5 terms):
   Specific research directions, technologies, or concepts you expect
   to EMERGE OR DOMINATE by 2030-2040 — building on but going BEYOND
   the confirmed 2021-2024 trends.
   Be specific: not "circular economy" but "enzymatic depolymerisation
   of mixed plastics" or "carbon-negative geopolymer cements".

3. WILD CARD (1 per cluster):
   A low-probability but high-impact development that could
   radically reshape this subdomain by 2040 if it materialises.

Research clusters:
{CLUSTER_LIST_STR}

Respond ONLY with this JSON:
{{
  "clusters": {{
    "C2":  {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>,
             "emerging_terms_2030": ["term1", "term2", "term3"],
             "wild_card": "<1 sentence>"}},
    "C3":  {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>,
             "emerging_terms_2030": ["term1", "term2", "term3"],
             "wild_card": "<1 sentence>"}},
    "C4":  {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>,
             "emerging_terms_2030": ["term1", "term2", "term3"],
             "wild_card": "<1 sentence>"}},
    "C5":  {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>,
             "emerging_terms_2030": ["term1", "term2", "term3"],
             "wild_card": "<1 sentence>"}},
    "C6":  {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>,
             "emerging_terms_2030": ["term1", "term2", "term3"],
             "wild_card": "<1 sentence>"}},
    "C7":  {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>,
             "emerging_terms_2030": ["term1", "term2", "term3"],
             "wild_card": "<1 sentence>"}},
    "C8":  {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>,
             "emerging_terms_2030": ["term1", "term2", "term3"],
             "wild_card": "<1 sentence>"}},
    "C9":  {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>,
             "emerging_terms_2030": ["term1", "term2", "term3"],
             "wild_card": "<1 sentence>"}},
    "C10": {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>,
             "emerging_terms_2030": ["term1", "term2", "term3"],
             "wild_card": "<1 sentence>"}},
    "C11": {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>,
             "emerging_terms_2030": ["term1", "term2", "term3"],
             "wild_card": "<1 sentence>"}},
    "C14": {{"growth_p10": <float>, "growth_p50": <float>, "growth_p90": <float>,
             "emerging_terms_2030": ["term1", "term2", "term3"],
             "wild_card": "<1 sentence>"}},
  }},
  "rationale": "<2-3 sentences on your overall 2030-2040 reasoning>"
}}

Constraints:
- growth_p10 <= growth_p50 <= growth_p90, all in [0,1]
- emerging_terms_2030 must be DIFFERENT from the confirmed 2021-2024 terms
- Be specific and technical — avoid generic phrases
"""
    return [{"role": "system", "content": system},
            {"role": "user",   "content": user}]


# ── Response parser ────────────────────────────────────────────────────────

def parse_response_I(text: str) -> dict | None:
    if not text:
        return None
    try:
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

            p10 = float(entry.get("growth_p10", 0))
            p50 = float(entry.get("growth_p50", 0))
            p90 = float(entry.get("growth_p90", 0))

            if not (0.0 <= p10 <= p50 <= p90 <= 1.0):
                return None

            terms = entry.get("emerging_terms_2030", entry.get("emerging_terms", []))
            if not isinstance(terms, list):
                terms = []

            parsed[c] = {
                "growth_p10":          p10,
                "growth_p50":          p50,
                "growth_p90":          p90,
                "emerging_terms_2030": [str(t).strip().lower() for t in terms[:6]],
                "wild_card":           entry.get("wild_card", ""),
            }

        return {
            "clusters":  parsed,
            "rationale": data.get("rationale", ""),
        }
    except Exception:
        return None


# ── Facilitator summary ────────────────────────────────────────────────────

def build_facilitator_summary(round_results: list) -> tuple[str, dict]:
    cluster_stats = {}
    for c in CLUSTERS:
        p50_vals  = []
        all_terms = []
        wildcards = []
        for r in round_results:
            if r and "clusters" in r and c in r["clusters"]:
                entry = r["clusters"][c]
                p50_vals.append(entry.get("growth_p50", 0))
                all_terms.extend(entry.get("emerging_terms_2030", []))
                wc = entry.get("wild_card", "")
                if wc:
                    wildcards.append(wc)
        if p50_vals:
            mean_p50 = sum(p50_vals) / len(p50_vals)
            std_p50  = statistics.stdev(p50_vals) if len(p50_vals) > 1 else 0
            top_terms = [t for t, _ in Counter(all_terms).most_common(5)]
            cluster_stats[c] = {
                "mean_p50":  round(mean_p50, 3),
                "std_p50":   round(std_p50, 3),
                "top_terms": top_terms,
                "wildcards": wildcards[:2],
            }

    lines = [f"Panel aggregate — 2030-2040 growth_p50 | top emerging terms:"]
    for c in CLUSTERS:
        if c in cluster_stats:
            s = cluster_stats[c]
            lines.append(
                f"  C{c} ({CLUSTER_LABELS[c][:35]}): "
                f"p50={s['mean_p50']:.3f} ± {s['std_p50']:.3f} | "
                f"top terms: {', '.join(s['top_terms'][:3])}"
            )

    return "\n".join(lines), cluster_stats


# ── Convergence ────────────────────────────────────────────────────────────

def compute_convergence(all_round_results: list) -> list:
    metrics = []
    for round_idx, round_results in enumerate(all_round_results):
        valid = [r for r in round_results if r and "clusters" in r]
        if not valid:
            continue
        row = {"round": round_idx + 1}
        for c in CLUSTERS:
            p50_vals = [r["clusters"][c]["growth_p50"]
                        for r in valid if c in r["clusters"]]
            if len(p50_vals) < 2:
                continue
            mean = sum(p50_vals) / len(p50_vals)
            std  = statistics.stdev(p50_vals)
            cv   = std / mean if mean > 0 else float("inf")
            sp   = sorted(p50_vals)
            q1   = sp[len(sp) // 4]
            q3   = sp[3 * len(sp) // 4]
            row[f"C{c}_cv"]  = round(cv, 4)
            row[f"C{c}_iqr"] = round(q3 - q1, 4)
        metrics.append(row)
    return metrics


# ── Aggregate final output ─────────────────────────────────────────────────

def aggregate_final_output(valid_final: list) -> dict:
    """
    Aggregate Round 3 output: top terms by frequency + wild cards.
    Returns a structured dict for saving and paper use.
    """
    output = {}
    for c in CLUSTERS:
        all_terms = []
        all_wcs   = []
        p50_vals  = []
        for r in valid_final:
            if r and "clusters" in r and c in r["clusters"]:
                entry = r["clusters"][c]
                all_terms.extend(entry.get("emerging_terms_2030", []))
                wc = entry.get("wild_card", "")
                if wc and len(wc) > 5:
                    all_wcs.append(wc)
                p50_vals.append(entry.get("growth_p50", 0))

        top_terms = [t for t, _ in Counter(all_terms).most_common(10)]
        mean_p50  = sum(p50_vals) / len(p50_vals) if p50_vals else 0

        output[c] = {
            "label":              CLUSTER_LABELS[c],
            "mean_growth_2030":   round(mean_p50, 3),
            "top_terms_2030":     top_terms,
            "wild_cards_sample":  all_wcs[:3],
            "n_experts":          len(p50_vals),
        }

    return output


# ── LLM call ──────────────────────────────────────────────────────────────

def _call_llm_raw(model, messages, temperature=0.7, max_tokens=2000):
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
            r = requests.post(OPENROUTER_URL, headers=headers,
                              json=payload, timeout=90)
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


# ── Expert processing ──────────────────────────────────────────────────────

def process_one_expert(args):
    (i, expert, rag_docs, round_num, facilitator_summary,
     g_prior, bridge_terms) = args

    messages = build_prompt_condition_I(
        expert, rag_docs, round_num,
        facilitator_summary, g_prior, bridge_terms
    )
    raw    = _call_llm_raw(expert["model"], messages,
                           temperature=0.7, max_tokens=2000)
    parsed = parse_response_I(raw)
    if parsed:
        parsed["expert_id"] = expert["expert_id"]
        parsed["model"]     = expert["model"]
        parsed["round"]     = round_num
    return i, parsed


# ── Main pipeline ──────────────────────────────────────────────────────────

def run():
    output_dir = RESULTS_BASE / "condition_I"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"CE-Delphi Condition I — Prospective Horizon {HORIZON}")
    print(f"Models: {MODELS_I}")
    print(f"Experts: {len(MODELS_I) * N_EXPERTS_PER_MODEL} total")
    print("=" * 60)

    sci_corpus   = load_scientific_corpus()
    bridge_terms = load_bridge_terms()
    g_prior      = load_condition_G_prior()

    print(f"\nCondition G prior loaded: {len(g_prior.splitlines())} lines")
    print(f"Bridge terms loaded: {list(bridge_terms.keys())}")

    rng     = random.Random(42)
    experts = []
    for model_idx in range(len(MODELS_I)):
        for expert_idx in range(N_EXPERTS_PER_MODEL):
            experts.append(generate_expert_profile(model_idx, expert_idx))

    rag_corpora = [
        build_expert_rag(e, sci_corpus, rng) for e in experts
    ]

    # Resume logic
    all_round_results   = []
    facilitator_summary = None
    start_round         = 1

    for r in range(1, N_ROUNDS + 1):
        f = output_dir / f"round_{r}_results.json"
        if f.exists():
            loaded = json.loads(f.read_text())
            all_round_results.append(loaded)
            valid = [x for x in loaded if x]
            if valid:
                facilitator_summary, _ = build_facilitator_summary(valid)
            start_round = r + 1
            print(f"[RESUME] Round {r}: {len(valid)} valid responses")

    for round_num in range(start_round, N_ROUNDS + 1):
        print(f"\n{'='*50}\nROUND {round_num}/{N_ROUNDS}\n{'='*50}")

        results_map = {}
        counter     = {"ok": 0, "fail": 0}
        task_args   = [
            (i, experts[i], rag_corpora[i], round_num,
             facilitator_summary, g_prior, bridge_terms)
            for i in range(len(experts))
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
                    if done % 5 == 0 or done == len(experts):
                        print(f"  {done}/{len(experts)} | "
                              f"ok={counter['ok']} fail={counter['fail']}")

        round_results = [results_map.get(i) for i in range(len(experts))]
        all_round_results.append(round_results)
        (output_dir / f"round_{round_num}_results.json").write_text(
            json.dumps(round_results, indent=2)
        )

        valid = [r for r in round_results if r]
        if valid:
            facilitator_summary, stats = build_facilitator_summary(valid)
            print(f"\nRound {round_num} ({len(valid)}/{len(experts)} valid):")
            top3 = sorted(CLUSTERS,
                          key=lambda c: -stats.get(c, {}).get("mean_p50", 0))[:3]
            for c in top3:
                s = stats[c]
                print(f"  C{c}: p50={s['mean_p50']:.3f} ± {s['std_p50']:.3f} "
                      f"| {s['top_terms'][:3]}")

    # Analysis
    print(f"\n{'='*60}\nANALYSIS\n{'='*60}")
    valid_final  = [r for r in all_round_results[-1] if r]
    convergence  = compute_convergence(all_round_results)
    final_output = aggregate_final_output(valid_final)

    print("\n--- CONVERGENCE ---")
    for row in convergence:
        cvs    = [row.get(f"C{c}_cv", float("nan")) for c in CLUSTERS
                  if f"C{c}_cv" in row]
        mean_cv = sum(cvs) / len(cvs) if cvs else float("nan")
        print(f"  Round {row['round']}: mean_CV={mean_cv:.4f}")

    print("\n--- 2030-2040 FORECAST (top terms per cluster) ---")
    for c in CLUSTERS:
        d = final_output[c]
        print(f"  C{c} ({CLUSTER_LABELS[c][:40]}):")
        print(f"    Growth score: {d['mean_growth_2030']}")
        print(f"    Top terms:    {d['top_terms_2030'][:5]}")
        if d["wild_cards_sample"]:
            print(f"    Wild card:    {d['wild_cards_sample'][0][:80]}")

    # Save
    summary = {
        "condition":     "I",
        "horizon":       HORIZON,
        "models":        MODELS_I,
        "n_experts":     len(experts),
        "n_rounds":      len(all_round_results),
        "n_valid_final": len(valid_final),
        "convergence":   convergence,
        "forecast_2030": final_output,
    }
    (output_dir / "pipeline_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    if convergence:
        with open(output_dir / "convergence.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=convergence[0].keys())
            writer.writeheader()
            writer.writerows(convergence)

    with open(output_dir / "expert_responses.csv", "w", newline="") as f:
        fields = (["round", "expert_id", "model", "discipline", "region"]
                  + [f"C{c}_p50" for c in CLUSTERS]
                  + [f"C{c}_terms_2030" for c in CLUSTERS]
                  + [f"C{c}_wildcard" for c in CLUSTERS]
                  + ["rationale"])
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for round_idx, round_results in enumerate(all_round_results):
            for j, result in enumerate(round_results):
                if result is None:
                    continue
                e   = experts[j] if j < len(experts) else {}
                row = {
                    "round":      round_idx + 1,
                    "expert_id":  result.get("expert_id", ""),
                    "model":      result.get("model", ""),
                    "discipline": e.get("discipline", ""),
                    "region":     e.get("region", ""),
                    "rationale":  result.get("rationale", ""),
                }
                for c in CLUSTERS:
                    entry = result.get("clusters", {}).get(c, {})
                    row[f"C{c}_p50"]         = entry.get("growth_p50", "")
                    row[f"C{c}_terms_2030"]  = "|".join(
                        entry.get("emerging_terms_2030", []))
                    row[f"C{c}_wildcard"]    = entry.get("wild_card", "")
                writer.writerow(row)

    print(f"\nSaved to {output_dir}/")
    print("  pipeline_summary.json — use forecast_2030 for paper Section 4.5")
    print("  convergence.csv")
    print("  expert_responses.csv")


if __name__ == "__main__":
    if not OPENROUTER_API_KEY:
        print("ERROR: OPENROUTER_API_KEY not set. Check .env file.")
    else:
        run()
