#!/usr/bin/env python3
"""
ce_hybrid_eval.py — Hybrid BM25+Dense RRF retrieval baseline

Tests whether Reciprocal Rank Fusion (BM25 + SBERT dense) closes
the vocabulary gap for AWS-isolated clusters (C10, C11, C14).

RRF formula: score(d) = sum_r 1 / (k + rank_r(d))  where k=60

Usage:
    conda activate aws
    python ce_hybrid_eval.py

Output:
    ce_retrieval_results/hybrid_rrf_table.csv
    ce_retrieval_results/hybrid_rrf_report.txt
"""

import json
import csv
import numpy as np
from pathlib import Path
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────────────
ABSTRACTS_PATH = Path.home() / "Desktop/openalex/ce_abstracts.json"
PAPER_TOPICS   = Path.home() / "Desktop/openalex/results_circular_economy/full_corpus/paper_topics_clean.csv"
TOPICS_PATH    = Path.home() / "Desktop/openalex/results_circular_economy/full_corpus/semantic_topics.csv"
BRIDGE_TERMS   = Path.home() / "Desktop/openalex/bridge_terms.json"
OUT_DIR        = Path.home() / "Desktop/openalex/ce_retrieval_results"

CLUSTERS     = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14]
ISOLATED     = {10, 11, 14}
NON_ISOLATED = {2, 3, 4, 5, 6, 7, 8, 9}
K_VALUES     = [10, 50]
RRF_K        = 60   # standard RRF constant
SBERT_MODEL  = "all-MiniLM-L6-v2"

# ── Loaders (same as ce_retrieval_eval.py) ─────────────────────────────────

def load_abstracts():
    with open(ABSTRACTS_PATH) as f:
        raw = json.load(f)
    def extract(v):
        if isinstance(v, str): return v
        if isinstance(v, dict):
            for k in ("abstract", "text", "content", "body"):
                if v.get(k) and isinstance(v[k], str): return v[k]
        return ""
    if isinstance(raw, list):
        return {d["doi"]: extract(d) for d in raw if isinstance(d, dict) and extract(d)}
    return {k: extract(v) for k, v in raw.items() if extract(v)}


def load_paper_cluster():
    doi_to_cluster = {}
    with open(PAPER_TOPICS) as f:
        for row in csv.DictReader(f):
            doi_to_cluster[row["doi"]] = int(row["cluster"])
    return doi_to_cluster


def load_cluster_terms():
    terms = {}
    with open(TOPICS_PATH) as f:
        for row in csv.DictReader(f):
            c = int(row["cluster"])
            if c in CLUSTERS:
                terms[c] = [t.strip() for t in row["top_terms"].split(";")]
    return terms


def load_bridge_terms_dict():
    if not BRIDGE_TERMS.exists():
        return {}
    with open(BRIDGE_TERMS) as f:
        bt = json.load(f)
    result = {}
    for k, v in bt.items():
        cid = int(k.replace("C", ""))
        terms = list(v.get("distinctive_terms", []))
        for nb in v.get("neighbours", []):
            terms.extend(nb.get("bridge_terms", []))
        result[cid] = list(set(terms))
    return result


def build_corpus(abstracts, doi_to_cluster):
    docs, clusters = [], []
    for doi, cluster in doi_to_cluster.items():
        if cluster not in CLUSTERS:
            continue
        text = abstracts.get(doi, "")
        if text and len(text.split()) >= 10:
            docs.append(text[:512])
            clusters.append(cluster)
    return docs, clusters


def precision_at_k(retrieved_idx, doc_clusters, target_cluster, k):
    top = retrieved_idx[:k]
    hits = sum(1 for i in top if doc_clusters[i] == target_cluster)
    return hits / k


# ── RRF fusion ─────────────────────────────────────────────────────────────

def rrf_fuse(bm25_ranking, dense_ranking, k=60):
    """
    Reciprocal Rank Fusion of two ranked lists.
    Returns merged ranking as list of doc indices.
    """
    scores = defaultdict(float)
    for rank, doc_idx in enumerate(bm25_ranking):
        scores[doc_idx] += 1.0 / (k + rank + 1)
    for rank, doc_idx in enumerate(dense_ranking):
        scores[doc_idx] += 1.0 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda x: -scores[x])


# ── Main evaluation ─────────────────────────────────────────────────────────

def run():
    OUT_DIR.mkdir(exist_ok=True)

    print("Loading data...")
    abstracts      = load_abstracts()
    doi_to_cluster = load_paper_cluster()
    cluster_terms  = load_cluster_terms()
    bridge_terms   = load_bridge_terms_dict()
    docs, doc_clusters = build_corpus(abstracts, doi_to_cluster)
    print(f"Corpus: {len(docs)} documents")

    # ── BM25 index ────────────────────────────────────────────────────────
    from rank_bm25 import BM25Okapi
    print("Building BM25 index...")
    tokenized = [d.lower().split() for d in docs]
    bm25      = BM25Okapi(tokenized)
    print("  BM25 ready")

    # ── SBERT index ───────────────────────────────────────────────────────
    from sentence_transformers import SentenceTransformer
    print(f"Loading SBERT ({SBERT_MODEL})...")
    model       = SentenceTransformer(SBERT_MODEL)
    print("  Encoding corpus...")
    doc_embeddings = model.encode(docs, batch_size=256,
                                  show_progress_bar=True,
                                  convert_to_numpy=True)
    print("  SBERT ready")

    results = []

    for qc in CLUSTERS:
        if qc not in cluster_terms:
            continue

        base_terms = cluster_terms[qc]
        aug_terms  = base_terms + bridge_terms.get(qc, []) if qc in ISOLATED else base_terms

        for condition, terms in [("hybrid_rrf", base_terms),
                                  ("hybrid_rrf_bridge", aug_terms)]:
            if condition == "hybrid_rrf_bridge" and qc not in ISOLATED:
                continue

            query_text = " ".join(terms)

            # BM25 ranking
            bm25_scores  = bm25.get_scores(query_text.lower().split())
            bm25_ranking = list(np.argsort(bm25_scores)[::-1])

            # Dense ranking
            query_emb    = model.encode([query_text], convert_to_numpy=True)
            cos_scores   = np.dot(doc_embeddings, query_emb[0]) / (
                np.linalg.norm(doc_embeddings, axis=1) *
                np.linalg.norm(query_emb[0]) + 1e-9
            )
            dense_ranking = list(np.argsort(cos_scores)[::-1])

            # RRF fusion
            rrf_ranking = rrf_fuse(bm25_ranking, dense_ranking, k=RRF_K)

            for tc in CLUSTERS:
                for k in K_VALUES:
                    prec = precision_at_k(rrf_ranking, doc_clusters, tc, k)
                    results.append({
                        "query_cluster":  qc,
                        "target_cluster": tc,
                        "condition":      condition,
                        "k":              k,
                        "precision_at_k": round(prec, 4),
                        "is_isolated":    tc in ISOLATED,
                    })

        print(f"  C{qc} done")

    # ── Save CSV ──────────────────────────────────────────────────────────
    out_csv = OUT_DIR / "hybrid_rrf_table.csv"
    fields  = ["query_cluster", "target_cluster", "condition", "k",
               "precision_at_k", "is_isolated"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved: {out_csv}")

    # ── Report ────────────────────────────────────────────────────────────
    report = ["CE KG — HYBRID RRF (BM25 + SBERT) RETRIEVAL BASELINE",
              "=" * 60, ""]

    for condition in ["hybrid_rrf", "hybrid_rrf_bridge"]:
        for k in K_VALUES:
            subset = [r for r in results
                      if r["condition"] == condition
                      and int(r["k"]) == k
                      and r["query_cluster"] != r["target_cluster"]]
            if not subset:
                continue

            iso    = [float(r["precision_at_k"]) for r in subset
                      if int(r["target_cluster"]) in ISOLATED]
            noniso = [float(r["precision_at_k"]) for r in subset
                      if int(r["target_cluster"]) in NON_ISOLATED]

            mean_iso    = sum(iso)    / len(iso)    if iso    else 0
            mean_noniso = sum(noniso) / len(noniso) if noniso else 0
            zeros_iso   = sum(1 for v in iso if v == 0.0)

            report += [
                f"{condition.upper()} precision@{k}:",
                f"  Isolated  (C10,C11,C14): mean={mean_iso:.4f}  "
                f"zeros={zeros_iso}/{len(iso)}",
                f"  Non-iso   (C2-C9):       mean={mean_noniso:.4f}",
                f"  Gap: {mean_noniso - mean_iso:.4f}",
                "",
            ]

    report_path = OUT_DIR / "hybrid_rrf_report.txt"
    report_path.write_text("\n".join(report))
    print(f"Saved: {report_path}")

    # ── Print summary ─────────────────────────────────────────────────────
    print("\n" + "\n".join(report))


if __name__ == "__main__":
    run()
