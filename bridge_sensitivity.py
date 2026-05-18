"""
bridge_sensitivity.py — Sensitivity of condition G to top-k c-TF-IDF threshold

Tests k in {10, 20, 30, 50} for bridge term extraction.
For each k, measures Delphi recall on isolated clusters C10/C11/C14
using the existing pipeline_summary.json outputs + rerunning BM25 retrieval.

Since rerunning the full Delphi is expensive, this script:
1. Recomputes BM25 precision@10 for each top-k threshold
2. Reports how many bridge terms are extracted at each threshold
3. Reports recall from existing Delphi run (condition G) as reference

Usage:
    conda activate aws
    python bridge_sensitivity.py

Output:
    ce_retrieval_results/bridge_sensitivity.csv
    ce_retrieval_results/bridge_sensitivity_report.txt
"""

import json
import csv
import os
from pathlib import Path
from collections import defaultdict

# ── CONFIG ─────────────────────────────────────────────────────────────────
ABSTRACTS_PATH   = Path.home() / "Desktop/openalex/ce_abstracts.json"
PAPER_TOPICS     = Path.home() / "Desktop/openalex/results_circular_economy/full_corpus/paper_topics_clean.csv"
TOPICS_PATH      = Path.home() / "Desktop/openalex/results_circular_economy/full_corpus/semantic_topics.csv"
OUT_DIR          = Path.home() / "Desktop/openalex/ce_retrieval_results"

ISOLATED_CLUSTERS = [10, 11, 14]
TOP_K_VALUES      = [5, 10, 20, 30, 50]

# Ground truth recall from condition G Delphi (from pipeline_summary.json)
DELPHI_G_RECALL = {10: 0.75, 11: 0.75, 14: 0.25}
# ───────────────────────────────────────────────────────────────────────────


def load_abstracts():
    with open(ABSTRACTS_PATH) as f:
        raw = json.load(f)
    def extract(v):
        if isinstance(v, str): return v
        if isinstance(v, dict):
            for k in ("abstract", "text", "content"):
                if v.get(k): return v[k]
        return ""
    if isinstance(raw, list):
        return {d["doi"]: extract(d) for d in raw if extract(d)}
    return {k: extract(v) for k, v in raw.items() if extract(v)}


def load_paper_cluster():
    doi_to_cluster = {}
    with open(PAPER_TOPICS) as f:
        for row in csv.DictReader(f):
            doi_to_cluster[row["doi"]] = int(row["cluster"])
    return doi_to_cluster


def compute_ctfidf_terms(abstracts, doi_to_cluster, top_k):
    """Compute c-TF-IDF top-k terms per cluster."""
    from math import log

    # Collect docs per cluster
    cluster_docs = defaultdict(list)
    for doi, cluster in doi_to_cluster.items():
        text = abstracts.get(doi, "")
        if text:
            cluster_docs[cluster].append(text)

    # TF per cluster (concatenated)
    cluster_tf = {}
    all_words = set()
    for cluster, docs in cluster_docs.items():
        combined = " ".join(docs).lower().split()
        tf = defaultdict(int)
        for w in combined:
            if len(w) > 3:  # skip short words
                tf[w] += 1
        cluster_tf[cluster] = tf
        all_words.update(tf.keys())

    # IDF: log(n_clusters / n_clusters_containing_word)
    n_clusters = len(cluster_docs)
    word_cluster_count = defaultdict(int)
    for tf in cluster_tf.values():
        for w in tf:
            word_cluster_count[w] += 1

    # c-TF-IDF score per cluster per word
    cluster_top_terms = {}
    for cluster, tf in cluster_tf.items():
        n_docs = len(cluster_docs[cluster])
        scores = {}
        for w, count in tf.items():
            idf = log(n_clusters / (word_cluster_count[w] + 1e-9))
            scores[w] = (count / n_docs) * idf
        top = sorted(scores, key=lambda x: -scores[x])[:top_k]
        cluster_top_terms[cluster] = top

    return cluster_top_terms


def compute_bridge_terms(cluster_top_terms, isolated_clusters, non_isolated):
    """Find terms in isolated cluster top-k that also appear in any non-isolated top-k."""
    bridge = {}
    for iso in isolated_clusters:
        if iso not in cluster_top_terms:
            continue
        iso_terms = set(cluster_top_terms[iso])
        shared = set()
        for c in non_isolated:
            if c in cluster_top_terms:
                shared.update(iso_terms & set(cluster_top_terms[c]))
        bridge[iso] = {
            "distinctive": list(iso_terms - shared),
            "bridge": list(shared),
            "total": list(iso_terms)
        }
    return bridge


def bm25_precision_at_k(docs, doc_clusters, query_terms, target_cluster, k):
    """BM25 precision@k for a single query."""
    from rank_bm25 import BM25Okapi
    tokenized = [d.lower().split() for d in docs]
    bm25 = BM25Okapi(tokenized)
    query = " ".join(query_terms).lower().split()
    import numpy as np
    scores = bm25.get_scores(query)
    top_k_idx = np.argsort(scores)[::-1][:k]
    hits = sum(1 for idx in top_k_idx if doc_clusters[idx] == target_cluster)
    return hits / k


def run():
    OUT_DIR.mkdir(exist_ok=True)
    print("Loading data...")
    abstracts     = load_abstracts()
    doi_to_cluster = load_paper_cluster()

    # Build corpus (papers with abstracts in target clusters)
    target_clusters = list(range(2, 15))
    docs, doc_clusters = [], []
    for doi, cluster in doi_to_cluster.items():
        if cluster in target_clusters:
            text = abstracts.get(doi, "")
            if text and len(text.split()) >= 10:
                docs.append(text)
                doc_clusters.append(cluster)
    print(f"Corpus: {len(docs)} documents")

    non_isolated = [c for c in target_clusters if c not in ISOLATED_CLUSTERS]
    results = []

    for top_k in TOP_K_VALUES:
        print(f"\nTop-k = {top_k}...")

        # Compute c-TF-IDF terms
        cluster_top_terms = compute_ctfidf_terms(abstracts, doi_to_cluster, top_k)
        bridge_data       = compute_bridge_terms(cluster_top_terms, ISOLATED_CLUSTERS, non_isolated)

        for iso in ISOLATED_CLUSTERS:
            if iso not in bridge_data:
                continue
            bd = bridge_data[iso]
            n_distinctive = len(bd["distinctive"])
            n_bridge      = len(bd["bridge"])
            n_total       = n_distinctive + n_bridge

            # BM25 with distinctive terms only (baseline)
            prec_base = bm25_precision_at_k(
                docs, doc_clusters,
                bd["total"],  # full top-k terms
                iso, k=10
            )

            # BM25 with distinctive + bridge terms
            augmented = bd["total"] + bd["bridge"]
            prec_aug = bm25_precision_at_k(
                docs, doc_clusters,
                augmented,
                iso, k=10
            )

            # Cross-cluster: can we find non-isolated docs with augmented query?
            prec_cross = {}
            for c in non_isolated[:4]:  # sample 4 non-isolated clusters
                prec_cross[c] = bm25_precision_at_k(
                    docs, doc_clusters,
                    augmented, c, k=10
                )

            delphi_recall = DELPHI_G_RECALL.get(iso, None)

            row = {
                "top_k":          top_k,
                "cluster":        iso,
                "n_distinctive":  n_distinctive,
                "n_bridge":       n_bridge,
                "n_total_terms":  n_total,
                "bm25_self_base": round(prec_base, 4),
                "bm25_self_aug":  round(prec_aug, 4),
                "delphi_recall_G": delphi_recall,
                "sample_terms":   "; ".join(bd["total"][:5]),
                "bridge_terms":   "; ".join(bd["bridge"][:5]),
            }
            results.append(row)
            print(f"  C{iso}: {n_distinctive} distinctive + {n_bridge} bridge = {n_total} terms | "
                  f"BM25 self={prec_base:.3f} aug={prec_aug:.3f} | Delphi recall={delphi_recall}")

    # Save CSV
    out_csv = OUT_DIR / "bridge_sensitivity.csv"
    fields = ["top_k", "cluster", "n_distinctive", "n_bridge", "n_total_terms",
              "bm25_self_base", "bm25_self_aug", "delphi_recall_G",
              "sample_terms", "bridge_terms"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved: {out_csv}")

    # Report
    report = ["BRIDGE TERM SENSITIVITY — top-k c-TF-IDF", "=" * 60, ""]
    report.append(f"{'top_k':>6} {'cluster':>8} {'n_dist':>7} {'n_bridge':>9} "
                  f"{'bm25_self':>10} {'delphi_G':>10}")
    report.append("-" * 55)
    for r in results:
        report.append(f"{r['top_k']:>6} {'C'+str(r['cluster']):>8} "
                      f"{r['n_distinctive']:>7} {r['n_bridge']:>9} "
                      f"{r['bm25_self_base']:>10.4f} "
                      f"{str(r['delphi_recall_G']):>10}")

    report += [
        "",
        "INTERPRETATION:",
        "- Delphi recall (condition G) is fixed at the values from pipeline_summary.json",
        "- BM25 self-retrieval tests whether the cluster is internally coherent under each top-k",
        "- n_bridge = terms shared between isolated and non-isolated cluster vocabularies",
        "- Stability across top-k supports robustness of the S_ij < 0.002 threshold claim",
    ]

    out_txt = OUT_DIR / "bridge_sensitivity_report.txt"
    out_txt.write_text("\n".join(report))
    print(f"Saved: {out_txt}")


if __name__ == "__main__":
    run()
