"""
ce_retrieval_eval.py — Document-level retrieval evaluation for DMKG 2026 paper

Answers reviewer question: "Can you provide recall@k / nDCG independent of LLM generation?"

For each cluster, uses its c-TF-IDF top terms as a query and measures how many
documents from each target cluster appear in the top-k retrieved results.

Conditions evaluated:
  - BM25 (lexical baseline)
  - Dense (SBERT cosine similarity)
  - Bridge-augmented BM25 (adds bridge terms for isolated clusters C10, C11, C14)

Usage:
    conda activate aws
    pip install rank_bm25 sentence-transformers --break-system-packages
    python ce_retrieval_eval.py
"""

import json
import csv
import os
import math
import numpy as np
from pathlib import Path
from collections import defaultdict

# ── CONFIG ────────────────────────────────────────────────────────────────────
ABSTRACTS_PATH  = Path.home() / "Desktop/openalex/ce_abstracts.json"
TOPICS_PATH     = Path.home() / "Desktop/openalex/results_circular_economy/full_corpus/semantic_topics.csv"
PAPER_TOPICS    = Path.home() / "Desktop/openalex/results_circular_economy/full_corpus/paper_topics_clean.csv"
BRIDGE_TERMS    = Path.home() / "Desktop/openalex/bridge_terms.json"
OUT_DIR         = Path.home() / "Desktop/openalex/ce_retrieval_results"

CLUSTERS_OF_INTEREST = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14]
ISOLATED_CLUSTERS    = [10, 11, 14]
K_VALUES             = [10, 50]
# ─────────────────────────────────────────────────────────────────────────────


def load_data():
    print("Loading abstracts...")
    with open(ABSTRACTS_PATH) as f:
        abstracts_raw = json.load(f)

    def extract_text(v):
        if isinstance(v, str):
            return v
        if isinstance(v, dict):
            for key in ("abstract", "text", "content", "body", "abstract_text"):
                if v.get(key) and isinstance(v[key], str):
                    return v[key]
        return ""

    if isinstance(abstracts_raw, list):
        abstracts = {}
        for d in abstracts_raw:
            if not isinstance(d, dict):
                continue
            doi = d.get("doi", "")
            text = extract_text(d)
            if doi and text:
                abstracts[doi] = text
    else:
        abstracts = {}
        for k, v in abstracts_raw.items():
            text = extract_text(v)
            if text:
                abstracts[k] = text

    print(f"  {len(abstracts)} abstracts loaded")

    print("Loading paper-cluster assignments...")
    doi_to_cluster = {}
    with open(PAPER_TOPICS) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cluster = int(row["cluster"])
            if cluster in CLUSTERS_OF_INTEREST:
                doi_to_cluster[row["doi"]] = cluster
    print(f"  {len(doi_to_cluster)} papers in target clusters")

    print("Loading cluster top terms...")
    cluster_terms = {}
    with open(TOPICS_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cluster = int(row["cluster"])
            if cluster in CLUSTERS_OF_INTEREST:
                terms = [t.strip() for t in row["top_terms"].split(";")]
                cluster_terms[cluster] = terms
    print(f"  {len(cluster_terms)} clusters with terms")

    # ── BRIDGE TERMS — fixed to match actual JSON structure ──────────────────
    # Actual structure:
    #   { "C10": { "distinctive_terms": [...], "neighbours": {"C4": [...], "C6": [...]} }, ... }
    # We extract: distinctive_terms + all terms from all neighbour clusters
    bridge_terms = {}
    if BRIDGE_TERMS.exists():
        with open(BRIDGE_TERMS) as f:
            bt = json.load(f)

        print(f"\n  [DEBUG] bridge_terms.json top-level keys: {list(bt.keys())}")
        first_key = list(bt.keys())[0]
        first_val = bt[first_key]
        print(f"  [DEBUG] Keys inside '{first_key}': {list(first_val.keys()) if isinstance(first_val, dict) else type(first_val)}")

        for k, v in bt.items():
            # Parse cluster id: "C10" → 10, "10" → 10
            try:
                cluster_id = int(k.replace("C", "").replace("c", ""))
            except ValueError:
                print(f"  WARNING: can't parse cluster key '{k}', skipping")
                continue

            if not isinstance(v, dict):
                print(f"  WARNING: value for '{k}' is {type(v)}, expected dict, skipping")
                continue

            # distinctive_terms is a flat list of strings
            distinctive = v.get("distinctive_terms", [])
            if not isinstance(distinctive, list):
                distinctive = []

            # neighbours is a LIST of dicts:
            # [{"cluster": "C2", "bridge_terms": ["term1", ...], "neigh_top5": [...]}, ...]
            # We extract bridge_terms from each neighbour dict
            neighbours_raw = v.get("neighbours", [])
            neighbour_terms = []
            if isinstance(neighbours_raw, list):
                for neigh in neighbours_raw:
                    if isinstance(neigh, dict):
                        # Primary: bridge_terms specific to this pair
                        neighbour_terms.extend(neigh.get("bridge_terms", []))
                    elif isinstance(neigh, str):
                        neighbour_terms.append(neigh)
            elif isinstance(neighbours_raw, dict):
                # fallback for old format
                for terms in neighbours_raw.values():
                    if isinstance(terms, list):
                        neighbour_terms.extend(terms)

            all_bridge = distinctive + neighbour_terms

            # Deduplicate while preserving order
            seen = set()
            deduped = []
            for t in all_bridge:
                if isinstance(t, str) and t.lower() not in seen:
                    seen.add(t.lower())
                    deduped.append(t)

            bridge_terms[cluster_id] = deduped
            print(f"  C{cluster_id}: {len(distinctive)} distinctive + {len(neighbour_terms)} neighbour terms = {len(deduped)} bridge terms")

        loaded_clusters = sorted(bridge_terms.keys())
        print(f"  Bridge terms loaded for clusters: {loaded_clusters}")
        for c in ISOLATED_CLUSTERS:
            if c in bridge_terms:
                print(f"    C{c} sample: {bridge_terms[c][:5]}")
            else:
                print(f"    WARNING: C{c} (isolated) has NO bridge terms!")
    else:
        print(f"  WARNING: {BRIDGE_TERMS} not found — bridge condition will be skipped")

    return abstracts, doi_to_cluster, cluster_terms, bridge_terms


def build_corpus(abstracts, doi_to_cluster):
    docs, dois, clusters = [], [], []
    for doi, cluster in doi_to_cluster.items():
        text = abstracts.get(doi, "")
        if text and len(text.split()) >= 10:
            docs.append(text)
            dois.append(doi)
            clusters.append(cluster)
    print(f"Corpus built: {len(docs)} documents")
    return docs, dois, clusters


def build_bm25_index(docs):
    from rank_bm25 import BM25Okapi
    print("Building BM25 index...")
    tokenized = [d.lower().split() for d in docs]
    bm25 = BM25Okapi(tokenized)
    print(f"  Index built over {len(docs)} documents")
    return bm25


def bm25_retrieve(bm25_index, query_terms, k):
    query = " ".join(query_terms).lower().split()
    scores = bm25_index.get_scores(query)
    top_k = np.argsort(scores)[::-1][:k]
    return top_k, scores


def dense_retrieve(doc_embs, query_terms, k, model):
    query_text = " ".join(query_terms)
    query_emb  = model.encode([query_text], normalize_embeddings=True)
    scores = (doc_embs @ query_emb.T).squeeze()
    top_k  = np.argsort(scores)[::-1][:k]
    return top_k, scores


def precision_at_k(retrieved_indices, doc_clusters, target_cluster, k):
    """Precision@k: fraction of top-k retrieved docs belonging to target cluster."""
    retrieved = retrieved_indices[:k]
    if k == 0:
        return 0.0
    n_hits = sum(1 for idx in retrieved if doc_clusters[idx] == target_cluster)
    return n_hits / k


def ndcg_at_k(retrieved_indices, doc_clusters, target_cluster, k):
    retrieved = retrieved_indices[:k]
    relevances = [1 if doc_clusters[idx] == target_cluster else 0
                  for idx in retrieved]
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(relevances))
    n_rel = sum(1 for c in doc_clusters if c == target_cluster)
    ideal = [1] * min(n_rel, k) + [0] * max(0, k - n_rel)
    idcg  = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


# S_ij hardcoded from circular_economy_aws_v2_pairwise.csv
_SIJ_DATA = [
    (2,3,0.006728),(2,4,0.015775),(2,5,0.004983),(2,6,0.019300),(2,7,0.003239),
    (2,8,0.006525),(2,9,0.005414),(2,10,0.000669),(2,11,0.000263),(2,14,0.001398),
    (3,4,0.022371),(3,5,0.007067),(3,6,0.027370),(3,7,0.004593),(3,8,0.009253),
    (3,9,0.007677),(3,10,0.000949),(3,11,0.000373),(3,14,0.001983),
    (4,5,0.016569),(4,6,0.064171),(4,7,0.010768),(4,8,0.021695),(4,9,0.018000),
    (4,10,0.002225),(4,11,0.000874),(4,14,0.004649),
    (5,6,0.020272),(5,7,0.003402),(5,8,0.006854),(5,9,0.005686),
    (5,10,0.000703),(5,11,0.000276),(5,14,0.001469),
    (6,7,0.013175),(6,8,0.026544),(6,9,0.022022),(6,10,0.002722),
    (6,11,0.001070),(6,14,0.005688),
    (7,8,0.004454),(7,9,0.003695),(7,10,0.000457),(7,11,0.000179),(7,14,0.000954),
    (8,9,0.007445),(8,10,0.000920),(8,11,0.000362),(8,14,0.001923),
    (9,10,0.000764),(9,11,0.000300),(9,14,0.001595),
    (10,11,0.000037),(10,14,0.000197),
    (11,14,0.000077),
]

def load_sij():
    sij = {}
    for ca, cb, val in _SIJ_DATA:
        sij[(ca, cb)] = val
        sij[(cb, ca)] = val
    print(f"S_ij loaded: {len(sij)//2} cluster pairs (hardcoded)")
    return sij


def run_evaluation():
    OUT_DIR.mkdir(exist_ok=True)

    abstracts, doi_to_cluster, cluster_terms, bridge_terms = load_data()
    docs, dois, doc_clusters = build_corpus(abstracts, doi_to_cluster)
    sij_data = load_sij()

    bm25_index = build_bm25_index(docs)

    try:
        from sentence_transformers import SentenceTransformer
        print("Loading SBERT model...")
        sbert = SentenceTransformer("all-MiniLM-L6-v2")
        CACHE_FILE = OUT_DIR / "doc_embeddings.npy"
        if CACHE_FILE.exists():
            print("  Loading cached embeddings (skipping re-encode)...")
            doc_embs = np.load(str(CACHE_FILE))
            print(f"  Loaded: {doc_embs.shape}")
        else:
            print("  Encoding all documents (first time — will cache)...")
            doc_embs = sbert.encode(docs, normalize_embeddings=True,
                                    batch_size=64, show_progress_bar=True)
            np.save(str(CACHE_FILE), doc_embs)
            print(f"  Embeddings cached → {CACHE_FILE}")
        use_dense = True
        print("  SBERT ready")
    except ImportError:
        print("sentence-transformers not available — dense retrieval skipped")
        use_dense = False
        sbert = None
        doc_embs = None

    results = []

    for query_cluster in CLUSTERS_OF_INTEREST:
        if query_cluster not in cluster_terms:
            continue

        base_terms = cluster_terms[query_cluster]
        print(f"\nQuery cluster C{query_cluster}: {base_terms[:5]}...")

        # Build augmented terms for bridge condition
        aug_terms = list(base_terms)
        if query_cluster in bridge_terms and bridge_terms[query_cluster]:
            aug_terms = aug_terms + bridge_terms[query_cluster]
            print(f"  Bridge augmented: +{len(bridge_terms[query_cluster])} terms → {len(aug_terms)} total")
        else:
            print(f"  No bridge terms for C{query_cluster}")

        conditions = [("bm25", base_terms), ("bm25_bridge", aug_terms)]
        if use_dense:
            conditions.append(("dense", base_terms))

        for condition, terms in conditions:
            # Only run bridge condition for isolated clusters
            if condition == "bm25_bridge" and query_cluster not in ISOLATED_CLUSTERS:
                continue

            if condition == "dense":
                top_idx, scores = dense_retrieve(doc_embs, terms, max(K_VALUES), sbert)
            else:
                top_idx, scores = bm25_retrieve(bm25_index, terms, max(K_VALUES))

            for target_cluster in CLUSTERS_OF_INTEREST:
                for k in K_VALUES:
                    prec = precision_at_k(top_idx, doc_clusters, target_cluster, k)
                    ndcg = ndcg_at_k(top_idx, doc_clusters, target_cluster, k) if k == 10 else None

                    is_isolated_target = target_cluster in ISOLATED_CLUSTERS
                    sij_val = sij_data.get((query_cluster, target_cluster), None)

                    results.append({
                        "query_cluster":   query_cluster,
                        "target_cluster":  target_cluster,
                        "condition":       condition,
                        "k":               k,
                        "precision_at_k":  round(prec, 4),
                        "ndcg_at_k":       round(ndcg, 4) if ndcg is not None else "",
                        "is_isolated":     is_isolated_target,
                        "S_ij":            round(sij_val, 6) if sij_val is not None else "",
                    })

    # Save full results
    fields = ["query_cluster", "target_cluster", "condition", "k",
              "precision_at_k", "ndcg_at_k", "is_isolated", "S_ij"]
    out_csv = OUT_DIR / "retrieval_table.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved: {out_csv}")

    # S_ij vs precision@50 scatter
    sij_prec = [(r["S_ij"], r["precision_at_k"])
                for r in results
                if r["condition"] == "bm25"
                and r["k"] == 50
                and r["S_ij"] != ""
                and r["query_cluster"] != r["target_cluster"]]

    sij_csv = OUT_DIR / "sij_vs_precision.csv"
    with open(sij_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["S_ij", "precision_at_50_bm25"])
        w.writeheader()
        for s, p in sij_prec:
            w.writerow({"S_ij": s, "precision_at_50_bm25": p})
    print(f"Saved: {sij_csv}")

    write_report(results, OUT_DIR / "retrieval_report.txt")


def write_report(results, out_path):
    lines = [
        "CE KNOWLEDGE GRAPH — DOCUMENT-LEVEL RETRIEVAL EVALUATION",
        "=" * 60,
        "",
        "Metric: precision@k = (docs from target cluster in top-k) / k",
        "Query: c-TF-IDF top terms for each source cluster",
        "",
    ]

    for condition in ["bm25", "dense", "bm25_bridge"]:
        cond_results = [r for r in results if r["condition"] == condition and r["k"] == 10]
        if not cond_results:
            continue

        isolated     = [r for r in cond_results if r["is_isolated"]
                        and r["query_cluster"] != r["target_cluster"]]
        non_isolated = [r for r in cond_results if not r["is_isolated"]
                        and r["query_cluster"] != r["target_cluster"]]

        mean_iso     = sum(r["precision_at_k"] for r in isolated)     / len(isolated)     if isolated     else float("nan")
        mean_non_iso = sum(r["precision_at_k"] for r in non_isolated) / len(non_isolated) if non_isolated else float("nan")

        lines += [
            f"Condition: {condition.upper()} — precision@10",
            f"  AWS-isolated clusters (C10, C11, C14): mean precision = {mean_iso:.4f}",
            f"  Non-isolated clusters (C2-C9):         mean precision = {mean_non_iso:.4f}",
            f"  Gap: {mean_non_iso - mean_iso:.4f}",
            "",
        ]

    # Bridge vs baseline for isolated clusters
    lines += ["Bridge augmentation effect (isolated clusters only):", "-" * 40]
    for c in ISOLATED_CLUSTERS:
        for k in K_VALUES:
            base = [r["precision_at_k"] for r in results
                    if r["query_cluster"] == c and r["condition"] == "bm25" and r["k"] == k]
            bridge = [r["precision_at_k"] for r in results
                      if r["query_cluster"] == c and r["condition"] == "bm25_bridge" and r["k"] == k]
            if base and bridge:
                b_mean = sum(base) / len(base)
                br_mean = sum(bridge) / len(bridge)
                lines.append(f"  C{c} @{k}: BM25={b_mean:.4f} → BM25+bridge={br_mean:.4f} (Δ={br_mean-b_mean:+.4f})")

    out_path.write_text("\n".join(lines))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    run_evaluation()