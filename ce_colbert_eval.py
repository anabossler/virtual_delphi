"""
ce_colbert_eval.py — ColBERT late-interaction retrieval evaluation

Uses PyLate (https://github.com/lightonai/pylate) with ColBERT v2 model.
Compares ColBERT vs BM25 vs Dense on CE KG cross-cluster retrieval.

Install:
    pip install pylate --break-system-packages

Usage:
    conda activate aws
    python ce_colbert_eval.py

Output:
    ce_retrieval_results/colbert_table.csv
    ce_retrieval_results/colbert_report.txt
"""

import json
import csv
import os
from pathlib import Path
from collections import defaultdict
import numpy as np

# ── CONFIG ─────────────────────────────────────────────────────────────────
ABSTRACTS_PATH = Path.home() / "Desktop/openalex/ce_abstracts.json"
PAPER_TOPICS   = Path.home() / "Desktop/openalex/results_circular_economy/full_corpus/paper_topics_clean.csv"
TOPICS_PATH    = Path.home() / "Desktop/openalex/results_circular_economy/full_corpus/semantic_topics.csv"
BRIDGE_TERMS   = Path.home() / "Desktop/openalex/bridge_terms.json"
OUT_DIR        = Path.home() / "Desktop/openalex/ce_retrieval_results"

CLUSTERS        = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14]
ISOLATED        = {10, 11, 14}
NON_ISOLATED    = {2, 3, 4, 5, 6, 7, 8, 9}
K_VALUES        = [10, 50]
COLBERT_MODEL   = "colbert-ir/colbertv2.0"
MAX_DOC_LEN     = 180   # tokens — ColBERT default
MAX_QUERY_LEN   = 32
# ───────────────────────────────────────────────────────────────────────────


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


def load_bridge_terms():
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
            # Truncate to ~512 chars for ColBERT efficiency on CPU
            docs.append(text[:512])
            clusters.append(cluster)
    return docs, clusters


def precision_at_k(retrieved_idx, doc_clusters, target_cluster, k):
    top = retrieved_idx[:k]
    hits = sum(1 for i in top if doc_clusters[i] == target_cluster)
    return hits / k


def run_bm25(docs, doc_clusters, cluster_terms, bridge_terms, k_values):
    """BM25 baseline — built once, queried per cluster."""
    from rank_bm25 import BM25Okapi
    print("Building BM25 index...")
    tokenized = [d.lower().split() for d in docs]
    bm25 = BM25Okapi(tokenized)
    print(f"  Index built over {len(docs)} docs")

    results = []
    for qc in CLUSTERS:
        if qc not in cluster_terms:
            continue
        base_terms = cluster_terms[qc]
        aug_terms  = base_terms + bridge_terms.get(qc, []) if qc in ISOLATED else base_terms

        for condition, terms in [("bm25", base_terms), ("bm25_bridge", aug_terms)]:
            if condition == "bm25_bridge" and qc not in ISOLATED:
                continue
            scores  = bm25.get_scores(" ".join(terms).lower().split())
            top_idx = np.argsort(scores)[::-1]

            for tc in CLUSTERS:
                for k in k_values:
                    prec = precision_at_k(top_idx, doc_clusters, tc, k)
                    results.append({
                        "query_cluster":  qc,
                        "target_cluster": tc,
                        "condition":      condition,
                        "k":              k,
                        "precision_at_k": round(prec, 4),
                        "is_isolated":    tc in ISOLATED,
                    })
    return results


def run_colbert(docs, doc_clusters, cluster_terms, bridge_terms, k_values):
    """ColBERT late-interaction retrieval via PyLate."""
    try:
        from pylate import indexes, models, retrieve
    except ImportError:
        print("PyLate not installed. Run: pip install pylate --break-system-packages")
        return []

    print(f"Loading ColBERT model: {COLBERT_MODEL}")
    model = models.ColBERT(model_name_or_path=COLBERT_MODEL)

    # Index all documents
    print(f"Encoding {len(docs)} documents (this will take a while on CPU)...")
    index_path = OUT_DIR / "colbert_index"
    index_path.mkdir(exist_ok=True)

    # PyLate PLAID index
    index = indexes.Voyager(
        index_folder=str(index_path),
        index_name="ce_corpus",
        override=False,
    )

    # Encode docs in batches
    batch_size = 32
    doc_embeddings = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        embs  = model.encode(batch, is_query=False, show_progress_bar=False)
        doc_embeddings.extend(embs)
        if (i // batch_size) % 10 == 0:
            print(f"  Encoded {min(i+batch_size, len(docs))}/{len(docs)} docs")

    index.add_documents(
        documents_ids=list(range(len(docs))),
        documents_embeddings=doc_embeddings,
    )
    retriever = retrieve.ColBERT(index=index)

    results = []
    for qc in CLUSTERS:
        if qc not in cluster_terms:
            continue
        base_terms = cluster_terms[qc]
        aug_terms  = base_terms + bridge_terms.get(qc, []) if qc in ISOLATED else base_terms

        for condition, terms in [("colbert", base_terms), ("colbert_bridge", aug_terms)]:
            if condition == "colbert_bridge" and qc not in ISOLATED:
                continue

            query_text = " ".join(terms)
            query_emb  = model.encode([query_text], is_query=True, show_progress_bar=False)
            top_results = retriever.retrieve(
                queries_embeddings=query_emb,
                k=max(k_values),
            )
            # top_results is list of lists of (doc_id, score)
            # PyLate returns list of dicts with "id" and "score" keys
            top_idx = [int(r["id"]) for r in top_results[0]]

            for tc in CLUSTERS:
                for k in k_values:
                    prec = precision_at_k(top_idx, doc_clusters, tc, k)
                    results.append({
                        "query_cluster":  qc,
                        "target_cluster": tc,
                        "condition":      condition,
                        "k":              k,
                        "precision_at_k": round(prec, 4),
                        "is_isolated":    tc in ISOLATED,
                    })
        print(f"  C{qc} done")

    return results


def write_report(all_results, out_path):
    ISOLATED_L = list(ISOLATED)
    NON_ISO_L  = list(NON_ISOLATED)

    lines = ["CE KG — COLBERT vs BM25 RETRIEVAL COMPARISON", "=" * 60, ""]

    for condition in ["bm25", "colbert", "bm25_bridge", "colbert_bridge"]:
        for k in [10, 50]:
            subset = [r for r in all_results
                      if r["condition"] == condition
                      and int(r["k"]) == k
                      and r["query_cluster"] != r["target_cluster"]]
            if not subset:
                continue
            iso    = [float(r["precision_at_k"]) for r in subset if int(r["target_cluster"]) in ISOLATED]
            noniso = [float(r["precision_at_k"]) for r in subset if int(r["target_cluster"]) in NON_ISOLATED]
            mean_iso    = sum(iso)/len(iso) if iso else 0
            mean_noniso = sum(noniso)/len(noniso) if noniso else 0
            zeros_iso   = sum(1 for v in iso if v == 0.0)
            lines += [
                f"{condition.upper()} precision@{k}:",
                f"  Isolated  (C10,C11,C14): mean={mean_iso:.4f}  zeros={zeros_iso}/{len(iso)}",
                f"  Non-iso   (C2-C9):       mean={mean_noniso:.4f}",
                f"  Gap: {mean_noniso - mean_iso:.4f}",
                "",
            ]

    out_path.write_text("\n".join(lines))


def run():
    OUT_DIR.mkdir(exist_ok=True)
    print("Loading data...")
    abstracts      = load_abstracts()
    doi_to_cluster = load_paper_cluster()
    cluster_terms  = load_cluster_terms()
    bridge_terms   = load_bridge_terms()
    docs, doc_clusters = build_corpus(abstracts, doi_to_cluster)
    print(f"Corpus: {len(docs)} documents")

    all_results = []

    # BM25 baseline
    all_results += run_bm25(docs, doc_clusters, cluster_terms, bridge_terms, K_VALUES)
    print(f"BM25 done: {len(all_results)} rows")

    # ColBERT
    colbert_results = run_colbert(docs, doc_clusters, cluster_terms, bridge_terms, K_VALUES)
    all_results += colbert_results
    print(f"ColBERT done: {len(colbert_results)} rows")

    # Save
    out_csv = OUT_DIR / "colbert_table.csv"
    fields  = ["query_cluster", "target_cluster", "condition", "k",
               "precision_at_k", "is_isolated"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_results)
    print(f"Saved: {out_csv}")

    write_report(all_results, OUT_DIR / "colbert_report.txt")
    print(f"Saved: {OUT_DIR / 'colbert_report.txt'}")


if __name__ == "__main__":
    run()