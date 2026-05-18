# virtual_delphi

---

## What this repo contains

| File | Description |
|------|-------------|
| `ce_delphi_v3.py` | Main Delphi pipeline — 8 conditions (A–H), 60 agents, 3 rounds |
| `ce_delphi_condition_I.py` | Prospective horizon 2030–2040 panel (condition I) |
| `ce_delphi_stats.py` | Permutation-based Spearman + bootstrap CI per condition |
| `ce_retrieval_eval.py` | BM25 + SBERT retrieval evaluation, precision@k |
| `ce_colbert_eval.py` | ColBERT v2 via PyLate, precision@k |
| `bridge_sensitivity.py` | Bridge-term sensitivity to top-k ∈ {5,10,20,30,50} |

---

## Setup

```bash
conda activate aws
# or
source .venv/bin/activate

pip install -r requirements.txt
```

Required environment variables in `.env`:
```
OPENROUTER_API_KEY=your_key_here
```

---

## Data

The CE knowledge graph (16,256 publications, 1996–2020) is built from
[OpenAlex](https://openalex.org). Required files under `~/Desktop/openalex/`:

| File | Description |
|------|-------------|
| `bridge_terms.json` | c-TF-IDF top-50 + bridge terms for isolated clusters C10, C11, C14 |
| `bridge_authors.json` | Bilingual vocabulary of bridge authors per isolated cluster |
| `weak_signals.json` | Pre-extracted policy signals for condition F |
| `results_circular_economy/full_corpus/paper_topics_clean.csv` | 20,066 CE papers with cluster assignments |
| `ce_abstracts.json` | 9,779 indexed abstracts |
| `cluster_fulltexts/` | Full-text corpus per cluster (pre-2020) |
| `cluster_news/` | Grey literature corpus per cluster |

---

## Running the pipeline

```bash
# Condition A (no KG baseline)
python ce_delphi_v3.py --condition A

# Condition G (bridge-term injection — best recall)
python ce_delphi_v3.py --condition G

# Condition H (bridge-author Graph RAG — best rho-recall balance)
python ce_delphi_v3.py --condition H

# Resume from saved rounds
python ce_delphi_v3.py --condition G --analysis-only

# Prospective 2030-2040 horizon (requires condition G results)
python ce_delphi_condition_I.py

# Retrieval evaluation (BM25 + SBERT)
python ce_retrieval_eval.py

# ColBERT v2 evaluation
python ce_colbert_eval.py

# Bridge-term sensitivity
python bridge_sensitivity.py

# Statistics table for paper
python ce_delphi_stats.py
```

---

## Key results

| Condition | Strategy | N valid | ρ [95% CI] | Recall |
|-----------|----------|---------|------------|--------|
| A | No KG (baseline) | 44 | 0.036 [−0.617, 0.766] | 0.288 |
| B | Full-text (1500c) | 47 | 0.491 [−0.237, 0.923] | 0.311 |
| D | Abstract (300c) | 48 | 0.536 [−0.211, 0.944] | 0.265 |
| E | Minimal context | 53 | 0.273 [−0.589, 0.869] | 0.402 |
| F | Synthesised signals | 54 | 0.227 [−0.514, 0.888] | 0.288 |
| **G** | **Bridge terms (KG vocab)** | **56** | **0.064 [−0.596, 0.649]** | **0.492** |
| **H** | **Bridge authors (KG graph)** | **52** | **0.409 [−0.349, 0.841]** | **0.379** |

Isolated clusters (C10, C11, C14): recall = 0.0 under all conditions A–F.
Retriever-agnostic: BM25, SBERT, ColBERT v2 all produce 90% zero-precision
for isolated-cluster targets (27/30 cases).

---

## Exposure analysis (condition G)

Ground-truth terms present in pre-2020 c-TF-IDF top-50 injected under G:

| Cluster | GT terms exposed | GT set |
|---------|-----------------|--------|
| C10 | 3/4 | tensile strength, mechanical properties, thermal stability ✓; additive manufacturing ✗ |
| C11 | 4/4 | all four terms present in pre-2020 vocabulary |
| C14 | 0/4 | none exposed — recall gain (0→0.25) is exposure-free |

Note: condition E (same documents, no bridge injection) produces recall = 0.0
for C10 and C11, confirming the bridge vocabulary activates correct
term generation for the right clusters.

---

## License

MIT License — see `LICENSE`.
