"""
Delphi Ranking Stability Bootstrap Analysis

This script evaluates the stability of Delphi rankings by resampling agent 
responses with replacement (Bootstrap, default n=1000 iterations). 
It applies an aggregated Dirichlet score allocation to compute:
  - Percentages of frequency for each position per factor (FA)
  - 95% Confidence Intervals (CI) for scores
  - Stability rates for Top-1 and Top-2 ranks
  - Auto-export of summaries to CSV for direct academic reporting.
"""

from __future__ import annotations

import json
import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

# ===========================================================================
# CONFIGURACIÓN GLOBAL Y PARÁMETROS METODOLÓGICOS
# ===========================================================================

MAS = ["FA1", "FA2", "FA3", "FA4", "FA5"]

# Escala de peso para transformar la evidencia agregada en pseudo-conteos Dirichlet
WEIGHT_SCALE = 10.0

# ===========================================================================
# NÚCLEO METODOLÓGICO: AGREGACIÓN Y RANKING
# ===========================================================================

def dirichlet_reconciliation(results: list) -> dict:
    """
    Agrega las respuestas usando una distribución Dirichlet aplanada (prior=1.0)
    y escala las evidencias observadas para obtener las puntuaciones relativas.
    """
    alpha = {ma: 1.0 for ma in MAS}
    for r in results:
        for ma in MAS:
            alpha[ma] += r.get(ma, 0.0) * WEIGHT_SCALE
    total = sum(alpha.values())
    return {ma: alpha[ma] / total for ma in MAS}


def compute_ranking(scores: dict) -> list:
    """Ordena los factores (FA) de mayor a menor puntuación."""
    return sorted(MAS, key=lambda x: scores[x], reverse=True)


# ===========================================================================
# PIPELINE DE EJECUCIÓN (BOOTSTRAP Y EXPORTACIÓN)
# ===========================================================================

def run(results_path: str, n_iter: int = 1000, seed: int = 42) -> None:
    # Fijar semillas para asegurar reproducibilidad exacta exigida en revisión
    random.seed(seed)
    np.random.seed(seed)

    path = Path(results_path)
    if not path.exists():
        sys.exit(f"[ERROR] Results file not found: {results_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    agents = [item for item in data if item is not None]
    n = len(agents)
    print(f"Agentes válidos cargados: {n}")

    # Análisis del dataset original completo
    full_scores = dirichlet_reconciliation(agents)
    full_rank = compute_ranking(full_scores)
    print(f"\nRanking completo original: {full_rank}")
    print("Scores originales:", {ma: round(full_scores[ma], 4) for ma in full_rank})

    # Estructuras para almacenar las iteraciones del Bootstrap
    rank_counts = defaultdict(lambda: defaultdict(int))  # rank_counts[FA][pos]
    scores_boot = defaultdict(list)
    top1_stable = 0
    top2_stable = 0
    full_top2 = set(full_rank[:2])

    print(f"\nEjecutando remuestreo Bootstrap ({n_iter} iteraciones)...")
    for _ in range(n_iter):
        sample = random.choices(agents, k=n)
        s = dirichlet_reconciliation(sample)
        r = compute_ranking(s)
        
        for pos, ma in enumerate(r):
            rank_counts[ma][pos] += 1
        for ma in MAS:
            scores_boot[ma].append(s[ma])
            
        if r[0] == full_rank[0]:
            top1_stable += 1
        if set(r[:2]) == full_top2:
            top2_stable += 1

    # -----------------------------------------------------------------------
    # IMPRESIÓN DE RESULTADOS EN CONSOLA Y PREPARACIÓN DE DATOS
    # -----------------------------------------------------------------------
    print(f"\n=== ESTABILIDAD DEL RANKING (Bootstrap n={n_iter}) ===")
    print(f"Top-1 estable ({full_rank[0]}): {top1_stable/n_iter*100:.1f}%")
    print(f"Top-2 estable {full_rank[:2]}: {top2_stable/n_iter*100:.1f}%")

    print(f"\n{'FA':<6} {'mean':>8} {'95% CI':>20}  {'top1%':>7}  {'top2%':>7}")
    
    summary_rows = []
    for ma in full_rank:
        arr = np.array(scores_boot[ma])
        lo, hi = np.percentile(arr, [2.5, 97.5])
        t1 = rank_counts[ma][0] / n_iter * 100
        t2 = (rank_counts[ma][0] + rank_counts[ma][1]) / n_iter * 100
        
        print(f"{ma:<6} {arr.mean():>8.4f} [{lo:.4f}, {hi:.4f}]  {t1:>6.1f}%  {t2:>6.1f}%")
        
        summary_rows.append({
            "FA": ma,
            "mean_score": round(arr.mean(), 4),
            "ci_95_lower": round(lo, 4),
            "ci_95_upper": round(hi, 4),
            "top1_frequency_pct": round(t1, 1),
            "top2_frequency_pct": round(t2, 1)
        })

    # Matriz de posiciones en consola
    print("\n=== MATRIZ DE POSICIONES (% de iteraciones) ===")
    header = "       " + "".join(f"  pos{i+1}" for i in range(len(MAS)))
    print(header)
    
    matrix_rows = []
    for ma in MAS:
        row_str = f"{ma:<6} " + "".join(
            f"  {rank_counts[ma][i]/n_iter*100:>4.1f}%" for i in range(len(MAS))
        )
        print(row_str)
        
        row_dict = {"FA": ma}
        for i in range(len(MAS)):
            row_dict[f"pos_{i+1}_pct"] = round(rank_counts[ma][i] / n_iter * 100, 1)
        matrix_rows.append(row_dict)

    # -----------------------------------------------------------------------
    # EXPORTACIÓN AUTOMÁTICA A CSV (Para anexos y tablas de artículos)
    # -----------------------------------------------------------------------
    output_dir = path.parent
    
    df_summary = pd.DataFrame(summary_rows)
    summary_out = output_dir / "delphi_bootstrap_summary.csv"
    df_summary.to_csv(summary_out, index=False)
    
    df_matrix = pd.DataFrame(matrix_rows)
    matrix_out = output_dir / "delphi_position_matrix.csv"
    df_matrix.to_csv(matrix_out, index=False)
    
    print(f"\n[INFO] Archivos de análisis guardados exitosamente en:\n  - {summary_out}\n  - {matrix_out}")


def parse_args():
    p = argparse.ArgumentParser(description="Bootstrap tool for Delphi ranking stability.")
    p.add_argument("--results", default="food_delphi_results/round_3_results.json")
    p.add_argument("--n-iter", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.results, args.n_iter, args.seed)
