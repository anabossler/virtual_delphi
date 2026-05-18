"""
CE-Delphi: Permutation-based Spearman + Bootstrap CI
Auto-discovers all condition_X folders under RESULTS_DIR.

Usage:
    python ce_delphi_stats.py

Output:
    ce_delphi_stats.csv   -- per-condition table (rho, CI, permutation p)
    ce_delphi_stats.tex   -- LaTeX table ready for paper
"""

import json
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

# ── CONFIG ────────────────────────────────────────────────────────────────────
RESULTS_DIR    = Path.home() / "Desktop/openalex/ce_delphi_v3_results"
N_PERMUTATIONS = 10_000
BOOTSTRAP_N    = 10_000
CI_LEVEL       = 0.95
RNG_SEED       = 42
# ─────────────────────────────────────────────────────────────────────────────


def load_summary(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def extract_ranks(summary: dict) -> tuple[list, list]:
    sp   = summary.get("spearman", {})
    pred = sp.get("ranked_pred") or sp.get("predicted_rank")
    obs  = sp.get("ranked_obs")  or sp.get("observed_rank")
    if pred is None or obs is None:
        raise KeyError(f"Cannot find ranked_pred/ranked_obs in condition {summary.get('condition')}")
    return pred, obs


def ranks_to_numeric(pred: list, obs: list) -> tuple[np.ndarray, np.ndarray]:
    pos_obs  = np.arange(1, len(obs) + 1, dtype=float)
    pos_pred = np.array([pred.index(c) + 1 for c in obs], dtype=float)
    return pos_pred, pos_obs


def permutation_spearman(x, y, n_perm, rng) -> tuple[float, float]:
    rho_obs, _ = spearmanr(x, y)
    null = np.array([spearmanr(rng.permutation(x), y).statistic for _ in range(n_perm)])
    p_perm = np.mean(np.abs(null) >= abs(rho_obs))
    return float(rho_obs), float(p_perm)


def bootstrap_rho(x, y, n_boot, rng) -> tuple[float, float]:
    n    = len(x)
    boot = np.array([
        spearmanr(x[idx := rng.choice(n, size=n, replace=True)], y[idx]).statistic
        for _ in range(n_boot)
    ])
    lo, hi = np.percentile(boot, [(1 - CI_LEVEL) / 2 * 100, (1 + CI_LEVEL) / 2 * 100])
    return float(lo), float(hi)


def bootstrap_recall(per_cluster: dict, n_boot, rng) -> tuple[float, float, float]:
    recalls = np.array([v["recall"] for v in per_cluster.values()], dtype=float)
    macro   = recalls.mean()
    boot    = np.array([rng.choice(recalls, size=len(recalls), replace=True).mean()
                        for _ in range(n_boot)])
    lo, hi  = np.percentile(boot, [(1 - CI_LEVEL) / 2 * 100, (1 + CI_LEVEL) / 2 * 100])
    return float(macro), float(lo), float(hi)


def get_per_cluster(summary: dict) -> dict:
    return summary.get("signal_recall", {}).get("per_cluster", {})


def get_n_valid(summary: dict) -> int:
    v = summary.get("n_valid_final") or summary.get("valid_per_round")
    return v[-1] if isinstance(v, list) else int(v)


def process_condition(label: str, path: Path, rng) -> dict:
    summary     = load_summary(path)
    pred, obs   = extract_ranks(summary)
    x, y        = ranks_to_numeric(pred, obs)

    rho_obs, p_perm       = permutation_spearman(x, y, N_PERMUTATIONS, rng)
    rho_ci_lo, rho_ci_hi  = bootstrap_rho(x, y, BOOTSTRAP_N, rng)

    per_cluster            = get_per_cluster(summary)
    recall, rc_lo, rc_hi  = bootstrap_recall(per_cluster, BOOTSTRAP_N, rng)

    return {
        "condition":    label,
        "n_valid":      get_n_valid(summary),
        "rho":          rho_obs,
        "rho_ci_lo":    rho_ci_lo,
        "rho_ci_hi":    rho_ci_hi,
        "p_perm":       p_perm,
        "recall":       recall,
        "recall_ci_lo": rc_lo,
        "recall_ci_hi": rc_hi,
    }


def write_csv(rows, out: Path) -> None:
    import csv
    fields = ["condition", "n_valid", "rho", "rho_ci_lo", "rho_ci_hi",
              "p_perm", "recall", "recall_ci_lo", "recall_ci_hi"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved: {out}")


def write_latex(rows, out: Path) -> None:
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{CE-Delphi RAG ablation: Spearman $\rho$ (permutation-based) and macro recall "
        r"with 95\% bootstrap CI ($n=11$ clusters).}",
        r"\label{tab:delphi_stats}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Cond. & $N_{\text{valid}}$ & $\rho$ [95\% CI] & $p_{\text{perm}}$ & Recall [95\% CI] \\",
        r"\midrule",
    ]
    for r in rows:
        rho_str    = f"{r['rho']:.3f} [{r['rho_ci_lo']:.3f}, {r['rho_ci_hi']:.3f}]"
        recall_str = f"{r['recall']:.3f} [{r['recall_ci_lo']:.3f}, {r['recall_ci_hi']:.3f}]"
        p_str      = f"{r['p_perm']:.3f}" if r["p_perm"] >= 0.001 else r"$<$0.001"
        lines.append(
            f"  {r['condition']} & {r['n_valid']} & {rho_str} & {p_str} & {recall_str} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\small",
        r"\item Statistical tests are underpowered at $n=11$ clusters; we report effect sizes "
        r"and confidence intervals rather than significance thresholds.",
        r"\item $p_{\text{perm}}$: two-tailed permutation test, 10{,}000 permutations.",
        r"\item Bootstrap CI: 10{,}000 resamples, seed 42.",
        r"\end{tablenotes}",
        r"\end{table}",
    ]
    out.write_text("\n".join(lines))
    print(f"Saved: {out}")


def main():
    rng = np.random.default_rng(RNG_SEED)

    condition_dirs = sorted(RESULTS_DIR.glob("condition_*/pipeline_summary.json"))
    if not condition_dirs:
        print(f"No condition folders found under {RESULTS_DIR}")
        return

    rows = []
    for path in condition_dirs:
        label = path.parent.name.replace("condition_", "")
        print(f"Processing condition {label} ...")
        try:
            row = process_condition(label, path, rng)
            rows.append(row)
            print(f"  rho={row['rho']:.3f} [{row['rho_ci_lo']:.3f}, {row['rho_ci_hi']:.3f}]"
                  f"  p_perm={row['p_perm']:.3f}"
                  f"  recall={row['recall']:.3f} [{row['recall_ci_lo']:.3f}, {row['recall_ci_hi']:.3f}]")
        except Exception as e:
            print(f"  [ERROR] {e}")

    if rows:
        out_dir = RESULTS_DIR.parent
        write_csv(rows, out_dir / "ce_delphi_stats.csv")
        write_latex(rows, out_dir / "ce_delphi_stats.tex")


if __name__ == "__main__":
    main()
