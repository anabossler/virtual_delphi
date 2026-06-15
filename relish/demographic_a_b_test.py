"""
Field of Study Sensitivity Analysis for Persona Grounding

This script tests whether LLM agent outputs causally shift when the academic
field_of_study varies, holding all true demographic contexts (age, country) constant.
If FA distributions differ systematically above the baseline intra-method noise ceiling,
we establish that persona framing causally alters model judgment.

Design:
    - 1 fixed true demographic profile (early 20s, living in Spain)
    - 6 academic field_of_study variants running concurrently
    - 5 narratives sampled (seed=42, identical across all conditions)
    - 3 models evaluated using OpenRouter API
    - Output validation ensuring weight matrices adhere to sum-to-one constraints.
"""

import json
import urllib.request
import time
import re
import random
import argparse
import sys
from pathlib import Path
import pandas as pd

# ===========================================================================
# CONFIGURACIÓN DE PARÁMETROS Y CLI
# ===========================================================================

parser = argparse.ArgumentParser(description="Field of study sensitivity test for LLM persona grounding.")
parser.add_argument("--api-key", required=True, help="OpenRouter API key string.")
parser.add_argument("--input-dir", default=".", help="Directory containing the source sentiment_results.csv file.")
args = parser.parse_args()

API_KEY = args.api_key
BASE = Path(args.input_dir)
OUT = BASE / "field_of_study_ab_results"
OUT.mkdir(exist_ok=True)

MODELS = ["qwen/qwen3-8b", "google/gemma-3-12b-it", "google/gemma-3-27b-it"]
FIELDS = [
    "Humanities and the Arts",
    "Natural Sciences, Mathematics and Statistics",
    "Health and Welfare",
    "Social Sciences, Journalism and Information",
    "Business, Administration and Law",
    "Engineering, Manufacturing and Construction",
]
N_REPS = 3
SEED = 42

FA_LABELS = {
    "FA1": "Technology & Smart Cooking",
    "FA2": "Sustainability & Alternative Proteins",
    "FA3": "Tradition, Culture & Social Eating",
    "FA4": "Health & Personalised Nutrition",
    "FA5": "Food Systems & Distribution",
}

PROMPT_TEMPLATE = """You are a {field} university student in your early 20s, living in Spain.
You have been asked to read these student visions of food in 2050 and estimate
how prevalent each of the following five macroareas is in the corpus.

Visions:
{narratives}

Macroareas:
FA1: Technology & Smart Cooking
FA2: Sustainability & Alternative Proteins
FA3: Tradition, Culture & Social Eating
FA4: Health & Personalised Nutrition
FA5: Food Systems & Distribution

Output ONLY JSON with weights summing to 1:
{{"FA1": 0.x, "FA2": 0.x, "FA3": 0.x, "FA4": 0.x, "FA5": 0.x}}"""

# ===========================================================================
# LLAMADAS A API Y CONTROL DE CALIDAD MATEMÁTICA
# ===========================================================================

def query(prompt: str, model: str, retries: int = 3) -> dict | None:
    """Consulta OpenRouter y valida estrictamente la consistencia sintáctica y matemática."""
    data = json.dumps({
        "model": model, 
        "temperature": 0.7, 
        "max_tokens": 200,
        "messages": [{"role": "user", "content": prompt}]
    }).encode()
    
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
    )
    
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=45) as r:
                raw = json.loads(r.read())["choices"][0]["message"]["content"].strip()
                m = re.search(r'\{[^}]*FA1[^}]*\}', raw, re.DOTALL)
                if m:
                    resp_dict = json.loads(m.group(0))
                    
                    # Validación metodológica obligatoria: comprobar la suma de probabilidades
                    total_probability = sum(float(v) for v in resp_dict.values() if isinstance(v, (int, float)))
                    if abs(total_probability - 1.0) > 0.05:
                        print(f"  [Warning] Invalid distribution sum ({total_probability:.3f}) from model response. Retrying...")
                        continue
                    
                    return resp_dict
        except Exception as e:
            print(f"  [API Error] Attempt {attempt + 1} failed for model {model}: {e}")
            time.sleep(2)
            
    return None

# ===========================================================================
# PIPELINE PRINCIPAL DE ANÁLISIS
# ===========================================================================

def main():
    sentiment_csv = BASE / "sentiment_results.csv"
    if not sentiment_csv.exists():
        sys.exit(f"[ERROR] Source file not found: {sentiment_csv}\nEnsure --input-dir points to the right path.")

    df = pd.read_csv(sentiment_csv)
    df = df[df["ai_flag"] == "human"].copy()
    df = df[df["day_in_life_2050"].str.len() >= 80].copy()
    all_narr = df["day_in_life_2050"].tolist()

    if len(all_narr) < 5:
        sys.exit("[ERROR] Insufficient valid human responses to extract fixed seed narratives sample.")

    rng = random.Random(SEED)
    sample = rng.sample(all_narr, 5)
    narratives_text = "\n\n".join(f"[{i+1}] {n[:400]}" for i, n in enumerate(sample))
    print(f"Sample fixed for all conditions (seed={SEED}): {len(sample)} narratives")

    rows = []
    total = len(FIELDS) * len(MODELS) * N_REPS
    done = 0

    for field in FIELDS:
        prompt = PROMPT_TEMPLATE.format(field=field, narratives=narratives_text)
        for model in MODELS:
            for rep in range(N_REPS):
                resp = query(prompt, model)
                done += 1
                status = "ok" if resp else "fail"
                print(f"  [{done}/{total}] {field[:30]:30s} {model[:25]:25s} rep={rep} {status}")
                if resp:
                    rows.append({"field": field, "model": model, "rep": rep, **resp})
                time.sleep(0.15)

    if not rows:
        sys.exit("[ERROR] No valid iterations were captured from the API. Execution aborted.")

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "field_of_study_ab_responses.csv", index=False)

    print(f"\nValid iterations: {len(res)}/{total}")
    print("\n=== Mean FA distribution by field_of_study ===")
    means = res.groupby("field")[list(FA_LABELS)].mean().round(3)
    print(means.to_string())

    print("\n=== Across-field spread (max - min, pp) ===")
    spread = (means.max() - means.min()) * 100
    for fa in FA_LABELS:
        print(f"  {fa}: {spread[fa]:.2f} pp")

    # Comparación formal frente al ruido intrínseco del método de simulación
    print("\n=== INTERPRETATION ===")
    max_spread = spread.max()
    print(f"Maximum across-field spread: {max_spread:.2f} pp")
    print(f"Intra-method noise ceiling (seed sensitivity): 1.22 pp")
    
    if max_spread > 1.22 * 2:
        print("=> Across-field spread EXCEEDS baseline variance: academic persona context causally matters.")
    elif max_spread > 1.22:
        print("=> Across-field spread COMPARABLE to baseline noise: marginal/weak persona anchoring effect.")
    else:
        print("=> Across-field spread BELOW noise ceiling: academic context does NOT systematically shift model output.")

    summary = {
        "n_responses": len(res),
        "means_by_field": means.to_dict(),
        "spread_pp": spread.to_dict(),
        "max_spread_pp": float(max_spread),
        "noise_ceiling_pp": 1.22,
    }
    
    summary_path = OUT / "field_of_study_ab_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved analysis artifacts down inside: {OUT}")


if __name__ == "__main__":
    main()
