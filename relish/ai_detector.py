"""
AI Screening and Semantic Analysis Pipeline

This software implements a rule-based AI screening procedure
combined with semantic topic analysis for open-ended survey responses.

Important:
This tool is not intended as a definitive AI detector.
It provides heuristic screening based on structural and stylistic signals.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Verificación proactiva de recursos NLTK
def ensure_nltk_resources():
    """Comprueba que los recursos de NLTK existen."""
    resources = {
        "corpora/stopwords": "stopwords",
        "tokenizers/punkt": "punkt",
    }

    for resource_path, resource_name in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            print(f"[INFO] Downloading NLTK resource: {resource_name}")
            nltk.download(resource_name)

ensure_nltk_resources()

# ===========================================================================
# 1. CONFIGURACIÓN Y PATRONES DEL DETECTOR DE IA
# ===========================================================================

STRUCTURAL_PATTERNS: list[tuple[str, str]] = [
    (r"\*\*[^*]+\*\*",          "markdown_bold"),
    (r"(?m)^#{1,6}\s+\S",      "markdown_header"),
    (r"(?m)^---\s*$",           "horizontal_rule"),
    (r"\bhere(?:'|\u2019)s\s+(a|an|the|your|my)\b",     "heres_a"),
    (r"\bsure[!,]?\s+here(?:'|\u2019)s\b",              "sure_heres"),
    (r"\bcertainly[!,]?\s+here\b",                       "certainly_here"),
    (r"\bof\s+course[!,]?\s+here\b",                    "of_course_here"),
    (r"\bi(?:'|\u2019)ll\s+(write|create|draft|provide|craft)\b", "ill_write"),
    (r"\b(below|above)\s+is\s+(a|an|the)\b",            "below_is_a"),
    (r"\bas\s+(requested|instructed|per\s+your)\b",      "as_requested"),
    (r"\bhope\s+this\s+(helps|works)\b",                 "hope_this_helps"),
    (r"\blet\s+me\s+know\s+if\b",                        "let_me_know"),
    (r"\bfeel\s+free\s+to\b",                            "feel_free"),
    (r"\baqu[i\u00ed]\s+tienes\b",   "es_aqui_tienes"),
    (r"\bclaro[,!]\s",               "es_claro"),
    (r"\bvale[,!]\s+aqu",            "es_vale_aqui"),
    (r"\bespero\s+que\s+te\s+sirva\b", "es_espero_sirva"),
    (r"\bvoici\s+(une?|le|la|mon|ma)\b",          "fr_voici"),
    (r"\bj(?:'|\u2019)esp[e\u00e8]re\s+que\b",   "fr_jespere"),
    (r"[\U0001F300-\U0001F9FF\u2700-\u27BF]",    "emoji"),
]

LLM_CLICHE_PATTERNS: list[str] = [
    r"\bsoft(?:ly)?\s+(?:hum|glow|amber|light)\b",
    r"\bthe\s+(?:soft|gentle|warm)\s+(?:hum|glow|light|whisper)\b",
    r"\bgentle\s+\w+\b",
    r"\bthe\s+air\s+(?:is|feels|smells|hums)\b",
    r"\bbarely\s+a\s+whisper\b",
    r"\bnot\s+just\s+\w+,?\s+but\b",
    r"\bisn'?t\s+just\s+\w+,?\s+it'?s\b",
    r"\bstands?\s+(?:as\s+)?a\s+testament\b",
    r"\btestament\s+to\b",
    r"\btapestry\s+of\b",
    r"\bsymphony\s+of\b",
    r"\bin\s+the\s+heart\s+of\b",
    r"\ba\s+world\s+where\b",
    r"\bmuch\s+like\s+a\b",
    r"\bseamless(?:ly)?\b",
    r"\bensuring\b",
    r"\bharness(?:ing)?\b",
    r"\bempowering?\b",
    r"\bunderscores?\b",
    r"\bdelve\b",
    r"\bnuanced\b",
    r"\bholistic\b",
    r"\bvibrant\b",
    r"\bbustling\b",
    r"\bever[- ]evolving\b",
    r"\brevolutioniz(?:e|ing|ed)\b",
    r"\bparadigm\s+shift\b",
    r"\blandscape\s+of\b",
    r"\bnavigat(?:e|ing)\s+the\s+\w+\s+landscape\b",
    r"\boptimal(?:ly)?\b",
    r"\bpalpable\b",
    r"\bbio[- ]?(?:monitor|sensor|metric|link)\b",
    r"\bsmart[- ](?:glass|kitchen|home|fridge|appliance)\b",
    r"\bculinary\s+(?:synthesizer|orchestra|landscape|journey|hub)\b",
    r"\bnutrient\s+(?:profile|cartridge|optimization|dense)\b",
    r"\bpersonaliz(?:e|ed|ing)\s+nutrition\b",
    r"\bmolecular\s+(?:gastronomy|printer|assembler|resonance)\b",
    r"\blab[- ](?:grown|cultured|created)\b",
    r"\bvertical\s+farm(?:ing|s)?\b",
    r"\bsubdermal\s+sensor\b",
]

_SMART_QUOTE_RE = re.compile(r"[\u201c\u201d\u2018\u2019]")
_EM_DASH_RE     = re.compile(r"\u2014|\u2013")
_SENTENCE_SPLIT = re.compile(r"[.!?]+")
_CLICHE_RES     = [re.compile(p) for p in LLM_CLICHE_PATTERNS]

EM_TIER_1     = 2    
EM_TIER_2     = 3    
EM_TIER_3     = 5    
SQ_TIER_1     = 5    
SQ_TIER_2     = 10   
CLICHE_TIER_1 = 2    
CLICHE_TIER_2 = 4    
AWL_THRESHOLD       = 4.8   
AWL_MIN_WORDS       = 200   
AWS_THRESHOLD       = 12.0  

THRESHOLD_PROBABLE = 3   
THRESHOLD_SUSPECT  = 1   

# ===========================================================================
# 2. FUNCIONES INTERNAS DEL DETECTOR DE IA
# ===========================================================================

def _count_sentences(text: str) -> int:
    parts = _SENTENCE_SPLIT.split(text)
    return max(1, sum(1 for p in parts if p.strip()))

def extract_features(text: str) -> dict:
    words = text.split()
    word_count = len(words)
    n_sentences = _count_sentences(text)

    structural_hits = [
        label
        for pattern, label in STRUCTURAL_PATTERNS
        if re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
    ]

    cliche_hits = sum(1 for r in _CLICHE_RES if r.search(text.lower()))
    em_dash_count    = len(_EM_DASH_RE.findall(text))
    smart_quote_count = len(_SMART_QUOTE_RE.findall(text))

    unique_words = {w.lower().strip(".,;:!?\"'()[]") for w in words}
    ttr = len(unique_words) / word_count if word_count else 0.0

    avg_word_len = (
        sum(len(w) for w in words) / word_count if word_count else 0.0
    )
    avg_words_per_sentence = word_count / n_sentences

    return {
        "word_count":             word_count,
        "em_dash_count":          em_dash_count,
        "smart_quote_count":      smart_quote_count,
        "llm_cliche_hits":        cliche_hits,
        "type_token_ratio":       round(ttr, 6),
        "avg_word_len":           round(avg_word_len, 6),
        "avg_words_per_sentence": round(avg_words_per_sentence, 6),
        "_structural_hits":       structural_hits,
    }

def classify(features: dict, min_words: int = 50) -> tuple[str, int, list[str]]:
    if features["word_count"] < min_words:
        return "human", 0, []

    struct = features["_structural_hits"]
    
    # Cada trigger estructural suma evidencia en lugar de marcar positivo directo
    score = len(struct)

    em = features["em_dash_count"]
    if em >= EM_TIER_1: score += 1
    if em >= EM_TIER_2: score += 1
    if em >= EM_TIER_3: score += 1

    sq = features["smart_quote_count"]
    if sq >= SQ_TIER_1: score += 1
    if sq >= SQ_TIER_2: score += 1

    ch = features["llm_cliche_hits"]
    if ch >= CLICHE_TIER_1: score += 1
    if ch >= CLICHE_TIER_2: score += 1

    if (
        features["avg_word_len"] > AWL_THRESHOLD
        and features["word_count"] >= AWL_MIN_WORDS
        and features["avg_words_per_sentence"] > AWS_THRESHOLD
    ):
        score += 1

    if score >= THRESHOLD_PROBABLE:
        label = "probable_ai"
    elif score >= THRESHOLD_SUSPECT:
        label = "suspect"
    else:
        label = "human"

    return label, score, struct

# ===========================================================================
# 3. FUNCIONES DEL NUEVO ANÁLISIS SEMÁNTICO (BILINGÜE EN/ES)
# ===========================================================================

def preprocess_text(text: str) -> str:
    """Limpia el texto eliminando puntuación y conectores en inglés y español."""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)  
    
    # Unir diccionarios de conectores (stopwords) para encuestas bilingües
    stop_words = set(stopwords.words('english')).union(set(stopwords.words('spanish')))
    
    # Eliminar palabras que sesgan el análisis cualitativo
    stop_words.update(['2050', 'day', 'life', 'future', 'ano', 'vida', 'futuro', 'dia', 'mas', 'like', 'one']) 
    
    words = text.split()
    filtered = [w for w in words if w not in stop_words and not w.isdigit() and len(w) > 2]
    return " ".join(filtered)


def analyze_semantics(results_csv: str, original_csv: str, text_col: str):
    """Procesa cualitativamente solo las respuestas clasificadas como humanas."""
    path_results = Path(results_csv)
    path_original = Path(original_csv)
    
    df_results = pd.read_csv(path_results)
    df_original = pd.read_csv(path_original)

    df_results[text_col] = df_original[text_col]

    df_human = df_results[df_results['ai_flag'] == 'human'].copy()
    n_human = len(df_human)
    print(f"\n" + "="*60)
    print(f" INICIANDO ANÁLISIS SEMÁNTICO EN {n_human} RESPUESTAS HUMANAS")
    print("="*60 + "\n")

    if n_human == 0:
        print("No hay respuestas marcadas como 'human' para analizar.")
        return

    df_human[text_col] = df_human[text_col].fillna("").astype(str)
    df_human['cleaned_text'] = df_human[text_col].apply(preprocess_text)

    # --- Capa 1: Conceptos Clave (N-gramas) ---
    print("1. Extrayendo los conceptos compuestos más repetidos (N-gramas)...")
    try:
        vectorizer = CountVectorizer(ngram_range=(2, 3), max_features=10)
        ngrams_matrix = vectorizer.fit_transform(df_human['cleaned_text'])
        ngram_counts = np.asarray(ngrams_matrix.sum(axis=0)).flatten()
        ngram_words = vectorizer.get_feature_names_out()
        
        print("\n[Conceptos Clave del Futuro Humano]")
        for word, count in sorted(zip(ngram_words, ngram_counts), key=lambda x: x[1], reverse=True):
            print(f"  - '{word}': mencionado {count} veces")
    except ValueError:
        print("\n[AVISO] Datos insuficientes para extraer conceptos combinados.")
    print("-" * 50)

    # --- Capa 2: Ejes Temáticos (LDA) ---
    n_topics = 3
    print(f"2. Agrupando las respuestas en {n_topics} grandes ejes temáticos...")
    try:
        # Metodológicamente consistente: LDA sobre conteos de términos (Bag of Words)
        count_vectorizer = CountVectorizer(max_features=500, min_df=2)
        count_matrix = count_vectorizer.fit_transform(df_human['cleaned_text'])
        
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(count_matrix)
        
        terms = count_vectorizer.get_feature_names_out()
        print("\n[Ejes Temáticos Detectados en la Población]")
        for topic_idx, topic in enumerate(lda.components_):
            top_terms_idx = topic.argsort()[:-6:-1]  
            top_terms = [terms[i] for i in top_terms_idx]
            print(f"  Tema #{topic_idx + 1}: {', '.join(top_terms)}")
    except ValueError:
        print("\n[AVISO] Vocabulario muy corto para estructurar ejes temáticos.")
    print("-" * 50)

    # --- Capa 3: Tonalidad Emocional ---
    diccionario_tono = {
        'positive': ['sustainable', 'clean', 'peaceful', 'efficient', 'healthy', 'easy', 'together', 'sostenible', 'saludable', 'mejor', 'limpio', 'paz'],
        'negative': ['crisis', 'expensive', 'collapsed', 'war', 'lonely', 'difficult', 'climate', 'scarcity', 'dificil', 'escasez', 'caro', 'guerra']
    }
    
    def eval_tone(text):
        tokens = text.split()
        pos = sum(1 for w in tokens if w in diccionario_tono['positive'])
        neg = sum(1 for w in tokens if w in diccionario_tono['negative'])
        if pos > neg: return 'Optimista (Utopía)'
        if neg > pos: return 'Preocupado (Distopía)'
        return 'Neutral / Balanceado'

    df_human['vision_tonality'] = df_human['cleaned_text'].apply(eval_tone)
    tone_summary = df_human['vision_tonality'].value_counts()
    
    print("3. Análisis de Tonalidad de la visión humana hacia el 2050:")
    for tone, count in tone_summary.items():
        print(f"  - {tone:<22} {count:>3} respuestas ({count/n_human*100:.1f}%)")
        
    output_path = path_results.parent / "human_semantic_analysis.csv"
    df_human.to_csv(output_path, index=False)
    print(f"\n[INFO] Análisis semántico guardado con éxito en: {output_path}")

# ===========================================================================
# 4. PIPELINE PRINCIPAL (CONECTA DETECTOR + ANALIZADOR)
# ===========================================================================

def run(
    input_path: str | Path,
    output_path: str | Path,
    text_col: str = "day_in_life_2050",
    min_words: int = 50,
    verbose: bool = False,
) -> pd.DataFrame:
    input_path  = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        sys.exit(f"[ERROR] Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    if text_col not in df.columns:
        sys.exit(f"[ERROR] Column '{text_col}' not found.")

    df[text_col] = df[text_col].fillna("").astype(str)

    feature_rows = []
    for text in df[text_col]:
        feats = extract_features(text)
        label, score, struct_hits = classify(feats, min_words=min_words)
        row_out = {k: v for k, v in feats.items() if not k.startswith("_")}
        row_out["ai_score"]            = score
        row_out["structural_triggers"] = "|".join(struct_hits)
        row_out["ai_flag"]             = label
        feature_rows.append(row_out)

    features_df = pd.DataFrame(feature_rows)

    drop_cols = {text_col, f"{text_col}-Comment", "prolific_id"}
    metadata_cols = [c for c in df.columns if c not in drop_cols]
    result = pd.concat([df[metadata_cols].reset_index(drop=True), features_df], axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    counts = result["ai_flag"].value_counts()
    n = len(result)
    print(f"\nResults saved to: {output_path}")
    print(f"\nClassification summary  (n={n}):")
    for label in ("probable_ai", "suspect", "human"):
        c = counts.get(label, 0)
        print(f"  {label:<14} {c:>4}  ({c / n * 100:.1f}%)")

    if verbose:
        prob = result[result["ai_flag"] == "probable_ai"]
        if not prob.empty:
            display_cols = ["word_count", "em_dash_count", "smart_quote_count", "llm_cliche_hits", "ai_score", "structural_triggers"]
            available = [c for c in display_cols if c in prob.columns]
            print("\nProbable-AI rows:")
            print(prob[available].to_string(index=False))

    # Ejecución automática del análisis semántico cualitativo
    analyze_semantics(
        results_csv=str(output_path), 
        original_csv=str(input_path), 
        text_col=text_col
    )

    print("\n[DISCLAIMER]")
    print(
        "This classifier is heuristic-based and should be used only "
        "as an exploratory screening tool."
    )

    return result

# ===========================================================================
# 5. CLI INTERFAZ DE COMANDOS
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect AI and analyze text semantics.")
    parser.add_argument("--input", default="visions_survey_data.csv")
    parser.add_argument("--output", default="ai_detection_results.csv")
    parser.add_argument("--text-col", default="day_in_life_2050")
    parser.add_argument("--min-words", type=int, default=50)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    run(
        input_path=args.input,
        output_path=args.output,
        text_col=args.text_col,
        min_words=args.min_words,
        verbose=args.verbose,
    )
