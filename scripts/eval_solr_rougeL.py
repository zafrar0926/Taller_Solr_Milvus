#!/usr/bin/env python3
# ===============================================================
# 🧠 Evaluador de Recuperación Léxica (Solr)
# ROUGE-L + Latencia + Diagnóstico de textos vacíos
# ===============================================================
from pathlib import Path
import json, requests, time
from tqdm import tqdm
from rouge_score import rouge_scorer
import pandas as pd
import matplotlib.pyplot as plt

# --- Configuración ---
API_URL = "http://localhost:8000/query_solr"
GOLD_PATH = Path("/home/zafrar09/Taller_RAG/data/gold_weak.jsonl")
REPORTS_DIR = Path("/home/zafrar09/Taller_RAG/reports")
REPORTS_DIR.mkdir(exist_ok=True)

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
results = []
empty_queries = []

# --- Cargar gold standard ---
with open(GOLD_PATH, encoding="utf-8") as f:
    gold_data = [json.loads(line) for line in f]

# --- Evaluación ---
for entry in tqdm(gold_data, desc="Evaluando ROUGE-L (Solr)"):
    query = entry["query"]
    relevant_ids = set(entry["relevant_doc_ids"])
    start = time.time()

    try:
        r = requests.post(API_URL, json={"query": query, "top_k": 3}, timeout=20)
        r.raise_for_status()
        retrieved = r.json().get("results", [])
    except Exception as e:
        print(f"⚠️ Error en '{query[:50]}': {e}")
        continue

    latency = time.time() - start

    # --- Limpieza y concatenación de textos ---
    retrieved_texts = []
    for doc in retrieved:
        text_raw = doc.get("text_raw", "")
        if isinstance(text_raw, list):
            text_raw = " ".join(map(str, text_raw))
        section = doc.get("section_title", "")
        if isinstance(section, list):
            section = " ".join(map(str, section))
        combined = f"{section} {text_raw}".strip()
        if combined:
            retrieved_texts.append(combined)

    retrieved_text = " ".join(retrieved_texts)
    # --- Construir referencia real a partir del corpus ---
    corpus_path = Path("/home/zafrar09/Taller_RAG/data/corpus/books_preprocessed_MWE.jsonl")
    corpus = {json.loads(line)["section_id"]: json.loads(line).get("text_raw", "")
            for line in open(corpus_path, encoding="utf-8")}

    ref_texts = [corpus.get(doc_id, "") for doc_id in relevant_ids if doc_id in corpus]
    ref_text = " ".join(ref_texts)

    # --- Diagnóstico ---
    if len(retrieved_text.strip()) == 0:
        empty_queries.append(query)
        rouge_score = 0.0
    else:
        rouge_score = scorer.score(ref_text, retrieved_text)['rougeL'].fmeasure

    results.append({
        "query": query,
        "rougeL_f": rouge_score,
        "latency_s": latency,
        "has_text": len(retrieved_text.strip()) > 0
    })

# --- Guardar resultados ---
df = pd.DataFrame(results)
csv_path = REPORTS_DIR / "metrics_solr_rougeL.csv"
df.to_csv(csv_path, index=False)

# --- Diagnóstico de texto vacío ---
total = len(df)
empty_count = len(df[df["has_text"] == False])
print("\n🩺 Diagnóstico de recuperación:")
print(f"• Total de queries evaluadas: {total}")
print(f"• Queries con texto vacío: {empty_count}")
if empty_count > 0:
    print(f"⚠️ Revisar estas queries sin texto devuelto por Solr:")
    for q in empty_queries:
        print("  -", q[:100])

print("\n✅ Métricas ROUGE-L guardadas en:", csv_path)
print(df.describe()[["rougeL_f", "latency_s"]])

# --- Visualización ---
plt.figure(figsize=(8,4))
plt.barh(df["query"].str[:60], df["rougeL_f"], color="#1f77b4")
plt.xlabel("ROUGE-L F1")
plt.title("Evaluación de Similaridad Textual - Solr")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "metrics_solr_rougeL.png", dpi=150)
plt.show()
