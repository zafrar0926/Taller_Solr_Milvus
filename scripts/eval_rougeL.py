from pathlib import Path
import json, requests, time
from tqdm import tqdm
from rouge_score import rouge_scorer
import pandas as pd
import matplotlib.pyplot as plt

# === Configuración de rutas ===
API_URL = "http://localhost:8000/query_milvus"
CORPUS_PATH = Path("/home/zafrar09/Taller_RAG/data/corpus/books_preprocessed_MWE.jsonl")
GOLD_PATH = Path("/home/zafrar09/Taller_RAG/data/gold_weak.jsonl")
REPORTS_DIR = Path("/home/zafrar09/Taller_RAG/reports")
REPORTS_DIR.mkdir(exist_ok=True)

# === Cargar el corpus completo (id -> texto) ===
corpus = {}
with open(CORPUS_PATH, encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        corpus[d["section_id"]] = d["text_raw"]

# === Configurar métrica ROUGE-L ===
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
results = []

# === Cargar gold estándar ===
with open(GOLD_PATH, encoding="utf-8") as f:
    gold_data = [json.loads(line) for line in f]

# === Evaluación principal ===
for entry in tqdm(gold_data, desc="Evaluando ROUGE-L"):
    query = entry["query"]
    relevant_ids = entry["relevant_doc_ids"]

    # Texto de referencia (concatenación de los documentos relevantes)
    ref_text = " ".join([corpus[i] for i in relevant_ids if i in corpus])

    # Solicitar resultados al API
    start = time.time()
    try:
        r = requests.post(API_URL, json={"query": query, "top_k": 3}, timeout=20)
        r.raise_for_status()
        retrieved = r.json().get("results", [])
        retrieved_text = " ".join([doc.get("text_raw", "") for doc in retrieved])
    except Exception as e:
        print(f"⚠️ Error en '{query[:50]}': {e}")
        continue

    latency = time.time() - start
    score = scorer.score(ref_text, retrieved_text)['rougeL'].fmeasure

    results.append({
        "query": query,
        "rougeL_f": round(score, 4),
        "latency_s": round(latency, 2)
    })

# === Guardar resultados ===
df = pd.DataFrame(results)
df.to_csv(REPORTS_DIR / "metrics_rougeL.csv", index=False)
print("\n✅ Métricas ROUGE-L guardadas en /reports/metrics_rougeL.csv")
print(df.describe())

# === Visualización ===
plt.figure(figsize=(8,4))
plt.barh(df["query"].str[:70], df["rougeL_f"], color="#6a5acd")
plt.xlabel("ROUGE-L F1")
plt.title("Evaluación de Similaridad Textual - Milvus (corregida)")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "metrics_rougeL.png", dpi=150)
plt.show()
