#!/usr/bin/env python3
# ===============================================================
# 🧪 Evaluador de Recuperación Léxica (Solr) - Recall@k, Latencia
# ===============================================================
import os, json, time, requests, collections
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# --- Configuración ---
API_URL = os.getenv("RAG_API", "http://localhost:8000/query_solr")
GOLD_PATH = Path("/home/zafrar09/Taller_RAG/data/gold_weak.jsonl")
REPORTS_DIR = Path("/home/zafrar09/Taller_RAG/reports")
REPORTS_DIR.mkdir(exist_ok=True)

K = int(os.getenv("EVAL_K", "5"))
results = []

# --- Función para cargar el gold standard ---
def load_gold(path):
    gold = collections.defaultdict(set)
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            for d in r.get("relevant_doc_ids", []):
                gold[r["query"]].add(str(d))
    return gold

# --- Cargar gold standard ---
gold = load_gold(GOLD_PATH)
qs = sorted(gold.keys())

# --- Evaluación principal ---
for q in tqdm(qs, desc="Evaluando Recall@K (Solr)"):
    gt = gold[q]
    start = time.time()

    try:
        r = requests.post(API_URL, json={"query": q, "top_k": K}, timeout=30)
        r.raise_for_status()
        data = r.json()
        docs = data.get("results", [])
    except Exception as e:
        print(f"⚠️ Error en query '{q[:40]}': {e}")
        continue

    latency = time.time() - start
    pred_ids = {str(d.get("id")) for d in docs}
    hits = len(pred_ids & gt)
    denom = min(K, len(gt)) if len(gt) > 0 else 1
    recall = hits / denom if denom > 0 else 0.0

    results.append({
        "query": q,
        "hits": hits,
        "gt_docs": len(gt),
        "recall@{}".format(K): recall,
        "latency_s": latency
    })
    print(f"[{q[:50]}] hits={hits}/{denom} recall@{K}={recall:.2f}")

# --- Guardar resultados ---
df = pd.DataFrame(results)
csv_path = REPORTS_DIR / "metrics_solr_recall.csv"
df.to_csv(csv_path, index=False)

avg_recall = df[f"recall@{K}"].mean()
avg_latency = df["latency_s"].mean()
print(f"\n🔥 AVG recall@{K}: {avg_recall:.3f} | ⏱️ Latencia media: {avg_latency:.2f}s")
print(f"📄 Resultados guardados en {csv_path}")
