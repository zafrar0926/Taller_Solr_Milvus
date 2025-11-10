import time, json, numpy as np, requests, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

API_URL = "http://localhost:8000/query_milvus"
GOLD_PATH = Path("/home/zafrar09/Taller_RAG/data/gold_weak.jsonl")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

K = 5  # top-k para recall y nDCG

# --- Funciones auxiliares ---
def recall_at_k(relevant, retrieved):
    return len(set(relevant) & set(retrieved[:K])) / max(1, len(relevant))

def mrr(relevant, retrieved):
    for rank, doc in enumerate(retrieved, 1):
        if doc in relevant:
            return 1 / rank
    return 0

def ndcg(relevant, retrieved):
    dcg = 0.0
    for i, doc in enumerate(retrieved[:K]):
        rel = 1 if doc in relevant else 0
        dcg += rel / np.log2(i + 2)
    ideal = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), K)))
    return dcg / ideal if ideal > 0 else 0

# --- Cargar gold estándar ---
with open(GOLD_PATH, encoding="utf-8") as f:
    gold_data = [json.loads(line) for line in f]

results = []

# --- Evaluación ---
for entry in tqdm(gold_data, desc="Evaluando queries"):
    q = entry["query"]
    relevant = entry["relevant_doc_ids"]

    start = time.time()
    try:
        r = requests.post(API_URL, json={"query": q, "top_k": K}, timeout=30)
        r.raise_for_status()
        docs = [d["id"] for d in r.json().get("results", [])]
    except Exception as e:
        print(f"⚠️ Error con query '{q[:40]}': {e}")
        continue
    latency = time.time() - start

    results.append({
        "query": q,
        "recall@5": recall_at_k(relevant, docs),
        "mrr": mrr(relevant, docs),
        "ndcg": ndcg(relevant, docs),
        "latency_s": latency
    })

# --- Resultados ---
df = pd.DataFrame(results)
df.to_csv(REPORTS_DIR / "metrics_milvus.csv", index=False)
print("\n✅ Resultados guardados en /reports/metrics_milvus.csv")
print(df.describe()[["recall@5", "mrr", "ndcg", "latency_s"]])

# --- Gráfico comparativo ---
plt.figure(figsize=(8,5))
df[["recall@5","mrr","ndcg"]].mean().plot(kind="bar", color=["#2ca02c","#1f77b4","#9467bd"])
plt.title("Métricas promedio - Milvus")
plt.ylabel("Score promedio")
plt.ylim(0,1)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(REPORTS_DIR / "metrics_summary_milvus.png", dpi=150)
plt.show()