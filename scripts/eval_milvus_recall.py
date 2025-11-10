import os, json, requests, collections, math
from pathlib import Path

API = os.getenv("RAG_API", "http://localhost:8000")
GOLD_PATH = Path("data/gold_weak.jsonl")
K = int(os.getenv("EVAL_K", "5"))

def load_gold(path):
    import collections, json
    gold = collections.defaultdict(set)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                for doc in r.get("relevant_doc_ids", []):
                    gold[r["query"]].add(doc)
            except Exception as e:
                print(f"⚠️ Error leyendo línea: {e}")
    return gold


def query_milvus(q, k):
    r = requests.post(f"{API}/query_milvus",
                      json={"query": q, "top_k": k},
                      timeout=30)
    r.raise_for_status()
    data = r.json()
    return [d["id"] for d in data.get("results", [])]

def main():
    gold = load_gold(GOLD_PATH)
    qs = sorted(gold.keys())
    totals = []
    for q in qs:
        gt = gold[q]
        pred = query_milvus(q, K)
        hit = len(set(pred) & gt)
        denom = min(K, len(gt)) if len(gt) > 0 else 1
        recall = hit / denom
        totals.append(recall)
        print(f"[{q}] hits={hit}/{denom} recall@{K}={recall:.2f}")

    avg = sum(totals) / len(totals) if totals else 0.0
    print(f"\n🔥 AVG recall@{K}: {avg:.3f} sobre {len(totals)} consultas")

if __name__ == "__main__":
    main()
