import os, json, requests, pathlib

BASE = pathlib.Path(__file__).resolve().parents[1]
SEED_PATH = BASE / "data/queries_seed.txt"
OUT_PATH  = BASE / "data/gold_weak.jsonl"

API = os.getenv("RAG_API", "http://localhost:8000")
TOP_K = int(os.getenv("GOLD_TOPK", "10"))

def query_api(endpoint: str, q: str, k: int):
    url = f"{API}/{endpoint}"
    data = {"query": q, "top_k": k}
    try:
        r = requests.post(url, json=data, timeout=30)
        r.raise_for_status()
        resp = r.json()
        return [d["id"] for d in resp.get("results", [])]
    except Exception as e:
        print(f"⚠️ Error en {endpoint} para '{q}': {e}")
        return []

def main():
    if not SEED_PATH.exists():
        raise FileNotFoundError(f"No existe {SEED_PATH}")
    lines = [l.strip() for l in SEED_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    out = OUT_PATH.open("w", encoding="utf-8")

    for q in lines:
        print(f"🔍 Procesando query: {q}")
        solr_docs = query_api("query_solr", q, TOP_K)
        milvus_docs = query_api("query_milvus", q, TOP_K)

        intersection = list(set(solr_docs) & set(milvus_docs))
        union = list(set(solr_docs) | set(milvus_docs))
        partial = [d for d in union if d not in intersection]

        rec = {
            "query": q,
            "relevant_doc_ids": intersection,
            "partially_relevant_doc_ids": partial,
            "expected_answer_summary": ""
        }
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    out.close()
    print(f"✅ Gold estándar débil generado en {OUT_PATH}")

if __name__ == "__main__":
    main()