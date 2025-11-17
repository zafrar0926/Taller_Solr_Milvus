import requests
import json

SOLR_URL = "http://localhost:8983/solr/rag_core/select"

def query_solr(query, top_k=5):
    params = {
        "q": f"text_raw:({query})",
        "fl": "id,score,section_title,text_raw",
        "rows": top_k,
        "wt": "json"
    }
    r = requests.get(SOLR_URL, params=params)
    r.raise_for_status()
    data = r.json()

    docs = data["response"]["docs"]
    return [
        {
            "id": d.get("id"),
            "score": d.get("score"),
            "title": d.get("section_title"),
            "text_snippet": (d.get("text_raw") or "")[:200]
        }
        for d in docs
    ]



if __name__ == "__main__":
    q = "participación de empresas privadas en el conflicto"
    print(f"🔍 Consulta SOLR: {q}\n")

    results = query_solr(q, top_k=5)
    print(json.dumps(results, indent=2, ensure_ascii=False))
