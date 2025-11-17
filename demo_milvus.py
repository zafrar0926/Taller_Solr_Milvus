from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
import json

# --- CONFIG ---
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "rag_corpus"   # cámbiala si tu colección tiene otro nombre

# --- Conectar ---
connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

# --- Cargar colección ---
col = Collection(COLLECTION_NAME)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def query_milvus(query, top_k=5):
    vec = model.encode([query]).tolist()
    res = col.search(
        data=vec,
        anns_field="embedding",
        param={"metric_type": "COSINE"},
        limit=top_k,
        output_fields=["id", "section_title", "text_raw"]

    )

    hits = res[0]
    out = []
    for h in hits:
        out.append({
            "id": h.entity.get("id"),
            "score": float(h.distance),
            "text_snippet": h.entity.get("text_raw", "")[:200]
        })
    return out

if __name__ == "__main__":
    q = "participación de empresas privadas en el conflicto"
    print(f"🔍 Consulta MILVUS: {q}\n")

    res = query_milvus(q, top_k=5)
    print(json.dumps(res, indent=2, ensure_ascii=False))
