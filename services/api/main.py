import os, time
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

from pymilvus import connections, Collection, utility
from sentence_transformers import SentenceTransformer

app = FastAPI(title="RAG Demo - Solr & Milvus (v2)")

# === ENV ===
BACKEND_SOLR    = os.getenv("BACKEND_SOLR", "http://solr:8983/solr/rag_core")
MILVUS_HOST     = os.getenv("MILVUS_HOST", "milvus")
MILVUS_PORT     = int(os.getenv("MILVUS_PORT", "19530"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_corpus")
EMBED_FIELD     = os.getenv("EMBED_FIELD", "embedding")
MODEL_NAME      = os.getenv("MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
TOPK_MAX        = int(os.getenv("TOPK_MAX", "20"))

# Carga perezosa
_model = None
_collection = None
_milvus_connected = False


# === MODELOS DE PETICIÓN ===
class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

class AskRequest(BaseModel):
    query: str
    top_k: int = 5
    backend: str = "both"  # "solr" | "milvus" | "both"


@app.get("/health")
def health():
    return {"status": "ok"}


# === CONEXIÓN A MILVUS ===
def connect_milvus_with_retry(retries: int = 20, sleep_s: float = 1.5):
    global _milvus_connected
    if _milvus_connected:
        return
    for i in range(1, retries + 1):
        try:
            connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
            _milvus_connected = True
            print("✅ Conectado a Milvus.")
            return
        except Exception as e:
            print(f"⏳ Intento {i}/{retries}: fallo conectando a Milvus ({e})")
            time.sleep(sleep_s)
    raise RuntimeError("❌ No se pudo conectar a Milvus tras varios intentos.")


def get_collection() -> Collection:
    global _collection
    if _collection is None:
        connect_milvus_with_retry()
        if not utility.has_collection(COLLECTION_NAME):
            raise RuntimeError(f"⚠️ La colección '{COLLECTION_NAME}' no existe en Milvus.")
        _collection = Collection(COLLECTION_NAME)
        _collection.load()
        print(f"✅ Colección {_collection.name} cargada en memoria.")
    return _collection


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


# === ARRANQUE AUTOMÁTICO ===
@app.on_event("startup")
def on_startup():
    """Warm-up de Solr y carga inicial de Milvus."""
    try:
        requests.get(f"{BACKEND_SOLR}/select?q=*:*&rows=0", timeout=2)
    except Exception:
        pass

    for i in range(10):
        try:
            col = get_collection()
            print("✅ Milvus cargado en memoria al inicio.")
            break
        except Exception as e:
            print(f"⏳ Esperando Milvus... intento {i+1}/10 ({e})")
            time.sleep(3)


@app.get("/")
def root():
    return {"message": "RAG API operativa 🚀 - Usa /query_solr, /query_milvus o /ask"}


# === Endpoint 1: RAG–Solr ===
@app.post("/query_solr")
def query_solr(request: QueryRequest):
    q = (request.query or "").strip()
    k = max(1, min(request.top_k, TOPK_MAX))

    params = {
        "defType": "edismax",
        "q": q,
        "qf": "text_raw lemmas section_title",
        "fl": "id,section_title,text_raw,score",
        "rows": k,
        "wt": "json"
    }
    try:
        r = requests.get(f"{BACKEND_SOLR}/select", params=params, timeout=15)
        r.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Solr error: {e}")

    data = r.json()
    docs = data.get("response", {}).get("docs", [])
    for d in docs:
        d["engine"] = "solr"
        d["norm_score"] = float(d.get("score", 0.0))
    return {"engine": "solr", "query": q, "results": docs}


# === Endpoint 2: RAG–Milvus ===
@app.post("/query_milvus")
def query_milvus(request: QueryRequest):
    q = (request.query or "").strip()
    k = max(1, min(request.top_k, TOPK_MAX))
    try:
        model = get_model()
        qvec = model.encode([q], normalize_embeddings=True).tolist()
        col = get_collection()

        # Asegurar que la colección esté cargada correctamente
        if utility.has_collection(col.name):
            col.load()
            print(f"✅ Colección {col.name} cargada en memoria.")
        else:
            raise HTTPException(status_code=500, detail=f"⚠️ Colección {col.name} no existe en Milvus.")

        results = col.search(
            data=qvec,
            anns_field=EMBED_FIELD,
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=k,
            output_fields=["id", "section_title", "text_raw"]
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Milvus error: {e}")

    hits = [{
        "id": r.entity.get("id"),
        "section_title": r.entity.get("section_title"),
        "text_raw": r.entity.get("text_raw"),
        "score": float(r.distance),
        "engine": "milvus",
        "norm_score": float(r.distance),
    } for r in results[0]]
    return {"engine": "milvus", "query": q, "results": hits}


# === Fusión RRF ===
def rrf_merge(solr_docs: List[Dict[str, Any]], milvus_docs: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    K = 60.0
    ranks: Dict[str, Dict[str, Any]] = {}

    for i, d in enumerate(solr_docs):
        rid = d["id"]
        ranks.setdefault(rid, {"obj": d, "rrf": 0.0})
        ranks[rid]["rrf"] += 1.0 / (K + i + 1)

    for i, d in enumerate(milvus_docs):
        rid = d["id"]
        if rid in ranks:
            ranks[rid]["obj"].setdefault("text_raw", d.get("text_raw"))
            ranks[rid]["obj"].setdefault("section_title", d.get("section_title"))
        else:
            ranks[rid] = {"obj": d, "rrf": 0.0}
        ranks[rid]["rrf"] += 1.0 / (K + i + 1)

    out = [dict(obj, rrf_score=pack["rrf"]) for _, pack in ranks.items()]
    out.sort(key=lambda x: x["rrf_score"], reverse=True)
    return out[:k]


# === Endpoint 3: Unified ASK ===
@app.post("/ask")
def ask(req: AskRequest):
    q = (req.query or "").strip()
    k = max(1, min(req.top_k, TOPK_MAX))
    backend = (req.backend or "both").lower()

    sols, mils = [], []
    if backend in ("solr", "both"):
        sols = query_solr(QueryRequest(query=q, top_k=k))["results"]
    if backend in ("milvus", "both"):
        mils = query_milvus(QueryRequest(query=q, top_k=k))["results"]

    if backend == "solr":
        return {"backend": "solr", "query": q, "results": sols}
    if backend == "milvus":
        return {"backend": "milvus", "query": q, "results": mils}

    merged = rrf_merge(sols, mils, k)
    return {"backend": "both", "query": q, "results": merged}
