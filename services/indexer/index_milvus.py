# services/indexer/index_milvus.py
import os, json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from pymilvus import (
    connections, utility, FieldSchema, CollectionSchema,
    DataType, Collection
)

# --- Config ---
MILVUS_HOST = os.getenv("MILVUS_HOST", "milvus")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_corpus")
CORPUS_PATH = Path("/app/data/corpus/books_preprocessed_MWE.jsonl")
MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
DIM = 384  # all-MiniLM-L6-v2

# Límites del esquema (coherentes con Milvus VARCHAR)
MAX_ID = 128
MAX_TITLE = 512
MAX_TEXT = 8192

def clip(s: str, max_len: int) -> str:
    if s is None:
        return ""
    s = str(s)
    if len(s) > max_len:
        # Log corto para saber que recortamos (sin saturar stdout)
        print(f"⚠️  Truncado a {max_len} chars (len={len(s)})")
        return s[:max_len]
    return s

print("🚀 Conectando a Milvus...")
connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

# 1) Crear colección si no existe
if not utility.has_collection(COLLECTION_NAME):
    print(f"📦 Creando colección {COLLECTION_NAME}...")
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=MAX_ID),
        FieldSchema(name="section_title", dtype=DataType.VARCHAR, max_length=MAX_TITLE),
        FieldSchema(name="text_raw", dtype=DataType.VARCHAR, max_length=MAX_TEXT),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIM),
    ]
    schema = CollectionSchema(fields, description="RAG corpus (MiniLM-L6)")
    coll = Collection(name=COLLECTION_NAME, schema=schema)
else:
    print(f"✅ Colección existente: {COLLECTION_NAME}")
    coll = Collection(COLLECTION_NAME)

# Crear índice si no existe aún (AUTOINDEX + COSINE)
if not coll.indexes:
    print("🔧 Creando índice AUTOINDEX/COSINE…")
    coll.create_index(
        field_name="embedding",
        index_params={"index_type": "AUTOINDEX", "metric_type": "COSINE", "params": {}},
    )

# 2) Cargar corpus
if not CORPUS_PATH.exists():
    raise FileNotFoundError(f"No se encontró el corpus en {CORPUS_PATH}")
def utf8_truncate(s: str, max_bytes: int) -> str:
    if s is None:
        return ""
    b = s.encode("utf-8")
    if len(b) <= max_bytes:
        return s
    return b[:max_bytes].decode("utf-8", errors="ignore")

docs = []
with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        docs.append({
            "id": (d["section_id"] or "")[:128],
            "section_title": utf8_truncate(d.get("section_title") or "", 512),   # 512 bytes
            "text_raw": utf8_truncate(d.get("text_raw") or "", 8192),            # 8192 bytes
        })

print(f"📄 Documentos a indexar en Milvus: {len(docs)}")

# 3) Embeddings + insert por lotes
model = SentenceTransformer(MODEL_NAME)
BATCH = 128

# (Opcional) cargar colección antes de operaciones intensivas
coll.load()

for i in range(0, len(docs), BATCH):
    batch = docs[i:i+BATCH]
    # Para embeddings: si text_raw está vacío, caemos al título
    texts = [b["text_raw"] if b["text_raw"] else b["section_title"] for b in batch]
    embs = model.encode(texts, normalize_embeddings=True).tolist()

    entities = [
        [b["id"] for b in batch],
        [b["section_title"] for b in batch],
        [b["text_raw"] for b in batch],
        embs,
    ]

    try:
        coll.insert(entities)
        print(f"✅ Inserción Milvus lote {i//BATCH+1} ({len(batch)} docs)")
    except Exception as e:
        # Si algún registro viola el esquema, log útil para depuración
        print(f"❌ Error en lote {i//BATCH+1}: {e}")
        # Opcional: intentar inserción doc a doc para identificar el problemático
        for j, one in enumerate(zip(*entities)):
            try:
                coll.insert([[one[0]], [one[1]], [one[2]], [one[3]]])
            except Exception as e1:
                print(f"   ↳ Falló doc #{j} del lote (id={one[0]}): {e1}")

# Sincroniza segmentos a disco
coll.flush()
coll.release()
print("🎯 Indexación Milvus completada.")
