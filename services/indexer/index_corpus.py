import json, requests, os, time
from pathlib import Path
from requests.exceptions import RequestException

# === Config por entorno (con valores por defecto) ===
BASE_DIR = Path(os.getenv("CORPUS_DIR", "/app/data/corpus"))
CORPUS_PATH = Path(os.getenv("CORPUS_PATH", str(BASE_DIR / "books_preprocessed_MWE.jsonl")))
SOLR_BASE = os.getenv("SOLR_BASE", "http://solr:8983/solr/rag_core")
SOLR_UPDATE = f"{SOLR_BASE}/update?commit=true"
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "50"))
RETRIES = int(os.getenv("SOLR_RETRIES", "20"))
SLEEP_S = float(os.getenv("SOLR_SLEEP", "1.5"))

print("🚀 Iniciando indexación del corpus en Solr...")
if not CORPUS_PATH.exists():
    raise FileNotFoundError(f"No se encontró el corpus en {CORPUS_PATH}")

# === Cargar documentos ===
with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    docs = [json.loads(line) for line in f]

# === Transformar formato para Solr ===
solr_docs = []
for d in docs:
    solr_docs.append({
        "id": d["section_id"],
        "section_title": d.get("section_title", ""),
        "text_raw": d.get("text_raw", ""),
        "lemmas": " ".join(d.get("lemmas", []))
    })

print(f"📄 Documentos a indexar: {len(solr_docs)}")

# === Ping simple a Solr con reintentos (por si el core tarda) ===
for attempt in range(1, RETRIES + 1):
    try:
        r = requests.get(f"{SOLR_BASE}/select?q=*:*&rows=0", timeout=3)
        if r.ok:
            break
    except RequestException:
        pass
    print(f"⏳ Esperando Solr… intento {attempt}/{RETRIES}")
    time.sleep(SLEEP_S)
else:
    raise RuntimeError("Solr no respondió a tiempo. Revisa el core/servicio.")

# === Enviar en lotes ===
for i in range(0, len(solr_docs), BATCH_SIZE):
    batch = solr_docs[i:i + BATCH_SIZE]
    try:
        resp = requests.post(SOLR_UPDATE, json=batch, timeout=10)
        if resp.status_code != 200:
            print(f"⚠️ Error en lote {i//BATCH_SIZE}: {resp.status_code} {resp.text[:300]}")
        else:
            print(f"✅ Lote {i//BATCH_SIZE + 1} indexado ({len(batch)} docs)")
    except RequestException as e:
        print(f"⚠️ Error de conexión en lote {i//BATCH_SIZE}: {e}")

print("🎯 Indexación completada.")
