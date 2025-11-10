#!/usr/bin/env bash
# ============================================================
# 🚀 Lanzador completo del entorno RAG (Solr + Milvus + API)
# ============================================================
set -euo pipefail

echo "🚀 Iniciando entorno RAG..."
BASE_DIR="$(pwd)"

# --- Helpers ---
wait_for_http() {
  local url="$1" ; local what="${2:-service}"
  echo "⏳ Esperando ${what} en ${url} ..."
  until curl -sf "${url}" >/dev/null; do
    sleep 3
  done
  echo "✅ ${what} OK"
}

wait_for_text() {
  local url="$1" ; local text="$2" ; local what="${3:-service}"
  echo "⏳ Esperando ${what} (${text}) en ${url} ..."
  until curl -sf "${url}" | grep -qi "${text}"; do
    sleep 3
  done
  echo "✅ ${what} OK"
}

# --- 1) Carpetas ---
echo "📂 Verificando estructura..."
mkdir -p "$BASE_DIR/data/corpus"
mkdir -p "$BASE_DIR/services/solr/data"
mkdir -p "$BASE_DIR/services/milvus/data"
mkdir -p "$BASE_DIR/services/api"
mkdir -p "$BASE_DIR/services/indexer"
mkdir -p "$BASE_DIR/reports"

# --- 2) Permisos ---
echo "🔧 Corrigiendo permisos de Solr..."
sudo chown -R 8983:8983 "$BASE_DIR/services/solr/data" 2>/dev/null || true
sudo chmod -R 775 "$BASE_DIR/services/solr/data" || true

# --- 3) Levantar servicios base ---
echo "🐳 Levantando Solr + Milvus..."
docker compose up -d solr etcd minio minio_setup milvus

# --- 4) Verificar salud ---
wait_for_http "http://localhost:8983/solr/" "Solr UI"
wait_for_text "http://localhost:9091/healthz" "ok" "Milvus REST health"

# --- 5) Crear/verificar core rag_core ---
echo "🧱 Creando/verificando core 'rag_core'..."
HAS_CORE=$(curl -sf "http://localhost:8983/solr/admin/cores?action=STATUS&core=rag_core" | grep -c "\"name\":\"rag_core\"" || true)
if [[ "$HAS_CORE" -eq 0 ]]; then
  curl -sf "http://localhost:8983/solr/admin/cores?action=CREATE&name=rag_core" >/dev/null
  echo "✅ Core 'rag_core' creado."
else
  echo "ℹ️  Core 'rag_core' ya existe."
fi

# --- 6) Schema ---
echo "🧬 Asegurando campos de schema en Solr..."
SCHEMA_URL="http://localhost:8983/solr/rag_core/schema"
add_field() {
  local payload="$1"
  curl -s -X POST -H 'Content-type:application/json' \
       --data "{\"add-field\":${payload}}" \
       "${SCHEMA_URL}" >/dev/null || true
}
add_field '{"name":"id","type":"string","stored":true,"indexed":true,"required":true}'
add_field '{"name":"section_title","type":"text_general","stored":true,"indexed":true}'
add_field '{"name":"text_raw","type":"text_general","stored":true,"indexed":true}'
add_field '{"name":"lemmas","type":"text_general","stored":true,"indexed":true}'
echo "✅ Campos verificados en Solr."

# --- 7) Indexar en Solr ---
echo "📥 Indexando corpus en Solr..."
docker compose up --build --exit-code-from indexer indexer

# --- 8) Indexar en Milvus ---
echo "📥 Indexando corpus en Milvus..."
docker compose up --build --exit-code-from indexer_milvus indexer_milvus

# --- 9) Levantar API ---
echo "🌐 Levantando API..."
docker compose up -d --build api
wait_for_http "http://localhost:8000/health" "FastAPI RAG"

# --- 10) Verificación final ---
echo "🔎 Chequeo final:"
SOLR_DOCS=$(curl -s "http://localhost:8983/solr/rag_core/select?q=*:*&rows=0" | sed -n 's/.*"numFound":\([0-9]\+\).*/\1/p' || true)
echo "   • Solr numFound: ${SOLR_DOCS:-desconocido}"
echo "   • Milvus colección: $(docker compose exec -T milvus curl -s http://localhost:9091/healthz | head -c 50)"

echo "📋 Contenedores activos:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo "🎯 Entorno RAG listo."
echo "➡️ Solr UI : http://localhost:8983"
echo "➡️ API FastAPI : http://localhost:8000"
