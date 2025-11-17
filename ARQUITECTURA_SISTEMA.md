# 🏗️ Arquitectura del Sistema RAG - Taller Solr vs Milvus

## 📋 Tabla de Contenidos
1. [Visión General](#visión-general)
2. [Componentes Principales](#componentes-principales)
3. [Flujo de Datos](#flujo-de-datos)
4. [Detalle Técnico por Servicio](#detalle-técnico-por-servicio)
5. [Pipeline de Indexación](#pipeline-de-indexación)
6. [Pipeline de Evaluación](#pipeline-de-evaluación)
7. [Infraestructura Docker](#infraestructura-docker)
8. [Decisiones de Diseño](#decisiones-de-diseño)

---

## 🎯 Visión General

Este sistema implementa un **RAG (Retrieval-Augmented Generation)** que compara dos estrategias complementarias de recuperación de documentos:

```
┌─────────────────────────────────────────────────────────┐
│                    USUARIO / API CLIENT                 │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │    FastAPI (8000)       │
        └────┬───────────┬────────┘
             │           │
      ┌──────▼─┐    ┌────▼──────┐
      │ SOLR   │    │  MILVUS   │
      │ (8983) │    │ (19530)   │
      └──────┬─┘    ┌────┬──────┘
             │      │    │
      ┌──────▼──────▼┐   │
      │  Corpus      │   │
      │  Indexado    │   │
      └──────────────┘   │
             BM25    Embeddings
             (léxico) (semántico)
```

**Dos paradigmas de búsqueda:**
- **Solr (BM25):** Recuperación léxica basada en frecuencia de términos
- **Milvus (Embeddings):** Recuperación semántica basada en similaridad vectorial

---

## 🧩 Componentes Principales

### 1. **API FastAPI** (`services/api/`)
- **Puerto:** 8000
- **Rol:** Punto de entrada unificado
- **Endpoints:**
  - `POST /query_solr` → Consulta Solr
  - `POST /query_milvus` → Consulta Milvus
  - `GET /health` → Verificación de salud

### 2. **Solr** (`services/solr/`)
- **Puerto:** 8983
- **Rol:** Motor de búsqueda léxica (BM25)
- **Core:** `rag_core`
- **Campo de indexación:** `text_raw`, `lemmas`, `section_title`
- **Volumen:** `./services/solr/data`

### 3. **Milvus** (`services/milvus/`)
- **Puerto:** 19530 (gRPC), 9091 (REST)
- **Rol:** Base de datos vectorial
- **Modelo:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Dependencias:**
  - **etcd** (2379): Coordinación distribuida
  - **MinIO** (9000): Almacenamiento de objetos (metadatos, logs)
- **Volúmenes:** `./services/milvus/data`, `./services/etcd/data`, `./services/minio/data`

### 4. **Indexadores** (`services/indexer/`)
- **index_corpus.py:** Carga el corpus en Solr
- **index_milvus.py:** Vectoriza y carga en Milvus

### 5. **Corpus** (`data/corpus/`)
- **Archivo:** `books_preprocessed_MWE.jsonl`
- **Formato:** JSONL (una línea por documento)
- **Campos:** `id`, `section_title`, `text_raw`, `lemmas`

---

## 🔄 Flujo de Datos

### **Fase 1: Inicialización (Setup)**
```
setup_rag.sh
    ├─ Crear estructura de carpetas
    ├─ Corregir permisos (Solr)
    ├─ Levantar servicios base
    │   ├─ Solr
    │   ├─ etcd
    │   ├─ MinIO
    │   └─ Milvus
    ├─ Crear core "rag_core" en Solr
    ├─ Configurar schema Solr
    ├─ Indexar corpus en Solr (via indexer)
    ├─ Vectorizar corpus en Milvus (via indexer_milvus)
    └─ Levantar API FastAPI
```

### **Fase 2: Consulta en Tiempo Real**
```
Cliente HTTP
    │
    ├─ POST /query_solr {"query": "¿qué es X?", "top_k": 10}
    │   └─ Solr BM25
    │       ├─ Tokenización
    │       ├─ Cálculo de TF-IDF
    │       └─ Ranking BM25
    │           └─ Retorna: [doc_id, score, content]
    │
    ├─ POST /query_milvus {"query": "¿qué es X?", "top_k": 10}
    │   └─ Milvus Embeddings
    │       ├─ Encutar query con modelo
    │       ├─ Búsqueda HNSW/IVF
    │       └─ Ranking por similaridad
    │           └─ Retorna: [doc_id, score, content]
    │
    └─ (Opcional) Procesamiento conjunto
        └─ Intersección / Unión de resultados
```

### **Fase 3: Evaluación**
```
Queries de prueba (queries_seed.txt)
    │
    ├─ Gold Estándar Débil (make_gold_agreement.py)
    │   ├─ Consultar ambos sistemas (TOP-10)
    │   ├─ Intersección → Documentos "altamente relevantes"
    │   └─ Unión - Intersección → Documentos "parcialmente relevantes"
    │       └─ Guardar: gold_weak.jsonl
    │
    ├─ Evaluación Recall (eval_*_recall.py)
    │   ├─ Ejecutar cada query contra Solr y Milvus
    │   └─ Calcular: Recall@5, Recall@10
    │
    ├─ Evaluación ROUGE-L (eval_*_rougeL.py)
    │   ├─ Recuperar TOP-1 documento
    │   ├─ Comparar con gold standard
    │   └─ Calcular: ROUGE-L (Precision, Recall, F1)
    │
    └─ Evaluación LLM Judge (eval_llm_judge_*.py)
        ├─ Recuperar TOP-1 documento
        ├─ Enviar a Gemini API para evaluación
        ├─ Calificar: Relevancia, Coherencia, Fidelidad (1-10)
        └─ Guardar resultados en CSV
```

---

## 🔧 Detalle Técnico por Servicio

### **Solr - Motor de Búsqueda Léxica**

#### Schema (Configuración de campos)
```json
{
  "fields": [
    {"name": "id", "type": "string", "stored": true, "indexed": true, "required": true},
    {"name": "section_title", "type": "text_general", "stored": true, "indexed": true},
    {"name": "text_raw", "type": "text_general", "stored": true, "indexed": true},
    {"name": "lemmas", "type": "text_general", "stored": true, "indexed": true}
  ]
}
```

#### Algoritmo BM25
```
score(d, q) = Σ(IDF(qi) * (f(qi, d) * (k1 + 1)) / (f(qi, d) + k1 * (1 - b + b * (|d| / avgdl))))

Donde:
- d = documento
- q = query
- f(qi, d) = frecuencia del término en el documento
- |d| = longitud del documento
- avgdl = longitud promedio de documentos
- k1, b = parámetros (típicamente k1=1.2, b=0.75)
```

#### Indexación
```python
# Archivo: services/indexer/index_corpus.py
with open(CORPUS_PATH) as f:
    for line in f:
        doc = json.loads(line)
        payload = {
            "id": doc["id"],
            "section_title": doc["section_title"],
            "text_raw": doc["text_raw"],
            "lemmas": doc["lemmas"]
        }
        requests.post(f"{SOLR_URL}/update", json=[payload])
```

---

### **Milvus - Base de Datos Vectorial**

#### Arquitectura de Dependencias
```
Milvus (Standalone)
├─ etcd (Coordinación)
│  └─ Almacena: Metadatos, información de colecciones
├─ MinIO (Object Storage)
│  └─ Almacena: Datos persistentes, logs de inserción
└─ RocksDB (Local Storage)
   └─ Almacena: Índices, datos temporales
```

#### Configuración de Colección
```python
# Archivo: services/indexer/index_milvus.py
collection_schema = CollectionSchema([
    FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=256, is_primary=True),
    FieldSchema(name="section_title", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="text_raw", dtype=DataType.VARCHAR, max_length=4096),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)
])

# Índice HNSW (Hierarchical Navigable Small World)
index_params = {
    "metric_type": "COSINE",
    "index_type": "HNSW",
    "params": {"M": 8, "efConstruction": 200}
}
```

#### Modelo de Embeddings
- **Modelo:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Dimensión:** 384 (embedding vector)
- **Idiomas:** Multilingüe (incluye español)
- **Ventaja:** Captura similitud semántica, no solo términos exactos

#### Búsqueda Vectorial
```python
# 1. Generar embedding de la query
query_embedding = model.encode(query, normalize_embeddings=True)

# 2. Búsqueda HNSW
results = collection.search(
    data=[query_embedding],
    anns_field="embedding",
    param={"metric_type": "COSINE", "params": {"ef": 64}},
    limit=10,
    output_fields=["id", "section_title", "text_raw"]
)

# 3. Retornar documentos ordenados por similaridad coseno
```

---

## 📥 Pipeline de Indexación

### **Paso 1: Preparación del Corpus**
```
Fuente: data/corpus/books_preprocessed_MWE.jsonl
Formato: 
{
  "id": "book_001_chap_02_sec_03",
  "section_title": "Introducción a la Física Cuántica",
  "text_raw": "La mecánica cuántica estudia el comportamiento...",
  "lemmas": "mecanica quantica estudiar comportamiento..."
}
```

### **Paso 2: Indexación Paralela**

#### **Rama A: Indexación Solr**
```bash
docker compose up --build --exit-code-from indexer indexer

# Internamente:
# 1. Lee CORPUS_PATH línea por línea
# 2. Extrae campos: id, section_title, text_raw, lemmas
# 3. POST /solr/rag_core/update con JSON array
# 4. Solr procesa:
#    - Tokenización
#    - Análisis de términos
#    - Cálculo de TF-IDF
#    - Construcción de índice invertido
# 5. Commit: /solr/rag_core/update?commit=true
```

#### **Rama B: Indexación Milvus**
```bash
docker compose up --build --exit-code-from indexer_milvus indexer_milvus

# Internamente:
# 1. Lee CORPUS_PATH línea por línea
# 2. Para cada documento:
#    a. Carga texto en modelo sentence-transformers
#    b. Genera embedding de 384 dimensiones
#    c. Normaliza embedding (coseno)
# 3. Inserta en Milvus:
#    - Vector embedding
#    - Metadatos: id, section_title, text_raw
# 4. Milvus construye:
#    - Índice HNSW (búsqueda jerárquica)
#    - Persistencia en MinIO
```

### **Paso 3: Verificación**
```bash
# Solr
curl "http://localhost:8983/solr/rag_core/select?q=*:*&rows=0"
# Retorna: "numFound": 12543

# Milvus
docker compose exec -T milvus python -c \
  "from milvus import connections; 
   connections.connect(host='milvus', port=19530);
   from milvus import Collection;
   c = Collection('documents');
   print(f'Total vectors: {c.num_entities}')"
```

---

## 📊 Pipeline de Evaluación

### **Arquitectura de Scripts de Evaluación**

```
eval_*.py (6 scripts paralelos)
├─ make_gold_agreement.py
│  ├─ Input: queries_seed.txt
│  ├─ Lógica: Acuerdo entre Solr + Milvus (TOP-10)
│  │   ├─ Intersección: highly relevant
│  │   └─ Unión - Inter: partially relevant
│  └─ Output: gold_weak.jsonl
│
├─ eval_solr_recall.py / eval_milvus_recall.py
│  ├─ Métrica: Recall@5, Recall@10
│  ├─ Cálculo: TP / (TP + FN)
│  │   - TP = documentos recuperados en TOP-K que están en gold
│  │   - FN = documentos en gold pero no recuperados
│  └─ Output: metrics_solr_recall.csv, metrics_milvus.csv
│
├─ eval_solr_rougeL.py / eval_rougeL.py
│  ├─ Métrica: ROUGE-L (Longest Common Subsequence)
│  ├─ Cálculo: 
│  │   - LCS = subsecuencia común más larga
│  │   - P = LCS / |retrieved|
│  │   - R = LCS / |gold|
│  │   - F1 = 2PR / (P + R)
│  └─ Output: metrics_solr_rougeL.csv, metrics_rougeL.csv
│
└─ eval_llm_judge_gemini.py / eval_llm_judge_solr.py
   ├─ Servicio: Google Gemini API
   ├─ Evaluación cualitativa:
   │   ├─ Relevancia (1-10): ¿Responde la pregunta?
   │   ├─ Coherencia (1-10): ¿Es el texto coherente?
   │   └─ Fidelidad (1-10): ¿Es factualmente correcto?
   └─ Output: metrics_llm_judge_*.csv
```

### **Métricas Detalladas**

#### **Recall@K**
```
Definición: Proporción de documentos relevantes recuperados en TOP-K

Recall@5 = |{documentos relevantes} ∩ {TOP-5 recuperados}| / |{documentos relevantes}|

Rango: [0, 1]
Interpretación:
  - 1.0 = Recuperó todos los relevantes en TOP-5
  - 0.5 = Recuperó 50% de los relevantes
  - 0.0 = No recuperó ninguno
```

#### **ROUGE-L (F1)**
```
Definición: Métrica que compara la subsecuencia común más larga

LCS(ref, hyp) = longest common subsequence length

Precision = LCS / len(hyp)
Recall = LCS / len(ref)
F1 = 2 * P * R / (P + R)

Rango: [0, 1]
Uso: Evaluar similitud entre documento recuperado y referencia
```

#### **LLM Judge (Gemini)**
```
Prompt para cada documento recuperado:

"Given the query: '{query}'
And the retrieved document: '{document}'

Rate on a scale 1-10:
1. Relevancia: ¿Responde directamente la pregunta?
2. Coherencia: ¿Es el texto bien estructurado y entendible?
3. Fidelidad: ¿Es factualmente correcto respecto al corpus?"

Salida: JSON
{
  "relevancia": 8,
  "coherencia": 9,
  "fidelidad": 7
}
```

### **Consolidación de Resultados**
```python
# exploracion_metricas.ipynb
# Carga todos los CSV y genera comparativas:

# 1. Boxplot ROUGE-L: Solr vs Milvus
# 2. Gráfico latencia por motor
# 3. Scorecard LLM: Promedios de relevancia/coherencia/fidelidad
# 4. Tabla resumen global

# Salida: resumen_global.csv, PNG comparativos
```

---

## 🐳 Infraestructura Docker

### **docker-compose.yml - Orquestación**

```yaml
# Servicios de Infraestructura (Milvus Stack)
etcd        → Coordinación distribuida (2379)
minio       → Object storage (9000, 9001)
minio_setup → Inicialización de buckets
milvus      → Base vectorial (19530, 9091)

# Servicios de Búsqueda
solr        → Motor léxico (8983)

# Servicios de Aplicación
indexer     → Script de indexación Solr
indexer_milvus → Script de indexación Milvus
api         → FastAPI unificada (8000)

# Red
rag_net (bridge) → Comunicación interna
```

### **Flujo de Dependencias**

```
Inicio:
solr ✓, etcd ✓, minio ✓
         ↓
    minio_setup ✓
         ↓
      milvus ✓
    /        \
indexer    indexer_milvus
  ↓            ↓
corpus en   corpus en
 Solr       Milvus
   \         /
    \       /
      api ✓
```

### **Volúmenes Persistentes**

| Servicio | Ruta Host | Ruta Container | Propósito |
|----------|-----------|-----------------|-----------|
| Solr | `./services/solr/data` | `/var/solr` | Índice BM25, configuración |
| Milvus | `./services/milvus/data` | `/var/lib/milvus` | Vectores, metadatos |
| etcd | `./services/etcd/data` | `/etcd-data` | Coordinación distribuida |
| MinIO | `./services/minio/data` | `/data` | Logs, metadatos |
| API | `./services/api` | `/app` | Código fuente, datos |
| Indexer | `./services/indexer` | `/app` | Scripts de indexación |
| Corpus | `./data` | `/app/data` | JSONL de documentos |

---

## 🎨 Decisiones de Diseño

### **1. Dual Retrieval (Solr + Milvus)**

**Razón:**
- **Solr (BM25):** Efectivo para queries exactas/léxicas, bajo overhead computacional
- **Milvus (Embeddings):** Captura similitud semántica, maneja variaciones léxicas

**Beneficio:** Comparación empírica de paradigmas de recuperación

### **2. Gold Estándar Débil (Weak Supervision)**

**Razón:**
- No disponían de anotadores humanos
- Ambos sistemas son _a priori_ válidos, pero diferentes

**Método:**
- **Intersección (Solr ∩ Milvus):** Alta confianza (ambos coinciden)
- **Unión - Intersección:** Baja confianza (solo uno lo encuentra)

**Limitación:** Pueden perder documentos relevantes que solo uno de los sistemas recupera

### **3. Modelo Multilingüe (sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)**

**Razón:**
- Corpus es en español
- Modelo preentrenado en 50+ idiomas
- Dimensión reducida (384) → bajo costo computacional
- Eficiente para dispositivos con recursos limitados

**Alternativas descartadas:**
- BERT/RoBERTa base (no multilingüe, dimensión 768)
- GPT embeddings (API externa, costo monetario)

### **4. Índice HNSW en Milvus**

**Razón:**
- Mejor relación velocidad/precisión que IVF
- Jerárquico (escalable)
- Bajo overhead de memoria

**Alternativas:**
- IVF_FLAT: Más rápido pero menos preciso
- BRUTE_FORCE: Más preciso pero O(n) en búsqueda

### **5. Métricas Múltiples (Recall, ROUGE-L, LLM Judge)**

**Razón:**
- **Recall:** Mide cobertura cuantitativa
- **ROUGE-L:** Mide similitud textual automáticamente
- **LLM Judge:** Evaluación cualitativa humana simulada

**Complementariedad:** Capturan diferentes aspectos de la recuperación

### **6. Pipeline Secuencial pero Modulable**

```
setup_rag.sh
  ├─ Infraestructura
  ├─ Indexación Solr
  ├─ Indexación Milvus
  └─ API

scripts/eval_*.py (independientes)
  ├─ Pueden correr en paralelo
  ├─ Salida: CSV individuales
  └─ Consolidación en notebook
```

**Ventaja:** Fácil de depurar y reutilizar componentes

---

## 📈 Diagrama de Componentes (C4 Model)

### **Nivel 1: Contexto**
```
┌─────────────────────────────────────────────┐
│          Usuario / Investigador             │
│  (Consulta documento, ejecuta evaluaciones) │
└────────────┬────────────────────────────────┘
             │
┌────────────▼────────────────────────────────┐
│        Sistema RAG (Contenedor)             │
│  Comparación Solr vs Milvus + Evaluación    │
└─────────────────────────────────────────────┘
```

### **Nivel 2: Contenedor**
```
Sistema RAG
├─ API FastAPI
│  ├─ Query Solr
│  └─ Query Milvus
├─ Solr (BM25)
│  └─ Corpus Indexado (léxico)
├─ Milvus (Embeddings)
│  ├─ etcd
│  ├─ MinIO
│  └─ Corpus Indexado (vectorial)
└─ Evaluación (scripts)
   ├─ Gold Generation
   ├─ Recall
   ├─ ROUGE-L
   └─ LLM Judge
```

### **Nivel 3: Componentes**
```
[API]─────┬──────[Solr]──[Índice BM25]
          │
          └──────[Milvus]─┬──[etcd]
                          ├──[MinIO]
                          └──[Índice HNSW]

[Corpus]──[Indexer Solr]──[Solr]
          [Indexer Milvus]─[Milvus]

[Queries]─[eval_*.py]─[Gold/Métricas]
```

---

## 🚀 Flujo de Ejecución Completo

```
1. setup_rag.sh
   ├─ Crear estructura
   ├─ Levantar Docker Compose
   ├─ Esperar servicios saludables
   ├─ Indexar corpus (Solr + Milvus)
   └─ Levantar API

2. make_gold_agreement.py
   ├─ Leer queries_seed.txt
   ├─ Consultar Solr + Milvus (TOP-10)
   ├─ Calcular intersección/unión
   └─ Guardar gold_weak.jsonl

3. Evaluación (scripts paralelos)
   ├─ eval_solr_recall.py → metrics_solr_recall.csv
   ├─ eval_milvus_recall.py → metrics_milvus.csv
   ├─ eval_solr_rougeL.py → metrics_solr_rougeL.csv
   ├─ eval_rougeL.py → metrics_rougeL.csv
   ├─ eval_llm_judge_solr.py → metrics_llm_judge_solr.csv
   └─ eval_llm_judge_gemini_fixed.py → metrics_llm_judge_gemini_fixed.csv

4. exploracion_metricas.ipynb
   ├─ Cargar todos los CSV
   ├─ Generar visualizaciones
   ├─ Comparativas Solr vs Milvus
   └─ Conclusiones
```

---

## 📝 Estructura de Directorios Completa

```
Taller_RAG/
├── docker-compose.yml          # Orquestación de servicios
├── setup_rag.sh               # Script de inicialización
├── README.md
│
├── data/
│   ├── corpus/
│   │   └── books_preprocessed_MWE.jsonl  # Corpus (12k+ docs)
│   ├── queries_seed.txt       # Queries de evaluación
│   └── gold_weak.jsonl        # Gold estándar (generado)
│
├── scripts/
│   ├── make_gold_agreement.py # Genera gold débil
│   ├── eval_*.py              # 6 scripts de evaluación
│   ├── exploracion_metricas.ipynb  # Análisis y visualización
│   └── eval_log.txt           # Log de ejecución
│
├── services/
│   ├── api/
│   │   ├── Dockerfile         # Image FastAPI
│   │   ├── main.py           # Código de la API
│   │   ├── requirements.txt
│   │   └── data/
│   │
│   ├── indexer/
│   │   ├── Dockerfile         # Image indexación
│   │   ├── index_corpus.py   # Indexación Solr
│   │   ├── index_milvus.py   # Indexación Milvus
│   │   ├── requirements.txt
│   │   └── data/
│   │
│   ├── solr/
│   │   └── data/              # Volumen persistente Solr
│   │
│   ├── milvus/
│   │   ├── data/              # Volumen persistente Milvus
│   │   ├── etcd/
│   │   └── minio/
│   │
│   ├── etcd/
│   │   └── data/              # Volumen persistente etcd
│   │
│   └── minio/
│       └── data/              # Volumen persistente MinIO
│
└── reports/
    ├── metrics_*.csv          # Resultados de evaluación
    ├── resumen_global.csv     # Consolidado
    └── *.png                  # Gráficos comparativos
```

---

## 🔍 Puntos Clave de Integración

### **API - Solr**
```
POST /query_solr
Request: {"query": str, "top_k": int}
Response: [{"id": str, "score": float, "content": str}, ...]
Conexión: HTTP REST a http://solr:8983/solr/rag_core
```

### **API - Milvus**
```
POST /query_milvus
Request: {"query": str, "top_k": int}
Response: [{"id": str, "score": float, "content": str}, ...]
Conexión: gRPC a milvus:19530
```

### **Indexación - Corpus**
```
Fuente: ./data/corpus/books_preprocessed_MWE.jsonl
Lectura: JSONL línea por línea
Destino Solr: POST /solr/rag_core/update
Destino Milvus: Vector DB + Metadata storage
```

---

## 🎓 Conclusión Arquitectónica

Este sistema es una **comparación empírica rigurosa** entre dos paradigmas de recuperación de información:

| Aspecto | Solr (BM25) | Milvus (Embeddings) |
|---------|-------------|-------------------|
| **Enfoque** | Léxico (términos exactos) | Semántico (significado) |
| **Algoritmo** | TF-IDF + BM25 | Embeddings + HNSW |
| **Overhead** | Bajo | Medio (GPU opcional) |
| **Precisión léxica** | Alta | Media |
| **Captura semántica** | Baja | Alta |
| **Escalabilidad** | Muy buena | Buena |

**Conclusión esperada:** En tareas de preguntas abiertas, Milvus capturará mejor la similitud semántica, mientras que Solr será más preciso en queries estructuradas.

