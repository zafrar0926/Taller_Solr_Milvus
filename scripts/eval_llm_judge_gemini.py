# ============================================
# 🤖 Evaluación Cualitativa con Gemini (versión robusta + gráficos)
# ============================================
import os, json, re, requests, time
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import google.generativeai as genai

# --- Configurar Gemini ---
genai.configure(api_key="AIzaSyA04JNJ6-uxgTH76kxnbXIP0E0GM8-pHBU")
llm = genai.GenerativeModel("gemini-2.5-flash-lite")

# --- Configuración de rutas ---
API_URL = "http://localhost:8000/query_milvus"
GOLD_PATH = Path("/home/zafrar09/Taller_RAG/data/gold_weak.jsonl")
REPORTS_DIR = Path("/home/zafrar09/Taller_RAG/reports")
REPORTS_DIR.mkdir(exist_ok=True)

# --- Prompt de evaluación ---
PROMPT_TEMPLATE = """
Responde **solo** en JSON válido.
Evalúa la calidad del texto recuperado en tres dimensiones (1 a 10):

Pregunta:
{query}

Texto recuperado:
\"\"\"{retrieved_text}\"\"\"

Devuelve este JSON exacto:
{{
  "relevancia": número,
  "coherencia": número,
  "fidelidad": número,
  "comentario": "una frase breve que resuma tu juicio"
}}
"""

# --- Funciones auxiliares ---
def ask_gemini(prompt):
    try:
        r = llm.generate_content(prompt)
        return r.text.strip()
    except Exception as e:
        return f"ERROR: {e}"

def extract_json(text):
    """Intenta extraer JSON válido incluso si Gemini devuelve texto extra."""
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                pass
        return {"relevancia": None, "coherencia": None, "fidelidad": None, "comentario": text[:200]}

# --- Cargar gold estándar ---
with open(GOLD_PATH, encoding="utf-8") as f:
    gold_data = [json.loads(line) for line in f]

results = []

# --- Evaluación ---
for entry in tqdm(gold_data, desc="Evaluando con Gemini (robusto)"):
    query = entry["query"]

    try:
        r = requests.post(API_URL, json={"query": query, "top_k": 3}, timeout=20)
        r.raise_for_status()
        retrieved = r.json().get("results", [])
        retrieved_text = " ".join([d.get("text_raw", "") for d in retrieved])
    except Exception as e:
        print(f"⚠️ Error al consultar '{query[:50]}': {e}")
        continue

    prompt = PROMPT_TEMPLATE.format(query=query, retrieved_text=retrieved_text[:4000])
    llm_output = ask_gemini(prompt)
    parsed = extract_json(llm_output)
    parsed["query"] = query
    results.append(parsed)

df = pd.DataFrame(results)

# --- Conversión numérica ---
for col in ["relevancia", "coherencia", "fidelidad"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# --- Guardar CSV ---
df.to_csv(REPORTS_DIR / "metrics_llm_judge_gemini_fixed.csv", index=False)
print("\n✅ Evaluación LLM (Gemini) guardada en /reports/metrics_llm_judge_gemini_fixed.csv")

# --- Estadísticas generales ---
if not df.empty:
    print("\n📊 Promedios generales:")
    print(df[["relevancia", "coherencia", "fidelidad"]].mean().round(2))
else:
    print("⚠️ No se generaron resultados válidos.")

# --- 📈 Visualización comparativa ---
if not df.empty:
    plt.figure(figsize=(10, 6))
    bar_width = 0.25
    x = range(len(df))

    plt.barh([i - bar_width for i in x], df["relevancia"], bar_width, label="Relevancia", color="#4e79a7")
    plt.barh(x, df["coherencia"], bar_width, label="Coherencia", color="#f28e2b")
    plt.barh([i + bar_width for i in x], df["fidelidad"], bar_width, label="Fidelidad", color="#59a14f")

    plt.yticks(x, [q[:70] + ("…" if len(q) > 70 else "") for q in df["query"]])
    plt.xlabel("Puntuación (1–10)")
    plt.title("Evaluación Cualitativa - Gemini (Relevancia, Coherencia, Fidelidad)")
    plt.legend()
    plt.xlim(0, 10)
    plt.tight_layout()

    plt.savefig(REPORTS_DIR / "metrics_llm_judge_gemini_fixed.png", dpi=150)
    plt.show()
    print(f"\n📁 Gráfico guardado en {REPORTS_DIR / 'metrics_llm_judge_gemini_fixed.png'}")
