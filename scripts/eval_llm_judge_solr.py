#!/usr/bin/env python3
# ===============================================================
# 🤖 Evaluación cualitativa (LLM-as-a-Judge) para Solr (Gemini)
# ===============================================================
import os, json, re, time
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import google.generativeai as genai
import requests

# --- Configuración ---
API_URL = "http://localhost:8000/query_solr"
GOLD_PATH = Path("/home/zafrar09/Taller_RAG/data/gold_weak.jsonl")
REPORTS_DIR = Path("/home/zafrar09/Taller_RAG/reports")
REPORTS_DIR.mkdir(exist_ok=True)

# --- Gemini ---
genai.configure(api_key="AIzaSyA04JNJ6-uxgTH76kxnbXIP0E0GM8-pHBU")
llm = genai.GenerativeModel("gemini-2.5-flash-lite")

# --- Cargar datos ---
with open(GOLD_PATH, encoding="utf-8") as f:
    gold_data = [json.loads(line) for line in f]

results = []

for entry in tqdm(gold_data, desc="Evaluando con Gemini (Solr)"):
    query = entry["query"]
    try:
        r = requests.post(API_URL, json={"query": query, "top_k": 3}, timeout=20)
        r.raise_for_status()
        retrieved = r.json().get("results", [])
    except Exception as e:
        print(f"⚠️ Error en '{query[:40]}': {e}")
        continue

    retrieved_text = " ".join(
        " ".join(doc["text_raw"]) if isinstance(doc.get("text_raw"), list) else str(doc.get("text_raw", ""))
        for doc in retrieved
    )

    prompt = f"""
    Eres un evaluador experto de sistemas de recuperación de información.
    Evalúa el texto recuperado por Solr frente a la pregunta, asignando calificaciones del 0 al 5.

    Pregunta:
    {query}

    Respuesta recuperada:
    {retrieved_text[:3000]}

    Devuelve ÚNICAMENTE un JSON con el formato:
    {{
      "relevancia": <número>,
      "coherencia": <número>,
      "fidelidad": <número>
    }}
    """

    try:
        response = llm.generate_content(prompt)
        raw = response.text.strip()

        # --- Intento directo JSON ---
        try:
            score = json.loads(raw)
        except:
            # --- Regex de rescate si Gemini respondió en texto libre ---
            nums = re.findall(r"(\d+(?:\.\d+)?)", raw)
            score = {
                "relevancia": float(nums[0]) if len(nums) > 0 else 0.0,
                "coherencia": float(nums[1]) if len(nums) > 1 else 0.0,
                "fidelidad": float(nums[2]) if len(nums) > 2 else 0.0,
            }

    except Exception as e:
        print(f"⚠️ Error en LLM: {e}")
        score = {"relevancia": 0.0, "coherencia": 0.0, "fidelidad": 0.0}

    score["query"] = query
    results.append(score)

# --- Guardar resultados ---
df = pd.DataFrame(results)
csv_path = REPORTS_DIR / "metrics_llm_judge_solr.csv"
df.to_csv(csv_path, index=False)

print(f"\n✅ Evaluación LLM (Gemini) completada y guardada en {csv_path}")
print(df.head())

# --- Estadísticas ---
for col in ["relevancia", "coherencia", "fidelidad"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("\n📊 Promedios generales:")
print(df[["relevancia", "coherencia", "fidelidad"]].mean().round(2))
