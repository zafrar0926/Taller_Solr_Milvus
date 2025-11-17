import json
from demo_solr import query_solr
from demo_milvus import query_milvus

q = "¿Cómo participaron las empresas privadas en el conflicto armado?"

print("\n=================🔎 CONSULTA =================")
print(q)

print("\n================= 🔵 SOLR (BM25) ================")
solr_res = query_solr(q, 5)
print(json.dumps(solr_res, indent=2, ensure_ascii=False))

print("\n================= 🟢 MILVUS (Vectorial) =========")
milvus_res = query_milvus(q, 5)
print(json.dumps(milvus_res, indent=2, ensure_ascii=False))
