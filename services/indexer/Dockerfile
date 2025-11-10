FROM python:3.11-slim
WORKDIR /app

RUN pip install requests
RUN pip install pymilvus sentence-transformers

COPY ../../data /app/data
COPY index_corpus.py /app/
COPY index_milvus.py /app/


CMD ["python", "index_corpus.py",  "index_milvus.py"]
