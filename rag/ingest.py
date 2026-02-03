import os
from openai import OpenAI
import chromadb
from dotenv import load_dotenv

load_dotenv()

openai_client = OpenAI()

CHROMA_HOST = os.getenv("CHROMA_HOST", "chroma")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION_NAME = "kafka-docs"
DATA_PATH = "data/kafka_docs"
OPENAI_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


def _read_text_files(path):
    docs = []

    for file in os.listdir(path):
        if file.endswith(".txt"):
            with open(os.path.join(path, file), "r", encoding="utf-8") as f:
                text = f.read()
            docs.append({"id": file, "text": text})

    return docs


def _chunk_text(text, chunk_size=500, chunk_overlap=50):
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks


def _embed_texts(texts):
    # OpenAI Embeddings API using the new client
    resp = openai_client.embeddings.create(model=OPENAI_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def ingest_docs():
    docs = _read_text_files(DATA_PATH)

    all_texts = []
    ids = []
    metadatas = []

    for doc in docs:
        chunks = _chunk_text(doc["text"], chunk_size=500, chunk_overlap=50)
        for i, c in enumerate(chunks):
            all_texts.append(c)
            ids.append(f"{doc['id']}_chunk_{i}")
            metadatas.append({"source": doc["id"], "chunk": i})

    # Batch embeddings to avoid too-large requests
    embeddings = []
    batch_size = 1000
    for i in range(0, len(all_texts), batch_size):
        batch = all_texts[i : i + batch_size]
        embeddings.extend(_embed_texts(batch))

    chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

    try:
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
    except Exception:
        collection = chroma_client.create_collection(name=COLLECTION_NAME)

    collection.add(ids=ids, documents=all_texts, metadatas=metadatas, embeddings=embeddings)

    print("✅ Kafka docs ingested into Chroma (Docker)")


if __name__ == "__main__":
    ingest_docs()
