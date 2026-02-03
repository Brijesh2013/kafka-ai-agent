import os
from openai import OpenAI
import chromadb
from dotenv import load_dotenv

load_dotenv()

# Read API key from environment (loaded from .env by python-dotenv)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set. Set it in your environment or in a .env file (do not commit secrets to source control).")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

CHROMA_HOST = os.getenv("CHROMA_HOST", "127.0.0.1")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION_NAME = "kafka-docs"
OPENAI_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


class _Doc:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}


def get_retriever(k=3):
    """Return a simple retriever object with `get_relevant_documents(query)` method."""
    # CHROMA_MODE: 'http' (default) to connect to a Chroma server, or 'local' to use an
    # in-process Chroma client (no server required).
    CHROMA_MODE = os.getenv("CHROMA_MODE", "http").lower()

    client = None
    if CHROMA_MODE == "local":
        # Use local in-process client (persists to disk without running a separate server)
        client = chromadb.Client()
    else:
        # Try HTTP client first, fall back to local client on failure
        try:
            client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        except Exception as e:
            # Fall back to a local in-process client and warn the user
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                "Could not connect to Chroma HTTP server at %s:%s (%s). Falling back to local in-process Chroma client.",
                CHROMA_HOST,
                CHROMA_PORT,
                e,
            )
            client = chromadb.Client()

    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception:
        collection = client.create_collection(name=COLLECTION_NAME)

    class Retriever:
        def __init__(self, collection, k):
            self.collection = collection
            self.k = k

        def get_relevant_documents(self, query: str):
            # compute query embedding via OpenAI client
            resp = openai_client.embeddings.create(model=OPENAI_MODEL, input=[query])
            q_emb = resp.data[0].embedding

            res = self.collection.query(query_embeddings=[q_emb], n_results=self.k, include=["documents", "metadatas"])

            docs = []
            # res fields are lists per query; we use first query result
            docs_list = res.get("documents", [])
            metas_list = res.get("metadatas", [])

            if docs_list:
                for i, d in enumerate(docs_list[0]):
                    meta = metas_list[0][i] if metas_list and metas_list[0] else {}
                    docs.append(_Doc(page_content=d, metadata=meta))

            return docs

    return Retriever(collection=collection, k=k)
