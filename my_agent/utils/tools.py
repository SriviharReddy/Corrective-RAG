"""
Shared tools and indexes for the Corrective-RAG agent.

Everything here is initialised **once** when the module is imported
(i.e. at server startup) so subsequent requests pay no loading cost.

Vector store
------------
Uses Chroma with a persistent directory (``chroma_db/`` at the project root).
On first run it builds the index from every PDF in ``documents/`` and persists
it to disk.  On subsequent starts it loads the existing collection — no rebuild.

To force a full re-index (e.g. after adding new PDFs) delete the ``chroma_db/``
directory and restart the server.

Exports
-------
retriever   – Chroma-backed LangChain retriever over all PDFs in documents/
tavily      – TavilySearch tool (uses TAVILY_API_KEY from env)
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.documents import Document
from langchain_tavily import TavilySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
_ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DOCS_DIR    = os.path.join(_ROOT, "documents")
CHROMA_DIR  = os.path.join(_ROOT, "chroma_db")
COLLECTION  = "corrective_rag"

# ── Embeddings (shared between build and load paths) ──────────────────────
embeddings = FastEmbedEmbeddings()


# ── PDF loading & chunking ─────────────────────────────────────────────────

def _load_docs() -> list[Document]:
    print(f"  Loading PDF documents from {DOCS_DIR} using DirectoryLoader + PyMuPDFLoader...")
    loader = DirectoryLoader(
        DOCS_DIR,
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader
    )
    docs = loader.load()
    if not docs:
        raise FileNotFoundError(
            f"No PDF files found in '{DOCS_DIR}'. "
            "Add at least one PDF before starting the server."
        )
    return docs



def _build_index() -> Chroma:
    """Load PDFs, chunk them, embed, and persist a new Chroma collection."""
    print("Building vector index from PDFs...")
    raw_docs = _load_docs()

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=900, chunk_overlap=150
    ).split_documents(raw_docs)

    for chunk in chunks:
        chunk.page_content = (
            chunk.page_content.encode("utf-8", "ignore").decode("utf-8", "ignore")
        )

    print(f"  Embedding {len(chunks)} chunks -> persisting to {CHROMA_DIR}")
    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION,
    )
    print("  Index built and persisted.")
    return store


def _load_or_build() -> Chroma:
    """Return the persisted Chroma store, building it from scratch if needed."""
    store = Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION,
    )
    count = store._collection.count()  # number of stored vectors
    if count > 0:
        print(f"Loaded existing Chroma index ({count} chunks) from {CHROMA_DIR}")
        return store

    # Collection exists but is empty (first run or was cleared)
    return _build_index()


# ── Vector store & retriever ───────────────────────────────────────────────
vector_store = _load_or_build()
retriever = vector_store.as_retriever(
    search_type="similarity", search_kwargs={"k": 4}
)

# ── Web search tool ────────────────────────────────────────────────────────
tavily = TavilySearch(max_results=5)
