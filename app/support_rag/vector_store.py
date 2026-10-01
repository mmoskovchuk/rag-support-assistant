"""ChromaDB storage with Hugging Face embeddings, wrapped by LangChain."""

import logging
from collections.abc import Iterable
from functools import lru_cache

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

from .config import Settings

logger = logging.getLogger(__name__)


@lru_cache
def get_embeddings(model_name: str, device: str) -> Embeddings:
    # Imported lazily: it pulls in torch, which takes seconds and lots of RAM.
    from langchain_huggingface import HuggingFaceEmbeddings

    logger.info("Loading embedding model %s on %s", model_name, device)
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        # Unit-length vectors make cosine distance well defined and comparable.
        encode_kwargs={"normalize_embeddings": True},
    )


@retry(
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=1, max=15),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _connect(host: str, port: int) -> chromadb.ClientAPI:
    # ChromaDB may still be starting when this service boots, hence the retries.
    client = chromadb.HttpClient(host=host, port=port, settings=ChromaSettings(anonymized_telemetry=False))
    client.heartbeat()
    logger.info("Connected to ChromaDB at %s:%d", host, port)
    return client


class VectorStore:
    def __init__(self, settings: Settings) -> None:
        self._name = settings.chroma_collection
        self._embeddings = get_embeddings(settings.embedding_model, settings.embedding_device)
        self._client = _connect(settings.chroma_host, settings.chroma_port)
        self._lc, self._collection = self._open(self._name)

    def _open(self, name: str) -> tuple[Chroma, chromadb.Collection]:
        lc = Chroma(
            client=self._client,
            collection_name=name,
            embedding_function=self._embeddings,
            collection_configuration={"hnsw": {"space": "cosine"}},
        )
        # Raw collection handle for metadata-only operations (no embeddings needed).
        return lc, self._client.get_collection(name)

    def heartbeat(self) -> None:
        self._client.heartbeat()

    def has_document(self, doc_id: str) -> bool:
        return bool(self._collection.get(where={"doc_id": doc_id}, limit=1, include=[])["ids"])

    def upsert(self, documents: list[Document], ids: list[str]) -> None:
        # langchain-chroma uses collection.upsert(), so retries are idempotent.
        self._lc.add_documents(documents, ids=ids)

    def delete_other_versions(self, source: str, keep_doc_id: str) -> int:
        """Remove chunks of older versions of the same file name.

        Called only after the new version is stored, so the knowledge base is
        never left without a copy of the document.
        """
        where = {"$and": [{"source": source}, {"doc_id": {"$ne": keep_doc_id}}]}
        stale = self._collection.get(where=where, include=[])["ids"]
        if stale:
            self._collection.delete(ids=stale)
            logger.info("Removed %d chunk(s) of older versions of %s", len(stale), source)
        return len(stale)

    def rebuild(self, batches: Iterable[tuple[list[Document], list[str]]]) -> int:
        """Replace the whole index with new content, without a window where it is half-built.

        Everything is written into a temporary collection first. Only when that
        succeeded is the live collection dropped and the new one renamed in its
        place. If embedding fails halfway, the live index is untouched.
        A fresh collection is also the only way to switch to an embedding model
        with a different vector size.
        """
        temp_name = f"{self._name}__rebuild"
        if temp_name in {c.name for c in self._client.list_collections()}:
            self._client.delete_collection(temp_name)  # leftover of an interrupted rebuild

        temp_lc, temp_collection = self._open(temp_name)
        total = 0
        try:
            for documents, ids in batches:
                temp_lc.add_documents(documents, ids=ids)
                total += len(ids)
        except Exception:
            self._client.delete_collection(temp_name)
            raise

        self._client.delete_collection(self._name)
        temp_collection.modify(name=self._name)
        self._lc, self._collection = self._open(self._name)
        logger.info("Index rebuilt: %d chunk(s) in collection %s", total, self._name)
        return total

    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        """Return (chunk, relevance) pairs, relevance = cosine similarity in [0, 1]."""
        hits = self._lc.similarity_search_with_score(query, k=k)
        return [(doc, max(0.0, 1.0 - distance)) for doc, distance in hits]
