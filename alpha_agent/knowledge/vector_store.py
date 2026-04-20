"""Chroma vector store wrapper — unified RAG interface.

All knowledge sources (data fields, operators, papers/alpha history)
are ingested into the same Chroma client with separate collections.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils import embedding_functions

from alpha_agent.config import settings

COLLECTION_FIELDS = "datafields"
COLLECTION_OPERATORS = "operators"
COLLECTION_PAPERS = "papers"


class VectorStore:
    """Thin wrapper around Chroma providing add/query per collection."""

    def __init__(self, persist_dir: Path | None = None) -> None:
        dir_ = persist_dir or settings.chroma_persist_dir
        dir_.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(dir_))
        self._ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.embed_model
        )
        self._cols: dict[str, chromadb.Collection] = {}

    def _col(self, name: str) -> chromadb.Collection:
        if name not in self._cols:
            self._cols[name] = self._client.get_or_create_collection(
                name=name,
                embedding_function=self._ef,
                metadata={"hnsw:space": "cosine"},
            )
        return self._cols[name]

    # ── add ───────────────────────────────────────────────────────────────────

    def add(
        self,
        collection: str,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        col = self._col(collection)
        existing = set(col.get(ids=ids)["ids"])
        new_idx = [i for i, id_ in enumerate(ids) if id_ not in existing]
        if not new_idx:
            return
        col.add(
            ids=[ids[i] for i in new_idx],
            documents=[documents[i] for i in new_idx],
            metadatas=[metadatas[i] if metadatas else {} for i in new_idx],
        )

    def upsert(
        self,
        collection: str,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        col = self._col(collection)
        col.upsert(
            ids=ids,
            documents=documents,
            metadatas=[m or {} for m in (metadatas or [{}] * len(ids))],
        )

    # ── query ─────────────────────────────────────────────────────────────────

    def query(
        self,
        collection: str,
        query_text: str,
        k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return top-k relevant chunks from a collection."""
        col = self._col(collection)
        kwargs: dict[str, Any] = {"query_texts": [query_text], "n_results": min(k, col.count())}
        if where:
            kwargs["where"] = where
        if col.count() == 0:
            return []
        results = col.query(**kwargs)
        items = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            items.append({"document": doc, "metadata": meta, "distance": dist})
        return items

    def query_multi(
        self,
        query_text: str,
        collections: list[str] | None = None,
        k_per_col: int = 3,
    ) -> list[dict[str, Any]]:
        """Query across multiple collections and return merged results."""
        cols = collections or [COLLECTION_FIELDS, COLLECTION_OPERATORS, COLLECTION_PAPERS]
        results: list[dict[str, Any]] = []
        for col_name in cols:
            hits = self.query(col_name, query_text, k=k_per_col)
            for hit in hits:
                hit["collection"] = col_name
                results.append(hit)
        results.sort(key=lambda x: x["distance"])
        return results

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts using the store's embedding function."""
        return self._ef(texts)  # type: ignore[return-value]

    def count(self, collection: str) -> int:
        return self._col(collection).count()

    def reset_collection(self, collection: str) -> None:
        self._client.delete_collection(collection)
        self._cols.pop(collection, None)
