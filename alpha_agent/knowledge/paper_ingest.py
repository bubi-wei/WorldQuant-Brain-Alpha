"""Paper / document ingest pipeline.

Reads PDF, Markdown, or plain-text research papers and ingests them into
the Chroma 'papers' collection so they can be retrieved during RAG.

Supported input formats:
  - .txt / .md  — plain text, split by paragraph
  - .pdf        — requires the 'pypdf' package (optional dependency)

Usage:
    from alpha_agent.knowledge.paper_ingest import PaperIngest
    from alpha_agent.knowledge.vector_store import VectorStore

    store = VectorStore()
    pi = PaperIngest(store)
    pi.ingest_file("papers/momentum_effect.pdf", source="Harvey2016")
    pi.ingest_text("...", source="manual_notes")

    results = pi.search("cross-sectional momentum reversal", k=5)
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from alpha_agent.knowledge.vector_store import COLLECTION_PAPERS, VectorStore

_DEFAULT_CHUNK_SIZE = 400   # tokens (rough word count proxy)
_DEFAULT_CHUNK_OVERLAP = 80


class PaperIngest:
    """Chunk and embed documents into the 'papers' Chroma collection."""

    def __init__(
        self,
        vector_store: VectorStore,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        self._store = vector_store
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    # ── public API ────────────────────────────────────────────────────────────

    def ingest_file(self, path: str | Path, source: str | None = None) -> int:
        """Ingest a file. Returns number of chunks indexed."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")

        src = source or p.stem
        suffix = p.suffix.lower()

        if suffix == ".pdf":
            text = self._read_pdf(p)
        else:
            text = p.read_text(encoding="utf-8", errors="ignore")

        return self.ingest_text(text, source=src, file_path=str(p))

    def ingest_text(
        self,
        text: str,
        source: str = "manual",
        file_path: str = "",
        extra_meta: dict[str, Any] | None = None,
    ) -> int:
        """Chunk and ingest raw text. Returns number of chunks indexed."""
        chunks = self._chunk_text(text)
        if not chunks:
            return 0

        ids, docs, metas = [], [], []
        for i, chunk in enumerate(chunks):
            chunk_id = hashlib.md5(f"{source}:{i}:{chunk[:50]}".encode()).hexdigest()
            ids.append(chunk_id)
            docs.append(chunk)
            meta: dict[str, Any] = {
                "source": source,
                "chunk_index": i,
                "total_chunks": len(chunks),
            }
            if file_path:
                meta["file_path"] = file_path
            if extra_meta:
                meta.update(extra_meta)
            metas.append(meta)

        self._store.upsert(COLLECTION_PAPERS, ids=ids, documents=docs, metadatas=metas)
        return len(chunks)

    def ingest_directory(self, directory: str | Path, glob: str = "**/*.{pdf,md,txt}") -> dict[str, int]:
        """Ingest all matching files in a directory. Returns {filename: chunk_count}."""
        dir_path = Path(directory)
        results: dict[str, int] = {}
        # manual glob to avoid f-string issues with curly braces
        for suffix in ("pdf", "md", "txt"):
            for fp in dir_path.glob(f"**/*.{suffix}"):
                try:
                    n = self.ingest_file(fp)
                    results[fp.name] = n
                    print(f"[PaperIngest] Ingested {fp.name}: {n} chunks")
                except Exception as e:
                    print(f"[PaperIngest] Failed {fp.name}: {e}")
                    results[fp.name] = 0
        return results

    def search(
        self,
        query: str,
        k: int = 5,
        source_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic search over ingested papers."""
        where = {"source": source_filter} if source_filter else None
        return self._store.query(COLLECTION_PAPERS, query, k=k, where=where)

    # ── private helpers ───────────────────────────────────────────────────────

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping word-count chunks."""
        # Normalize whitespace
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        words = text.split()
        if not words:
            return []

        chunks: list[str] = []
        start = 0
        while start < len(words):
            end = min(start + self._chunk_size, len(words))
            chunk = " ".join(words[start:end])
            if len(chunk.strip()) > 30:  # skip tiny chunks
                chunks.append(chunk)
            start += self._chunk_size - self._chunk_overlap

        return chunks

    @staticmethod
    def _read_pdf(path: Path) -> str:
        try:
            import pypdf  # noqa: PLC0415
            reader = pypdf.PdfReader(str(path))
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n\n".join(pages)
        except ImportError:
            raise ImportError(
                "pypdf is required for PDF ingestion. "
                "Install it with: pip install pypdf"
            ) from None
