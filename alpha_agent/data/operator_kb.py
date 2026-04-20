"""FASTEXPR Operator Knowledge Base loader and indexer.

Loads operators from YAML, exposes lookup utilities,
and ingests operator descriptions into the vector store.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from alpha_agent.knowledge.vector_store import COLLECTION_OPERATORS, VectorStore

_ASSETS_DIR = Path(__file__).parent / "assets"
_OPERATORS_YAML = _ASSETS_DIR / "operators.yaml"


class OperatorKB:
    """In-memory operator registry loaded from YAML."""

    def __init__(self, yaml_path: Path | None = None) -> None:
        path = yaml_path or _OPERATORS_YAML
        with open(path) as f:
            raw = yaml.safe_load(f)
        self._operators: list[dict[str, Any]] = raw.get("operators", [])
        self._by_name: dict[str, dict[str, Any]] = {
            op["name"]: op for op in self._operators
        }

    @property
    def operators(self) -> list[dict[str, Any]]:
        return self._operators

    def get(self, name: str) -> dict[str, Any] | None:
        return self._by_name.get(name)

    def by_category(self, category: str) -> list[dict[str, Any]]:
        return [op for op in self._operators if op.get("category") == category]

    def all_names(self) -> list[str]:
        return [op["name"] for op in self._operators if not op["name"].startswith("_")]

    def to_prompt_text(self, categories: list[str] | None = None) -> str:
        """Format operators as a compact reference block for LLM prompts."""
        ops = self._operators
        if categories:
            ops = [op for op in ops if op.get("category") in categories]
        lines = ["FASTEXPR Operator Reference:"]
        for op in ops:
            if op["name"].startswith("_"):
                continue
            lines.append(
                f"  {op['signature']} → {op['return_type']}: {op['description']}"
            )
        return "\n".join(lines)

    def index_into_store(self, store: VectorStore, force_refresh: bool = False) -> int:
        """Ingest all operators into the vector store. Returns count indexed."""
        if not force_refresh and store.count(COLLECTION_OPERATORS) >= len(self._operators):
            return 0

        ids, docs, metas = [], [], []
        for op in self._operators:
            op_id = f"op:{op['name']}"
            document = (
                f"Operator: {op['name']}\n"
                f"Category: {op.get('category', '')}\n"
                f"Signature: {op.get('signature', '')}\n"
                f"Returns: {op.get('return_type', '')}\n"
                f"Description: {op.get('description', '')}\n"
                f"Example: {op.get('example', '')}\n"
                f"Tips: {op.get('tips', '')}"
            )
            ids.append(op_id)
            docs.append(document)
            metas.append({
                "name": op["name"],
                "category": op.get("category", ""),
                "arity": str(op.get("arity", -1)),
                "return_type": op.get("return_type", ""),
            })

        store.upsert(COLLECTION_OPERATORS, ids=ids, documents=docs, metadatas=metas)
        return len(ids)
