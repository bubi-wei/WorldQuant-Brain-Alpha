"""Data field indexer — pulls WQB fields and enriches them with LLM semantics.

Usage:
    async with WQBClient() as client:
        indexer = DataFieldIndexer(client, vector_store)
        await indexer.index_dataset("fundamental6", universe="TOP3000")
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any

import litellm

from alpha_agent.config import settings
from alpha_agent.data.wqb_client import WQBClient
from alpha_agent.llm_utils import supports_json_response_format
from alpha_agent.knowledge.vector_store import COLLECTION_FIELDS, VectorStore

_SEMANTIC_PROMPT = """\
You are a quantitative finance data expert.
Given the following data field from the WorldQuant Brain platform, write a concise 3-line description:
1. Category (e.g., Price/Volume, Fundamental, Analyst Estimate, Risk)
2. What this field measures or represents
3. Common usage patterns in alpha factor research

Field ID: {field_id}
Field description (from API): {field_desc}
Dataset: {dataset}

Reply in JSON with keys: category, meaning, usage
"""


class DataFieldIndexer:
    """Fetches data fields from WQB, generates LLM semantics, and indexes into Chroma."""

    def __init__(
        self,
        wqb_client: WQBClient,
        vector_store: VectorStore,
        llm_model: str | None = None,
        batch_size: int = 10,
    ) -> None:
        self._client = wqb_client
        self._store = vector_store
        self._model = llm_model or settings.llm_model
        self._batch_size = batch_size

    @staticmethod
    def _is_model_provider_supported(model: str) -> bool:
        """Best-effort validation to prevent repeated provider errors in batch calls."""
        # LiteLLM supports plain names for some default providers (e.g. gpt-4o),
        # so only validate strings that explicitly use "provider/model" format.
        if "/" not in model:
            return True

        provider = model.split("/", 1)[0].strip().lower()
        known_prefixes = {
            "openai",
            "anthropic",
            "deepseek",
            "openrouter",
            "ollama",
            "azure",
            "vertex_ai",
            "gemini",
            "huggingface",
            "bedrock",
            "mistral",
            "groq",
            "xai",
            "nvidia_nim",
            "novita",
        }
        return provider in known_prefixes

    def _build_fallback_semantics(
        self,
        field: dict[str, Any],
        dataset_id: str,
    ) -> dict[str, Any]:
        """Deterministic fallback when LLM is unavailable/misconfigured."""
        field_id = field.get("id", "")
        doc_id = hashlib.md5(f"{dataset_id}:{field_id}".encode()).hexdigest()
        field_desc = field.get("description", "")
        semantics_json = (
            "{"
            f'"category": "Unknown", '
            f'"meaning": "{field_desc or field_id}", '
            '"usage": "manual review required"'
            "}"
        )
        document = (
            f"Field: {field_id}\nDataset: {dataset_id}\n"
            f"Description: {field_desc}\n"
            f"Semantics: {semantics_json}"
        )
        return {
            **field,
            "doc_id": doc_id,
            "semantics_json": semantics_json,
            "document": document,
        }

    def _completion_kwargs(self, prompt: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }
        if supports_json_response_format(self._model):
            kwargs["response_format"] = {"type": "json_object"}
        return kwargs

    async def index_dataset(
        self,
        dataset_id: str,
        universe: str = "TOP3000",
        delay: int = 1,
        force_refresh: bool = False,
    ) -> list[dict[str, Any]]:
        """Index all MATRIX fields of a dataset into the vector store.

        Returns list of enriched field dicts.
        """
        print(f"[DataFieldIndexer] Fetching fields for dataset={dataset_id} universe={universe}")
        fields = await self._client.fetch_datafields(
            dataset_id=dataset_id,
            universe=universe,
            delay=delay,
        )
        if not fields:
            print(f"[DataFieldIndexer] No fields returned for {dataset_id}")
            return []

        print(f"[DataFieldIndexer] Got {len(fields)} fields, enriching with LLM semantics...")
        if self._is_model_provider_supported(self._model):
            enriched = await self._enrich_fields(fields, dataset_id, force_refresh)
        else:
            print(
                "[DataFieldIndexer] Invalid LLM_MODEL format for LiteLLM: "
                f"{self._model}. Falling back to deterministic semantics."
            )
            print(
                "[DataFieldIndexer] Use provider-prefixed model names, e.g. "
                "openai/gpt-4o or deepseek/deepseek-chat."
            )
            enriched = [self._build_fallback_semantics(f, dataset_id) for f in fields]

        self._upsert_to_store(enriched, dataset_id, universe)
        print(f"[DataFieldIndexer] Indexed {len(enriched)} fields into vector store.")
        return enriched

    async def _enrich_fields(
        self,
        fields: list[dict[str, Any]],
        dataset_id: str,
        force_refresh: bool,
    ) -> list[dict[str, Any]]:
        enriched: list[dict[str, Any]] = []

        # process in batches to avoid overwhelming the LLM rate limit
        for i in range(0, len(fields), self._batch_size):
            batch = fields[i : i + self._batch_size]
            tasks = [
                self._get_semantics(f, dataset_id, force_refresh)
                for f in batch
            ]
            results = await asyncio.gather(*tasks)
            enriched.extend(r for r in results if r is not None)
            if i + self._batch_size < len(fields):
                await asyncio.sleep(1)  # gentle rate-limit

        return enriched

    async def _get_semantics(
        self,
        field: dict[str, Any],
        dataset_id: str,
        force_refresh: bool,
    ) -> dict[str, Any] | None:
        field_id = field.get("id", "")
        if not field_id:
            return None

        doc_id = hashlib.md5(f"{dataset_id}:{field_id}".encode()).hexdigest()

        if not force_refresh:
            existing = self._store.query(COLLECTION_FIELDS, field_id, k=1)
            if existing and existing[0]["metadata"].get("field_id") == field_id:
                return {**field, "doc_id": doc_id, "semantics": existing[0]["document"]}

        prompt = _SEMANTIC_PROMPT.format(
            field_id=field_id,
            field_desc=field.get("description", ""),
            dataset=dataset_id,
        )
        try:
            resp = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: litellm.completion(**self._completion_kwargs(prompt)),
            )
            semantics_json = resp.choices[0].message.content or "{}"
        except Exception as e:
            print(f"[DataFieldIndexer] LLM error for {field_id}: {e}")
            semantics_json = (
                f'{{"category": "Unknown", "meaning": "{field_id}", "usage": "general"}}'
            )

        document = (
            f"Field: {field_id}\nDataset: {dataset_id}\n"
            f"Description: {field.get('description', '')}\n"
            f"Semantics: {semantics_json}"
        )

        return {
            **field,
            "doc_id": doc_id,
            "semantics_json": semantics_json,
            "document": document,
        }

    def _upsert_to_store(
        self,
        enriched: list[dict[str, Any]],
        dataset_id: str,
        universe: str,
    ) -> None:
        ids, docs, metas = [], [], []
        for f in enriched:
            doc_id = f.get("doc_id", f["id"])
            document = f.get("document", f["id"])
            ids.append(doc_id)
            docs.append(document)
            metas.append({
                "field_id": f.get("id", ""),
                "dataset": dataset_id,
                "universe": universe,
                "field_type": f.get("type", ""),
            })
        self._store.upsert(COLLECTION_FIELDS, ids=ids, documents=docs, metadatas=metas)

    def search_fields(self, query: str, k: int = 5, dataset: str | None = None) -> list[dict[str, Any]]:
        """Semantic search over indexed fields."""
        where = {"dataset": dataset} if dataset else None
        return self._store.query(COLLECTION_FIELDS, query, k=k, where=where)
