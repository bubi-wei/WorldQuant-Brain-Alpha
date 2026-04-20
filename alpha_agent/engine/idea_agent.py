"""Idea Agent — generates factor hypotheses via RAG + LLM.

Given a research direction (e.g., "momentum reversal in small-caps"),
this agent:
  1. Retrieves relevant field semantics, operator examples, and papers from Chroma
  2. Fetches top historical alphas and recent failures from AlphaMemory
  3. Calls the LLM to produce N structured factor hypotheses in JSON

Each hypothesis is a research idea expressed in natural language,
ready to be handed off to ExprSynthAgent for FASTEXPR generation.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import litellm

from alpha_agent.config import settings
from alpha_agent.knowledge.alpha_memory import AlphaMemory
from alpha_agent.knowledge.vector_store import VectorStore

_SYSTEM_PROMPT = """\
You are an expert quantitative researcher specializing in systematic equity alpha factors.
Your task is to generate creative, rigorous factor hypotheses for the WorldQuant Brain platform.

Rules:
- Each hypothesis must have a clear economic or behavioral rationale
- Prefer hypotheses that are NOT obvious extensions of existing momentum/value templates
- Candidate fields must come from the provided dataset
- Candidate operators must exist in the provided FASTEXPR operator reference
- Flag any expected risks (high turnover, data sparsity, crowd risk)
- Output MUST be valid JSON
"""

_USER_TEMPLATE = """\
Research direction: {direction}
Dataset: {dataset}
Explore/exploit bias: {bias}  (0=fully explore novel ideas, 1=exploit known winners)

=== Available fields (semantic search results) ===
{fields_context}

=== Relevant operators ===
{operators_context}

=== Relevant research findings ===
{papers_context}

=== Top historical alphas (for reference, do NOT copy) ===
{top_alphas}

=== Recent failures to AVOID repeating ===
{recent_failures}

Generate exactly {n} factor hypotheses. Reply ONLY with a JSON array, each element:
{{
  "id": "<short_slug>",
  "hypothesis": "<1-2 sentence factor story in English>",
  "rationale": "<economic/behavioral reasoning>",
  "candidate_fields": ["field1", "field2"],
  "candidate_operators": ["op1", "op2"],
  "expected_holding_period": "<intraday|short|medium|long>",
  "risk": "<main risk factor>",
  "novelty_angle": "<what makes this different from common factors>"
}}
"""


class IdeaAgent:
    """Generates structured factor hypotheses using RAG + LLM."""

    def __init__(
        self,
        vector_store: VectorStore,
        alpha_memory: AlphaMemory,
        model: str | None = None,
    ) -> None:
        self._store = vector_store
        self._memory = alpha_memory
        self._model = model or settings.llm_model

    def _completion_kwargs(self, temperature: float) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [],
            "temperature": temperature,
        }
        # deepseek-reasoner rejects response_format=json_object.
        if not self._model.startswith("deepseek/"):
            kwargs["response_format"] = {"type": "json_object"}
        return kwargs

    async def generate_ideas(
        self,
        direction: str,
        dataset: str,
        n: int | None = None,
        explore_exploit_bias: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Generate n factor hypotheses for a given research direction.

        Args:
            direction: Natural-language research direction / theme.
            dataset:   WQB dataset name (e.g., 'fundamental6').
            n:         Number of hypotheses to generate (default from settings).
            explore_exploit_bias: 0.0 = fully explore, 1.0 = exploit winners.

        Returns:
            List of hypothesis dicts as produced by the LLM.
        """
        n = n or settings.ideas_per_round
        context = await self._build_context(direction, dataset, explore_exploit_bias)
        prompt = _USER_TEMPLATE.format(
            direction=direction,
            dataset=dataset,
            bias=f"{explore_exploit_bias:.1f}",
            n=n,
            **context,
        )

        raw = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: litellm.completion(**{
                **self._completion_kwargs(0.7 + 0.2 * (1 - explore_exploit_bias)),
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            }),
        )

        content = raw.choices[0].message.content or "[]"
        return self._parse_ideas(content, n)

    # ── context building ──────────────────────────────────────────────────────

    async def _build_context(
        self,
        direction: str,
        dataset: str,
        bias: float,
    ) -> dict[str, str]:
        fields_hits = self._store.query(
            "datafields", direction, k=8, where={"dataset": dataset} if dataset else None
        )
        ops_hits = self._store.query("operators", direction, k=5)
        papers_hits = self._store.query("papers", direction, k=4)

        top_alphas = self._memory.top_by_metric("sharpe", k=5, qualified_only=True)
        recent_failures = self._memory.recent(n=10)
        failures_only = [a for a in recent_failures if not a["qualified"]][:5]

        return {
            "fields_context": self._format_hits(fields_hits),
            "operators_context": self._format_hits(ops_hits),
            "papers_context": self._format_hits(papers_hits),
            "top_alphas": self._format_alphas(top_alphas),
            "recent_failures": self._format_failures(failures_only),
        }

    @staticmethod
    def _format_hits(hits: list[dict[str, Any]]) -> str:
        if not hits:
            return "(none)"
        lines = []
        for h in hits:
            doc = h.get("document", "")[:200]
            lines.append(f"- {doc}")
        return "\n".join(lines)

    @staticmethod
    def _format_alphas(alphas: list[dict[str, Any]]) -> str:
        if not alphas:
            return "(none)"
        lines = []
        for a in alphas:
            m = a.get("metrics", {})
            lines.append(
                f"- expr: {a['expression'][:80]}  "
                f"sharpe={m.get('sharpe', 0):.2f} fitness={m.get('fitness', 0):.2f}"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_failures(failures: list[dict[str, Any]]) -> str:
        if not failures:
            return "(none)"
        lines = []
        for f in failures:
            reflection = f.get("reflection", "")
            expr = f.get("expression", "")[:60]
            lines.append(f"- expr: {expr}  reflection: {reflection[:100]}")
        return "\n".join(lines)

    @staticmethod
    def _parse_ideas(content: str, expected_n: int) -> list[dict[str, Any]]:
        def _try_parse(payload: str) -> list[dict[str, Any]]:
            try:
                parsed = json.loads(payload)
                if isinstance(parsed, list):
                    return parsed[:expected_n]
                # LLM sometimes wraps list in a dict key
                if isinstance(parsed, dict):
                    for v in parsed.values():
                        if isinstance(v, list):
                            return v[:expected_n]
            except json.JSONDecodeError:
                return []
            return []

        # 1) direct parse
        items = _try_parse(content)
        if items:
            return items

        # 2) parse fenced code blocks
        fence_matches = re.findall(r"```(?:json)?\\s*(.*?)\\s*```", content, flags=re.DOTALL)
        for block in fence_matches:
            items = _try_parse(block)
            if items:
                return items

        # 3) best-effort: extract first JSON array/object chunk
        for pattern in [r"(\[.*\])", r"(\{.*\})"]:
            m = re.search(pattern, content, flags=re.DOTALL)
            if m:
                items = _try_parse(m.group(1))
                if items:
                    return items

        return []
