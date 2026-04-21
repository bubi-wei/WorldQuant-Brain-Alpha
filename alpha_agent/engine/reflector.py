"""Reflector — LLM-powered post-simulation analysis.

After a simulation completes (pass or fail), the Reflector:
1. Interprets the failure reasons / metric shortfalls
2. Produces a structured natural-language reflection
3. Writes the reflection back to AlphaMemory for future RAG retrieval

The reflection becomes part of the RAG context in subsequent rounds,
teaching the system what NOT to repeat.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import litellm

from alpha_agent.config import settings
from alpha_agent.data.wqb_client import AlphaResult
from alpha_agent.llm_utils import supports_json_response_format
from alpha_agent.knowledge.alpha_memory import AlphaMemory

_SYSTEM_PROMPT = """\
You are a senior quantitative researcher reviewing a WorldQuant Brain alpha simulation result.
Analyze the metrics and failure checks, identify the ROOT CAUSE of underperformance,
and suggest concrete actionable improvements.

Be specific: mention actual parameter values, operator choices, and field-level issues.
"""

_USER_TEMPLATE = """\
Alpha expression: {expression}
Hypothesis: {hypothesis}
Dataset: {dataset}

=== Simulation metrics ===
Sharpe ratio:  {sharpe:.3f}  (target > {sharpe_min})
Fitness:       {fitness:.3f}  (target > {fitness_min})
Turnover:      {turnover:.3f}  (target {turnover_min} – {turnover_max})
IC Mean:       {ic_mean:.3f}  (target > {ic_mean_min})

=== Qualification checks ===
{checks}

=== Failure reasons ===
{failure_reasons}

=== Qualified? ===
{qualified}

Provide a reflection with these JSON fields:
{{
  "root_cause": "<main reason for failure or success>",
  "metric_analysis": "<interpretation of the specific metric values>",
  "improvement_suggestions": ["suggestion1", "suggestion2", "suggestion3"],
  "fields_to_try": ["field1", "field2"],
  "operators_to_try": ["op1", "op2"],
  "parameters_to_adjust": {{"window": "try 5-10 for high turnover", ...}},
  "avoid_patterns": ["pattern to avoid in future expressions"]
}}
"""


class Reflector:
    """Analyzes simulation results and produces actionable reflections."""

    def __init__(
        self,
        alpha_memory: AlphaMemory,
        model: str | None = None,
    ) -> None:
        self._memory = alpha_memory
        self._model = model or settings.llm_model

    def _completion_kwargs(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": 0.3,
        }
        if supports_json_response_format(self._model):
            kwargs["response_format"] = {"type": "json_object"}
        return kwargs

    async def reflect_and_store(
        self,
        result: AlphaResult,
        hypothesis: str = "",
    ) -> str:
        """Generate a reflection for a simulation result and persist it.

        Returns the reflection text.
        """
        reflection = await self._generate_reflection(result, hypothesis)
        self._memory.update_reflection(result.alpha_id, reflection)
        return reflection

    async def reflect_batch(
        self,
        results: list[AlphaResult],
        hypotheses: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """Reflect on a batch of results concurrently.

        Returns {alpha_id: reflection_text}.
        """
        hypotheses = hypotheses or {}
        tasks = [
            self.reflect_and_store(r, hypotheses.get(r.alpha_id, ""))
            for r in results
        ]
        reflections = await asyncio.gather(*tasks)
        return {r.alpha_id: ref for r, ref in zip(results, reflections)}

    # ── private ───────────────────────────────────────────────────────────────

    async def _generate_reflection(
        self,
        result: AlphaResult,
        hypothesis: str,
    ) -> str:
        checks_text = self._format_checks(result.metrics.checks)
        failure_text = "\n".join(f"- {r}" for r in result.failure_reasons) or "(none — alpha passed)"

        prompt = _USER_TEMPLATE.format(
            expression=result.expression,
            hypothesis=hypothesis or "(not recorded)",
            dataset=result.dataset,
            sharpe=result.metrics.sharpe,
            sharpe_min=settings.qual_sharpe_min,
            fitness=result.metrics.fitness,
            fitness_min=settings.qual_fitness_min,
            turnover=result.metrics.turnover,
            turnover_min=settings.qual_turnover_min,
            turnover_max=settings.qual_turnover_max,
            ic_mean=result.metrics.ic_mean,
            ic_mean_min=settings.qual_ic_mean_min,
            checks=checks_text,
            failure_reasons=failure_text,
            qualified="YES — submitted" if result.qualified else "NO — not submitted",
        )

        try:
            resp = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: litellm.completion(**self._completion_kwargs([
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ])),
            )
            return resp.choices[0].message.content or "{}"
        except Exception as e:
            return json.dumps({"root_cause": f"LLM error: {e}", "improvement_suggestions": []})

    @staticmethod
    def _format_checks(checks: list[dict[str, Any]]) -> str:
        if not checks:
            return "(no check data)"
        lines = []
        for c in checks:
            status = c.get("result", "?")
            name = c.get("name", "")
            value = c.get("value", "N/A")
            limit = c.get("limit", "N/A")
            symbol = "✓" if status == "PASS" else "✗"
            lines.append(f"  {symbol} {name}: {value} (limit: {limit})")
        return "\n".join(lines)
