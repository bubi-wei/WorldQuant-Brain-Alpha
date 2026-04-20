"""Expression Synthesis Agent — hypothesis → FASTEXPR expressions.

Takes a structured factor hypothesis produced by IdeaAgent and generates
K concrete FASTEXPR expressions, validated locally before returning.

Features:
- Few-shot examples drawn from AlphaMemory (successful alphas)
- K temperature-sampled variants per hypothesis
- Automatic retry on parse/validation failure (up to max_retries)
- Returns only syntactically valid expressions
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import litellm

from alpha_agent.config import settings
from alpha_agent.data.operator_kb import OperatorKB
from alpha_agent.engine.validator import ExprValidator, ValidationResult
from alpha_agent.knowledge.alpha_memory import AlphaMemory

_SYSTEM_PROMPT = """\
You are an expert quantitative developer for WorldQuant Brain's FASTEXPR language.
Given a factor hypothesis and available data fields, write concrete FASTEXPR expressions.

FASTEXPR rules:
- All function calls use parentheses: rank(x), ts_mean(x, d)
- String arguments use single quotes: bucket(rank(cap), range='0,1,0.1')
- Arithmetic: +  -  *  /
- Comparison: >  <  >=  <=  ==  !=
- No Python syntax (no list comprehensions, no f-strings, no print)
- Expressions must return a MATRIX (one value per stock per day)
- The portfolio is automatically made dollar-neutral by the platform

Available operators (use ONLY these unless you have strong reason otherwise):
{operator_ref}

Output MUST be a JSON object: {{"expressions": ["expr1", "expr2", ...]}}
"""

_USER_TEMPLATE = """\
Factor hypothesis:
{hypothesis}

Rationale: {rationale}
Candidate fields: {candidate_fields}
Candidate operators: {candidate_operators}
Expected holding period: {holding_period}

Available dataset fields (use ONLY from this list or common builtins):
{all_fields}

=== Few-shot examples from successful alphas ===
{few_shots}

Generate {k} diverse FASTEXPR expression variants for this hypothesis.
Vary window sizes, operators, and structure across variants.
Each expression should be syntactically complete and self-contained.
"""


class ExprSynthAgent:
    """Synthesizes FASTEXPR expressions from factor hypotheses."""

    def __init__(
        self,
        operator_kb: OperatorKB,
        alpha_memory: AlphaMemory,
        known_fields: set[str] | None = None,
        model: str | None = None,
        max_retries: int = 3,
    ) -> None:
        self._kb = operator_kb
        self._memory = alpha_memory
        self._validator = ExprValidator(operator_kb, known_fields)
        self._model = model or settings.llm_model
        self._max_retries = max_retries

    def _completion_kwargs(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": 0.9,
            "n": 1,
        }
        # deepseek-reasoner rejects response_format=json_object.
        if not self._model.startswith("deepseek/"):
            kwargs["response_format"] = {"type": "json_object"}
        return kwargs

    def set_known_fields(self, fields: set[str]) -> None:
        self._validator.set_known_fields(fields)

    async def synthesize(
        self,
        hypothesis: dict[str, Any],
        all_fields: list[str],
        k: int | None = None,
    ) -> list[str]:
        """Generate k valid FASTEXPR expressions for a hypothesis.

        Args:
            hypothesis: Dict from IdeaAgent (contains hypothesis, candidate_fields, etc.)
            all_fields: Full list of field IDs available in this dataset.
            k:          Number of variants (default from settings).

        Returns:
            List of validated FASTEXPR expression strings.
        """
        k = k or settings.variants_per_idea
        few_shots = self._get_few_shots(hypothesis)
        operator_ref = self._kb.to_prompt_text()

        user_prompt = _USER_TEMPLATE.format(
            hypothesis=hypothesis.get("hypothesis", ""),
            rationale=hypothesis.get("rationale", ""),
            candidate_fields=", ".join(hypothesis.get("candidate_fields", [])),
            candidate_operators=", ".join(hypothesis.get("candidate_operators", [])),
            holding_period=hypothesis.get("expected_holding_period", "medium"),
            all_fields=", ".join(all_fields[:80]),  # truncate for context window
            few_shots=few_shots,
            k=k,
        )

        system_prompt = _SYSTEM_PROMPT.format(operator_ref=operator_ref)

        valid_exprs: list[str] = []
        for attempt in range(self._max_retries):
            expressions = await self._call_llm(system_prompt, user_prompt, k)
            for expr in expressions:
                result: ValidationResult = self._validator.validate(expr)
                if result.ok:
                    valid_exprs.append(expr)
                else:
                    user_prompt = self._append_error_feedback(
                        user_prompt, expr, result.errors
                    )

            if valid_exprs:
                break

            await asyncio.sleep(1)  # brief pause before retry

        return list(dict.fromkeys(valid_exprs))[:k]  # deduplicate, cap at k

    # ── private helpers ───────────────────────────────────────────────────────

    async def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        k: int,
    ) -> list[str]:
        try:
            resp = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: litellm.completion(**self._completion_kwargs([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ])),
            )
            content = resp.choices[0].message.content or "{}"
            parsed = json.loads(content)
            exprs = parsed.get("expressions", [])
            return [str(e).strip() for e in exprs if e]
        except Exception as e:
            print(f"[ExprSynthAgent] LLM call failed: {e}")
            return []

    def _get_few_shots(self, hypothesis: dict[str, Any]) -> str:
        """Fetch similar successful alphas from memory as few-shot examples."""
        candidate_fields = hypothesis.get("candidate_fields", [])
        if not candidate_fields:
            alphas = self._memory.top_by_metric("sharpe", k=3, qualified_only=True)
        else:
            # Try to find alphas using similar fields
            field = candidate_fields[0]
            alphas_for_field = [
                a for a in self._memory.top_by_metric("sharpe", k=10, qualified_only=True)
                if field in a.get("expression", "")
            ]
            alphas = alphas_for_field[:3] or self._memory.top_by_metric("sharpe", k=3, qualified_only=True)

        if not alphas:
            return "(no historical examples yet)"

        lines = ["Here are successful alpha expressions for inspiration (do NOT copy verbatim):"]
        for a in alphas:
            m = a.get("metrics", {})
            lines.append(
                f"  {a['expression']}  "
                f"[sharpe={m.get('sharpe', 0):.2f}]"
            )
        return "\n".join(lines)

    @staticmethod
    def _append_error_feedback(prompt: str, expr: str, errors: list[str]) -> str:
        feedback = (
            f"\n\n[Validation feedback] Expression '{expr[:60]}' failed:\n"
            + "\n".join(f"  - {e}" for e in errors)
            + "\nPlease generate corrected variants."
        )
        return prompt + feedback
