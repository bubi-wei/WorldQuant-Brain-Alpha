"""Skeleton Agent — controlled variation within proven expression structures.

This is the "entropy-reduction" track of the dual-agent system.
It does NOT generate new expression structures; instead it:
  1. Picks seed skeletons from SkeletonRegistry (UCB/top/recent strategy)
  2. For each slot, uses VectorStore field-semantic RAG to find candidate fields
  3. Asks LLM to pick the best combination (low-temperature, structured JSON)
  4. Instantiates the template via SkeletonExtractor.instantiate()
  5. Validates locally (syntax only — no strict novelty filter)
  6. Returns candidate (expression, skeleton_id, slot_mapping) tuples

Key differences from ExplorerAgent:
  - Fixed structure, only slots change
  - LLM temperature 0.2-0.3 (exploit, not explore)
  - Generates 5-10 variants per skeleton (cheap)
  - Uses field_coverage novelty (not strict embedding novelty)
  - Explicitly avoids already-seen field combinations
"""

from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass, field
from typing import Any

from alpha_agent.config import settings
from alpha_agent.engine.skeleton_extractor import SkeletonExtractor
from alpha_agent.engine.validator import ExprValidator
from alpha_agent.llm_utils import supports_json_response_format
from alpha_agent.knowledge.skeleton_registry import SkeletonRecord, SkeletonRegistry
from alpha_agent.knowledge.vector_store import VectorStore

_SYSTEM_PROMPT = """\
You are a quantitative researcher instantiating a FASTEXPR template by selecting
specific data fields and parameter values.

Your goal is to produce diverse, economically meaningful combinations.
Always avoid field combinations that have already been tried (listed as "seen").
"""

_USER_TEMPLATE = """\
Template: {template_str}

Origin hypothesis: {hypothesis}

Field slots to fill:
{field_slots_text}

Parameter slots to fill:
{param_slots_text}

Group slots (fixed to one of the candidates):
{group_slots_text}

Available fields for this dataset (pick ONLY from this list):
{available_fields}

Already-tried field combinations (AVOID these exact combinations):
{seen_combos}

Generate {k} diverse combinations. Each combination must use different fields.
Reply ONLY as JSON: {{"combinations": [{{"$X1": "field_name", "$W1": 20, "$G1": "subindustry", ...}}, ...]}}
"""


@dataclass
class SkeletonVariant:
    """One instantiated variant from a skeleton."""
    expression: str
    skeleton_id: str
    field_mapping: dict[str, str]
    param_mapping: dict[str, Any]
    group_mapping: dict[str, str]

    def slot_combo_key(self) -> str:
        """Canonical string for deduplication."""
        combined = {**self.field_mapping, **self.param_mapping}
        return json.dumps(combined, sort_keys=True)


class SkeletonAgent:
    """Generates expression variants by instantiating skeleton templates."""

    def __init__(
        self,
        skeleton_registry: SkeletonRegistry,
        vector_store: VectorStore,
        known_fields: set[str] | None = None,
        model: str | None = None,
    ) -> None:
        self._registry = skeleton_registry
        self._store = vector_store
        self._known_fields = known_fields or set()
        self._model = model or settings.llm_model
        self._extractor = SkeletonExtractor(model=self._model)
        self._validator = ExprValidator(known_fields=self._known_fields)

    def set_known_fields(self, fields: set[str]) -> None:
        self._known_fields = fields
        self._validator.set_known_fields(fields)

    # ── public API ────────────────────────────────────────────────────────────

    async def generate(
        self,
        dataset: str,
        k_per_skeleton: int | None = None,
        max_seeds: int | None = None,
        pick_strategy: str | None = None,
    ) -> list[SkeletonVariant]:
        """Generate expression variants from skeleton seeds.

        Args:
            dataset:         WQB dataset name for field candidate lookup.
            k_per_skeleton:  How many variants to generate per skeleton.
            max_seeds:       How many skeletons to use in this round.
            pick_strategy:   'ucb' | 'top_sharpe' | 'recent'.

        Returns:
            List of SkeletonVariant (expression + provenance).
        """
        k = k_per_skeleton or settings.skeleton_variants_per_seed
        n_seeds = max_seeds or 3
        strategy = pick_strategy or settings.skeleton_pick_strategy

        seeds = self._registry.pick_seeds(strategy=strategy, k=n_seeds)
        if not seeds:
            return []

        all_variants: list[SkeletonVariant] = []
        for seed in seeds:
            variants = await self._instantiate_seed(seed, dataset=dataset, k=k)
            all_variants.extend(variants)

        return all_variants

    # ── per-seed instantiation ────────────────────────────────────────────────

    async def _instantiate_seed(
        self,
        seed: SkeletonRecord,
        dataset: str,
        k: int,
    ) -> list[SkeletonVariant]:
        # 1. Get candidate fields for each field slot via RAG
        slot_candidates = await self._get_slot_candidates(seed, dataset)

        # 2. Get already-seen combinations (to avoid exact repeats)
        seen_combos = self._registry.seen_field_combos(seed.skeleton_id)

        # 3. Ask LLM to pick k combinations
        raw_combos = await self._ask_llm_combinations(
            seed=seed,
            slot_candidates=slot_candidates,
            seen_combos=seen_combos,
            k=k,
        )

        # 4. Instantiate + validate
        variants: list[SkeletonVariant] = []
        seen_keys: set[str] = set()

        for combo in raw_combos:
            field_mapping = {
                k: v for k, v in combo.items()
                if k.startswith("$X") and isinstance(v, str)
            }
            param_mapping = {
                k: v for k, v in combo.items()
                if k.startswith("$W")
            }
            group_mapping = {
                k: v for k, v in combo.items()
                if k.startswith("$G") and isinstance(v, str)
            }

            # fill missing group slots with defaults from registry
            for gs in seed.group_slots:
                name = gs["name"]
                if name not in group_mapping:
                    candidates = gs.get("candidates", ["subindustry"])
                    group_mapping[name] = candidates[0] if candidates else "subindustry"

            # fill missing param slots with sampled values
            for ps in seed.param_slots:
                name = ps["name"]
                if name not in param_mapping:
                    param_mapping[name] = self._sample_param(ps)

            # instantiate
            expr = SkeletonExtractor.instantiate(
                seed.template_str,
                field_mapping=field_mapping,
                param_mapping=param_mapping,
                group_mapping=group_mapping,
            )

            # validate
            result = self._validator.validate(expr)
            if not result.ok:
                continue

            variant = SkeletonVariant(
                expression=expr,
                skeleton_id=seed.skeleton_id,
                field_mapping=field_mapping,
                param_mapping=param_mapping,
                group_mapping=group_mapping,
            )

            # dedup within this batch
            key = variant.slot_combo_key()
            if key in seen_keys:
                continue
            seen_keys.add(key)
            variants.append(variant)

        return variants

    async def _get_slot_candidates(
        self,
        seed: SkeletonRecord,
        dataset: str,
    ) -> dict[str, list[str]]:
        """For each field slot, RAG-search candidate fields."""
        candidates: dict[str, list[str]] = {}
        for slot in seed.field_slots:
            hint = slot.get("semantic_hint", slot.get("literal", ""))
            hits = self._store.query(
                "datafields",
                query_text=hint or "financial data field",
                k=10,
                where={"dataset": dataset} if dataset else None,
            )
            field_ids = [h["metadata"].get("field_id", "") for h in hits if h["metadata"].get("field_id")]
            # also include common builtins that may not be in Chroma
            builtins = ["close", "open", "high", "low", "volume", "returns", "vwap",
                        "turnover", "cap", "sharesout"]
            merged = list(dict.fromkeys(field_ids + builtins))
            if self._known_fields:
                merged = [f for f in merged if f in self._known_fields or f in builtins]
            candidates[slot["name"]] = merged[:15]
        return candidates

    async def _ask_llm_combinations(
        self,
        seed: SkeletonRecord,
        slot_candidates: dict[str, list[str]],
        seen_combos: list[dict[str, Any]],
        k: int,
    ) -> list[dict[str, Any]]:
        field_slots_text = "\n".join(
            f"  {s['name']} — hint: '{s.get('semantic_hint', s.get('literal', '?'))}' "
            f"candidates: {slot_candidates.get(s['name'], [])}"
            for s in seed.field_slots
        )
        param_slots_text = "\n".join(
            f"  {p['name']} — type: {p['type']}, range: {p['range']}, seen: {p['seen']}"
            for p in seed.param_slots
        ) or "  (none)"

        group_slots_text = "\n".join(
            f"  {g['name']} — candidates: {g.get('candidates', ['subindustry'])}"
            for g in seed.group_slots
        ) or "  (none)"

        seen_text = json.dumps(seen_combos[:10]) if seen_combos else "[]"

        # all available fields from known set, capped for context window
        avail_fields = list(self._known_fields)[:60] if self._known_fields else \
                       ["close", "open", "volume", "returns", "cap"]

        user_prompt = _USER_TEMPLATE.format(
            template_str=seed.template_str,
            hypothesis=seed.origin_hypothesis[:150] if seed.origin_hypothesis else "(not recorded)",
            field_slots_text=field_slots_text,
            param_slots_text=param_slots_text,
            group_slots_text=group_slots_text,
            available_fields=", ".join(avail_fields),
            seen_combos=seen_text,
            k=k,
        )

        try:
            import litellm  # noqa: PLC0415
            completion_kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.3,
            }
            if supports_json_response_format(self._model):
                completion_kwargs["response_format"] = {"type": "json_object"}
            resp = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: litellm.completion(**completion_kwargs),
            )
            content = resp.choices[0].message.content or "{}"
            parsed = json.loads(content)
            combos = parsed.get("combinations", [])
            return combos if isinstance(combos, list) else []
        except Exception as e:
            print(f"[SkeletonAgent] LLM call failed: {e}")
            return self._fallback_combos(seed, slot_candidates, k)

    def _fallback_combos(
        self,
        seed: SkeletonRecord,
        slot_candidates: dict[str, list[str]],
        k: int,
    ) -> list[dict[str, Any]]:
        """Generate random combinations if LLM fails."""
        combos = []
        for _ in range(k):
            combo: dict[str, Any] = {}
            for slot in seed.field_slots:
                cands = slot_candidates.get(slot["name"], ["close"])
                combo[slot["name"]] = random.choice(cands) if cands else "close"
            for param in seed.param_slots:
                combo[param["name"]] = self._sample_param(param)
            for grp in seed.group_slots:
                combo[grp["name"]] = grp.get("candidates", ["subindustry"])[0]
            combos.append(combo)
        return combos

    @staticmethod
    def _sample_param(param_slot: dict[str, Any]) -> Any:
        """Sample a parameter value near already-seen values (+/-30%)."""
        seen = param_slot.get("seen", [])
        lo, hi = param_slot.get("range", [1, 252])
        ptype = param_slot.get("type", "int_window")

        if seen:
            base = random.choice(seen)
            delta = max(1, int(abs(base) * 0.3))
            val = base + random.randint(-delta, delta)
        else:
            val = random.uniform(lo, hi)

        if ptype == "int_window":
            val = max(int(lo), min(int(hi), int(round(val))))
        else:
            val = max(float(lo), min(float(hi), round(float(val), 3)))
        return val
