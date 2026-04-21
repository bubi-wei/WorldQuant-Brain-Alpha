"""Novelty scoring for alpha expressions.

Two modes:
  - "strict" (Explorer track): 1 - max_cosine_similarity to ALL historical embeddings.
    High similarity to any past expression → low score → filtered out.
    This preserves the intent of the original design for Explorer.

  - "field_coverage" (Skeleton track): within the same skeleton, how many
    of the field slot combinations are genuinely NEW (not seen before)?
    Cross-skeleton expressions always get score 1.0.
    This avoids punishing skeleton variants for being structurally similar —
    that is a feature, not a bug.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

from alpha_agent.knowledge.alpha_memory import AlphaMemory
from alpha_agent.knowledge.vector_store import VectorStore

_MODE_STRICT = "strict"
_MODE_FIELD_COVERAGE = "field_coverage"


class NoveltyScorer:
    """Computes novelty scores for alpha expressions."""

    def __init__(
        self,
        vector_store: VectorStore,
        alpha_memory: AlphaMemory,
        expr_weight: float = 0.6,
        metric_weight: float = 0.4,
    ) -> None:
        self._store = vector_store
        self._memory = alpha_memory
        self._expr_w = expr_weight
        self._metric_w = metric_weight

    # ── public API ────────────────────────────────────────────────────────────

    def score_expression(
        self,
        expression: str,
        metrics: dict[str, Any] | None = None,
        mode: str = _MODE_STRICT,
        skeleton_id: str | None = None,
        field_mapping: dict[str, str] | None = None,
    ) -> float:
        """Compute novelty_score for a single expression.

        Args:
            expression:    FASTEXPR string.
            metrics:       Dict with sharpe/fitness/turnover/margin (optional).
            mode:          "strict" (Explorer) or "field_coverage" (Skeleton).
            skeleton_id:   Required when mode="field_coverage".
            field_mapping: Slot→field mapping used in this instantiation.

        Returns:
            novelty_score in [0, 1]. Higher = more novel.
        """
        if mode == _MODE_FIELD_COVERAGE:
            return self._field_coverage_novelty(skeleton_id, field_mapping)

        # strict mode (Explorer)
        expr_novelty = self._expression_novelty(expression)
        if metrics:
            metric_novelty = self._metric_novelty(metrics)
            return self._expr_w * expr_novelty + self._metric_w * metric_novelty
        return expr_novelty

    def score_batch(
        self,
        expressions: list[str],
        metrics_list: list[dict[str, Any]] | None = None,
        mode: str = _MODE_STRICT,
        skeleton_ids: list[str | None] | None = None,
        field_mappings: list[dict[str, str] | None] | None = None,
    ) -> list[float]:
        n = len(expressions)
        metrics_list = metrics_list or [None] * n  # type: ignore[list-item]
        skeleton_ids = skeleton_ids or [None] * n
        field_mappings = field_mappings or [None] * n
        return [
            self.score_expression(expr, m, mode=mode, skeleton_id=sk, field_mapping=fm)
            for expr, m, sk, fm in zip(expressions, metrics_list, skeleton_ids, field_mappings)
        ]

    def filter_novel(
        self,
        expressions: list[str],
        threshold: float = 0.3,
        mode: str = _MODE_STRICT,
        skeleton_ids: list[str | None] | None = None,
        field_mappings: list[dict[str, str] | None] | None = None,
    ) -> list[str]:
        """Return only expressions with novelty_score >= threshold."""
        scores = self.score_batch(
            expressions,
            mode=mode,
            skeleton_ids=skeleton_ids,
            field_mappings=field_mappings,
        )
        return [expr for expr, score in zip(expressions, scores) if score >= threshold]

    # ── strict mode (Explorer) ────────────────────────────────────────────────

    def _expression_novelty(self, expression: str) -> float:
        """1 - max_cosine_similarity to historical expressions."""
        embedding = self._store.embed([expression])[0]
        similar = self._memory.similar(embedding, k=5)
        if not similar:
            return 1.0
        max_sim = max(item.get("similarity", 0.0) for item in similar)
        return float(1.0 - max_sim)

    def _metric_novelty(self, metrics: dict[str, Any]) -> float:
        """Normalized distance in metric space from existing qualified alphas."""
        top_alphas = self._memory.top_by_metric("sharpe", k=20, qualified_only=True)
        if not top_alphas:
            return 1.0
        new_vec = self._metrics_to_vec(metrics)
        historical_vecs = [self._metrics_to_vec(a["metrics"]) for a in top_alphas]
        distances = [float(np.linalg.norm(new_vec - h_vec)) for h_vec in historical_vecs]
        min_dist = min(distances)
        return float(2 / (1 + math.exp(-min_dist)) - 1)

    @staticmethod
    def _metrics_to_vec(metrics: dict[str, Any]) -> np.ndarray:
        return np.array([
            float(metrics.get("sharpe", 0.0)),
            float(metrics.get("fitness", 0.0)),
            float(metrics.get("turnover", 0.0)),
            float(metrics.get("margin", 0.0)),
        ], dtype=np.float32)

    # ── field_coverage mode (Skeleton track) ──────────────────────────────────

    def _field_coverage_novelty(
        self,
        skeleton_id: str | None,
        field_mapping: dict[str, str] | None,
    ) -> float:
        """Score based on whether this field combination is new for this skeleton.

        - No skeleton_id → return 1.0 (treat as novel)
        - field_mapping is a subset of a never-seen combo → 1.0
        - Exact combo seen before → 0.0
        - Partial overlap → interpolated score
        """
        if skeleton_id is None or not field_mapping:
            return 1.0

        try:
            from alpha_agent.knowledge.skeleton_registry import SkeletonRegistry  # noqa: PLC0415
            registry = SkeletonRegistry()
            seen_combos = registry.seen_field_combos(skeleton_id)
            registry.close()
        except Exception:
            return 1.0

        if not seen_combos:
            return 1.0

        new_combo_str = json.dumps(field_mapping, sort_keys=True)
        # exact match → 0
        for seen in seen_combos:
            if json.dumps(seen, sort_keys=True) == new_combo_str:
                return 0.0

        # count how many individual field values overlap with any seen combo
        new_vals = set(field_mapping.values())
        overlap_fractions = []
        for seen in seen_combos:
            seen_vals = set(seen.values())
            overlap = len(new_vals & seen_vals)
            overlap_fractions.append(overlap / max(len(new_vals), 1))

        max_overlap = max(overlap_fractions) if overlap_fractions else 0.0
        # convert: 0 overlap = 1.0 novelty, full overlap (but not exact) = 0.3
        return float(max(0.3, 1.0 - max_overlap * 0.7))
