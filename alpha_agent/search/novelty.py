"""Novelty scoring for alpha expressions.

Computes a novelty_score ∈ [0, 1] for a new alpha based on:
  1. Expression-level: cosine similarity to historical expression embeddings
  2. Metric-level: distance in (sharpe, fitness, turnover, ic_mean) space
     relative to already-qualified alphas

A high novelty_score means the expression is structurally distinct from
previous attempts AND/OR produces a metric profile unlike existing alphas.
High novelty is rewarded; near-duplicates are deprioritized.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from alpha_agent.knowledge.alpha_memory import AlphaMemory
from alpha_agent.knowledge.vector_store import VectorStore


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

    def score_expression(
        self,
        expression: str,
        metrics: dict[str, Any] | None = None,
    ) -> float:
        """Compute novelty_score for a single expression.

        Args:
            expression: FASTEXPR string.
            metrics:    Dict with sharpe/fitness/turnover/margin keys (optional).

        Returns:
            novelty_score in [0, 1]. Higher = more novel.
        """
        expr_novelty = self._expression_novelty(expression)

        if metrics:
            metric_novelty = self._metric_novelty(metrics)
            return self._expr_w * expr_novelty + self._metric_w * metric_novelty

        return expr_novelty

    def score_batch(
        self,
        expressions: list[str],
        metrics_list: list[dict[str, Any]] | None = None,
    ) -> list[float]:
        metrics_list = metrics_list or [None] * len(expressions)  # type: ignore[list-item]
        return [
            self.score_expression(expr, m)
            for expr, m in zip(expressions, metrics_list)
        ]

    def filter_novel(
        self,
        expressions: list[str],
        threshold: float = 0.3,
    ) -> list[str]:
        """Return only expressions with novelty_score >= threshold."""
        scores = self.score_batch(expressions)
        return [expr for expr, score in zip(expressions, scores) if score >= threshold]

    # ── expression novelty ────────────────────────────────────────────────────

    def _expression_novelty(self, expression: str) -> float:
        """1 - max_cosine_similarity to historical expressions."""
        embedding = self._store.embed([expression])[0]

        similar = self._memory.similar(embedding, k=5)
        if not similar:
            return 1.0

        max_sim = max(item.get("similarity", 0.0) for item in similar)
        return float(1.0 - max_sim)

    # ── metric novelty ────────────────────────────────────────────────────────

    def _metric_novelty(self, metrics: dict[str, Any]) -> float:
        """Normalized distance in metric space from existing qualified alphas."""
        top_alphas = self._memory.top_by_metric("sharpe", k=20, qualified_only=True)
        if not top_alphas:
            return 1.0

        new_vec = self._metrics_to_vec(metrics)
        historical_vecs = [
            self._metrics_to_vec(a["metrics"])
            for a in top_alphas
        ]

        distances = [
            float(np.linalg.norm(new_vec - h_vec))
            for h_vec in historical_vecs
        ]
        min_dist = min(distances)

        # Normalize: typical metric-space distance is O(1–3)
        # Sigmoid so we get a [0,1] novelty score
        return float(2 / (1 + math.exp(-min_dist)) - 1)

    @staticmethod
    def _metrics_to_vec(metrics: dict[str, Any]) -> np.ndarray:
        return np.array([
            float(metrics.get("sharpe", 0.0)),
            float(metrics.get("fitness", 0.0)),
            float(metrics.get("turnover", 0.0)),
            float(metrics.get("margin", 0.0)),
        ], dtype=np.float32)
