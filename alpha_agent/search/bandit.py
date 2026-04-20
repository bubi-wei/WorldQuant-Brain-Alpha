"""UCB Bandit — direction / prompt-level exploration-exploitation.

Uses the Upper Confidence Bound (UCB1) algorithm to select the next
research direction from a pool of candidates.

Each "arm" is a research direction string. The reward is whether
the round produced at least one qualified alpha.

Usage:
    bandit = DirectionBandit(alpha_memory, directions=[...])
    direction = bandit.select()
    ...  # run orchestrator round
    bandit.update(direction, won=True)
"""

from __future__ import annotations

import math
from typing import Any

from alpha_agent.knowledge.alpha_memory import AlphaMemory


class DirectionBandit:
    """UCB1 bandit for research direction selection.

    Balances exploring new directions against exploiting known good ones.
    """

    _DEFAULT_DIRECTIONS = [
        "cross-sectional price momentum with volume confirmation",
        "short-term reversal using intraday open-close dynamics",
        "earnings quality factors from fundamental accounting data",
        "analyst revision momentum and earnings surprise decay",
        "volatility regime signals using realized vs implied divergence",
        "liquidity-driven cross-sectional ranking",
        "sector-relative valuation anomalies",
        "earnings yield combined with growth quality screens",
        "price-to-book anomaly with leverage adjustment",
        "overnight vs intraday return decomposition",
    ]

    def __init__(
        self,
        alpha_memory: AlphaMemory,
        directions: list[str] | None = None,
        exploration_constant: float = 2.0,
    ) -> None:
        self._memory = alpha_memory
        self._directions = directions or self._DEFAULT_DIRECTIONS
        self._c = exploration_constant
        # Local cache: {direction: (tries, wins)}
        self._local: dict[str, tuple[int, int]] = {}

    def select(self) -> str:
        """Select the next direction using UCB1."""
        stats = self._get_stats()
        total_tries = sum(s["tries"] for s in stats.values()) or 1

        ucb_scores: list[tuple[float, str]] = []
        for direction in self._directions:
            s = stats.get(direction, {"tries": 0, "wins": 0})
            tries = s["tries"]
            wins = s["wins"]
            if tries == 0:
                # Untried arm: infinite UCB → always prefer first
                ucb_scores.append((float("inf"), direction))
            else:
                exploit = wins / tries
                explore = self._c * math.sqrt(math.log(total_tries) / tries)
                ucb_scores.append((exploit + explore, direction))

        ucb_scores.sort(key=lambda x: x[0], reverse=True)
        return ucb_scores[0][1]

    def update(self, direction: str, won: bool) -> None:
        """Record the outcome of a round for a direction."""
        tries, wins = self._local.get(direction, (0, 0))
        self._local[direction] = (tries + 1, wins + int(won))
        self._memory.record_direction_outcome(direction, won)

    def add_direction(self, direction: str) -> None:
        """Dynamically add a new direction to the pool."""
        if direction not in self._directions:
            self._directions.append(direction)

    def stats_table(self) -> list[dict[str, Any]]:
        """Return sorted stats with UCB score for display."""
        stats = self._get_stats()
        total_tries = sum(s["tries"] for s in stats.values()) or 1
        rows = []
        for direction in self._directions:
            s = stats.get(direction, {"tries": 0, "wins": 0})
            tries = s["tries"]
            wins = s["wins"]
            if tries == 0:
                ucb = float("inf")
                rate = 0.0
            else:
                rate = wins / tries
                ucb = rate + self._c * math.sqrt(math.log(total_tries) / tries)
            rows.append({
                "direction": direction[:60],
                "tries": tries,
                "wins": wins,
                "win_rate": round(rate, 3),
                "ucb_score": round(ucb, 3) if ucb != float("inf") else "∞",
            })
        rows.sort(key=lambda x: (x["ucb_score"] == "∞", x["ucb_score"]), reverse=True)
        return rows

    # ── private ───────────────────────────────────────────────────────────────

    def _get_stats(self) -> dict[str, dict[str, int]]:
        """Merge DB stats with local session cache."""
        db_stats = {r["name"]: r for r in self._memory.direction_stats()}
        merged: dict[str, dict[str, int]] = {}
        for direction in self._directions:
            db = db_stats.get(direction, {"tries": 0, "wins": 0})
            local = self._local.get(direction, (0, 0))
            merged[direction] = {
                "tries": db["tries"] + local[0],
                "wins": db["wins"] + local[1],
            }
        return merged
