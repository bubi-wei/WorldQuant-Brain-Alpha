"""TrackBandit — two-arm UCB bandit for Explorer vs Skeleton track allocation.

Each arm represents one generation track:
  - "explorer": IdeaAgent → ExprSynthAgent (free LLM synthesis)
  - "skeleton": SkeletonAgent (slot substitution within proven templates)

The bandit allocates the total simulation budget each round between the two
tracks based on their observed pass rates, with:
  - UCB1 for exploration-exploitation balance
  - EXPLORER_FLOOR: minimum fraction always given to Explorer (avoids starvation)
  - SKELETON_MIN_SEEDS: skeleton arm disabled until the registry has enough seeds
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from alpha_agent.config import settings
from alpha_agent.knowledge.skeleton_registry import SkeletonRegistry

TRACK_EXPLORER = "explorer"
TRACK_SKELETON = "skeleton"


@dataclass
class TrackStats:
    name: str
    tries: int = 0
    wins: int = 0

    @property
    def win_rate(self) -> float:
        return self.wins / self.tries if self.tries else 0.0

    def ucb(self, total_tries: int, c: float = 2.0) -> float:
        if self.tries == 0:
            return float("inf")
        return self.win_rate + c * math.sqrt(math.log(max(total_tries, 1)) / self.tries)


class TrackBandit:
    """Two-arm UCB1 bandit controlling Explorer/Skeleton allocation."""

    def __init__(
        self,
        skeleton_registry: SkeletonRegistry,
        initial_explorer_ratio: float | None = None,
        explorer_floor: float | None = None,
        skeleton_min_seeds: int | None = None,
        exploration_constant: float = 2.0,
    ) -> None:
        self._registry = skeleton_registry
        self._explorer_ratio = initial_explorer_ratio or settings.track_explorer_ratio
        self._explorer_floor = explorer_floor or settings.explorer_floor
        self._min_seeds = skeleton_min_seeds or settings.skeleton_min_seeds
        self._c = exploration_constant

        self._stats: dict[str, TrackStats] = {
            TRACK_EXPLORER: TrackStats(TRACK_EXPLORER),
            TRACK_SKELETON: TrackStats(TRACK_SKELETON),
        }

    # ── allocation ────────────────────────────────────────────────────────────

    def allocate(self, total_budget: int) -> dict[str, int]:
        """Decide how many simulation slots each track gets this round.

        Returns:
            {"explorer": n_e, "skeleton": n_s}  where n_e + n_s == total_budget
        """
        skeleton_ready = self._registry.count() >= self._min_seeds

        if not skeleton_ready:
            return {TRACK_EXPLORER: total_budget, TRACK_SKELETON: 0}

        ratio = self._compute_explorer_ratio()
        n_explorer = max(
            int(self._explorer_floor * total_budget),
            int(ratio * total_budget),
        )
        n_skeleton = total_budget - n_explorer
        return {TRACK_EXPLORER: n_explorer, TRACK_SKELETON: max(0, n_skeleton)}

    def _compute_explorer_ratio(self) -> float:
        """Use UCB1 to compute the allocation ratio for Explorer."""
        e = self._stats[TRACK_EXPLORER]
        s = self._stats[TRACK_SKELETON]
        total = e.tries + s.tries

        e_ucb = e.ucb(total, self._c)
        s_ucb = s.ucb(total, self._c)

        # Both untried → use initial ratio
        if e_ucb == float("inf") and s_ucb == float("inf"):
            return self._explorer_ratio

        # Both finite → ratio proportional to UCB scores
        total_ucb = e_ucb + s_ucb
        if total_ucb == 0:
            return self._explorer_ratio

        raw_ratio = e_ucb / total_ucb
        # enforce floor
        return max(self._explorer_floor, raw_ratio)

    # ── feedback ──────────────────────────────────────────────────────────────

    def update(self, track: str, simulated: int, qualified: int) -> None:
        """Record outcome for a track (simulated = budget used, qualified = wins)."""
        if track not in self._stats:
            return
        self._stats[track].tries += simulated
        self._stats[track].wins += qualified

    # ── inspection ────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        e = self._stats[TRACK_EXPLORER]
        s = self._stats[TRACK_SKELETON]
        total = e.tries + s.tries
        return {
            "explorer": {
                "tries": e.tries,
                "wins": e.wins,
                "win_rate": round(e.win_rate, 3),
                "ucb": round(e.ucb(total, self._c), 3) if e.tries else "∞",
            },
            "skeleton": {
                "tries": s.tries,
                "wins": s.wins,
                "win_rate": round(s.win_rate, 3),
                "ucb": round(s.ucb(total, self._c), 3) if s.tries else "∞",
                "seeds_available": self._registry.count(),
                "min_seeds_needed": self._min_seeds,
                "skeleton_active": self._registry.count() >= self._min_seeds,
            },
            "total_tries": total,
        }

    def allocation_preview(self, budget: int = 10) -> str:
        alloc = self.allocate(budget)
        e, s = alloc[TRACK_EXPLORER], alloc[TRACK_SKELETON]
        return (
            f"Budget={budget}: explorer={e} ({e/budget:.0%}) | "
            f"skeleton={s} ({s/budget:.0%})"
        )
