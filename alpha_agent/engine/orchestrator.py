"""Orchestrator — the main generation loop.

Pipeline per round:
  1. IdeaAgent generates N factor hypotheses (RAG + LLM)
  2. ExprSynthAgent converts each hypothesis into K FASTEXPR variants
  3. Local Validator filters out syntactically broken expressions
  4. NoveltyScorer filters out near-duplicate expressions
  5. WQBClient simulates the remaining expressions (async, rate-limited)
  6. Qualifier classifies results as pass/soft/fail
  7. AlphaMemory stores all results (with embeddings)
  8. Reflector analyzes failures and writes back insights
  9. Loop to next round (Bandit selects next direction if enabled)

Usage:
    async with WQBClient() as client:
        orchestrator = Orchestrator(client=client, ...)
        report = await orchestrator.run(
            direction="cross-sectional momentum with volume confirmation",
            dataset="pv1",
            universe="TOP1000",
            n_rounds=3,
        )
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any

from rich.console import Console
from rich.table import Table

from alpha_agent.config import settings
from alpha_agent.data.operator_kb import OperatorKB
from alpha_agent.data.wqb_client import AlphaResult, WQBClient
from alpha_agent.engine.expr_synth_agent import ExprSynthAgent
from alpha_agent.engine.idea_agent import IdeaAgent
from alpha_agent.engine.reflector import Reflector
from alpha_agent.engine.validator import ExprValidator
from alpha_agent.knowledge.alpha_memory import AlphaMemory
from alpha_agent.knowledge.vector_store import VectorStore
from alpha_agent.search.novelty import NoveltyScorer

console = Console()


class RoundReport:
    """Summary of one orchestration round."""

    def __init__(self, round_num: int, direction: str) -> None:
        self.round_num = round_num
        self.direction = direction
        self.ideas: list[dict[str, Any]] = []
        self.expressions_generated: list[str] = []
        self.expressions_after_validation: list[str] = []
        self.expressions_after_novelty: list[str] = []
        self.results: list[AlphaResult | None] = []
        self.qualified_ids: list[str] = []
        self.started_at = datetime.utcnow()
        self.ended_at: datetime | None = None

    @property
    def pass_rate(self) -> float:
        total = len([r for r in self.results if r is not None])
        if total == 0:
            return 0.0
        qualified = len(self.qualified_ids)
        return round(qualified / total, 3)

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round_num,
            "direction": self.direction,
            "ideas": len(self.ideas),
            "expressions_generated": len(self.expressions_generated),
            "after_validation": len(self.expressions_after_validation),
            "after_novelty_filter": len(self.expressions_after_novelty),
            "simulated": len([r for r in self.results if r is not None]),
            "qualified": len(self.qualified_ids),
            "pass_rate": self.pass_rate,
            "duration_s": (
                (self.ended_at - self.started_at).total_seconds()
                if self.ended_at else None
            ),
        }


class Orchestrator:
    """End-to-end alpha generation pipeline."""

    def __init__(
        self,
        client: WQBClient,
        vector_store: VectorStore,
        alpha_memory: AlphaMemory,
        operator_kb: OperatorKB | None = None,
        model: str | None = None,
        novelty_threshold: float | None = None,
        auto_submit: bool = False,
    ) -> None:
        self._client = client
        self._store = vector_store
        self._memory = alpha_memory
        self._kb = operator_kb or OperatorKB()
        self._model = model or settings.llm_model
        self._novelty_threshold = novelty_threshold or settings.novelty_score_min
        self._auto_submit = auto_submit

        self._idea_agent = IdeaAgent(vector_store, alpha_memory, model=self._model)
        self._synth_agent = ExprSynthAgent(self._kb, alpha_memory, model=self._model)
        self._validator = ExprValidator(self._kb)
        self._novelty = NoveltyScorer(vector_store, alpha_memory)
        self._reflector = Reflector(alpha_memory, model=self._model)

    async def run(
        self,
        direction: str,
        dataset: str,
        universe: str = "TOP3000",
        n_rounds: int | None = None,
        ideas_per_round: int | None = None,
        variants_per_idea: int | None = None,
        explore_exploit_bias: float = 0.5,
        dry_run: bool = False,
    ) -> list[RoundReport]:
        """Execute N rounds of the generation pipeline.

        Args:
            direction:          Research theme / direction.
            dataset:            WQB dataset ID.
            universe:           Universe (TOP1000, TOP3000, etc.).
            n_rounds:           Override settings.max_rounds.
            ideas_per_round:    Override settings.ideas_per_round.
            variants_per_idea:  Override settings.variants_per_idea.
            explore_exploit_bias: 0.0=explore, 1.0=exploit.
            dry_run:            Skip WQB simulation (validate + novelty only).

        Returns:
            List of RoundReport, one per round.
        """
        n_rounds = n_rounds or settings.max_rounds
        ideas_n = ideas_per_round or settings.ideas_per_round
        variants_k = variants_per_idea or settings.variants_per_idea

        # Fetch known fields for validator
        known_fields = await self._fetch_known_fields(dataset, universe)
        self._validator.set_known_fields(known_fields)
        self._synth_agent.set_known_fields(known_fields)

        reports: list[RoundReport] = []

        for round_num in range(1, n_rounds + 1):
            console.rule(f"[bold cyan]Round {round_num}/{n_rounds}[/] — {direction}")
            report = await self._run_round(
                round_num=round_num,
                direction=direction,
                dataset=dataset,
                universe=universe,
                ideas_n=ideas_n,
                variants_k=variants_k,
                bias=explore_exploit_bias,
                dry_run=dry_run,
            )
            report.ended_at = datetime.utcnow()
            reports.append(report)
            self._print_round_summary(report)

        self._print_session_summary(reports)
        return reports

    # ── round execution ───────────────────────────────────────────────────────

    async def _run_round(
        self,
        round_num: int,
        direction: str,
        dataset: str,
        universe: str,
        ideas_n: int,
        variants_k: int,
        bias: float,
        dry_run: bool,
    ) -> RoundReport:
        report = RoundReport(round_num=round_num, direction=direction)

        # Step 1: Generate ideas
        console.print(f"[yellow]Step 1[/] Generating {ideas_n} ideas...")
        ideas = await self._idea_agent.generate_ideas(
            direction=direction,
            dataset=dataset,
            n=ideas_n,
            explore_exploit_bias=bias,
        )
        report.ideas = ideas
        console.print(f"  Got {len(ideas)} ideas")

        # Step 2: Synthesize expressions
        console.print(f"[yellow]Step 2[/] Synthesizing {variants_k} variants per idea...")
        all_exprs: list[str] = []
        idea_map: dict[str, str] = {}  # expr -> hypothesis text
        all_fields_list = list(self._validator._known_fields)

        for idea in ideas:
            variants = await self._synth_agent.synthesize(
                hypothesis=idea,
                all_fields=all_fields_list,
                k=variants_k,
            )
            for expr in variants:
                idea_map[expr] = idea.get("hypothesis", "")
            all_exprs.extend(variants)

        report.expressions_generated = all_exprs
        console.print(f"  Generated {len(all_exprs)} expressions")

        # Step 3: Local validation
        console.print("[yellow]Step 3[/] Validating locally...")
        valid_exprs = [e for e in all_exprs if self._validator.validate(e).ok]
        report.expressions_after_validation = valid_exprs
        console.print(f"  {len(valid_exprs)} passed local validation")

        # Step 4: Novelty filter
        console.print("[yellow]Step 4[/] Novelty filtering...")
        novel_exprs = self._novelty.filter_novel(valid_exprs, threshold=self._novelty_threshold)
        report.expressions_after_novelty = novel_exprs
        console.print(f"  {len(novel_exprs)} passed novelty filter (threshold={self._novelty_threshold})")

        if dry_run or not novel_exprs:
            if dry_run:
                console.print("[dim]  DRY RUN — skipping WQB simulation[/]")
            return report

        # Step 5: WQB simulation
        console.print(f"[yellow]Step 5[/] Simulating {len(novel_exprs)} expressions on WQB...")
        results = await self._client.simulate_batch(novel_exprs, dataset=dataset, universe=universe)
        report.results = results

        # Step 6: Store results + embed
        qualified_results: list[AlphaResult] = []
        for result in results:
            if result is None:
                continue
            hypothesis_text = idea_map.get(result.expression, "")
            embedding = self._store.embed([result.expression])[0]

            self._memory.upsert_alpha(
                alpha_id=result.alpha_id,
                expression=result.expression,
                hypothesis=hypothesis_text,
                dataset=dataset,
                universe=universe,
                metrics=result.metrics.model_dump(),
                checks=result.metrics.checks,
                qualified=result.qualified,
                failure_reasons=result.failure_reasons,
                embedding=embedding,
            )

            if result.qualified:
                qualified_results.append(result)
                report.qualified_ids.append(result.alpha_id)

        console.print(f"  {len(qualified_results)}/{len([r for r in results if r])} qualified")

        # Step 7: Auto-submit if enabled
        if self._auto_submit and qualified_results:
            console.print(f"[yellow]Step 6[/] Auto-submitting {len(qualified_results)} alphas...")
            for result in qualified_results:
                success = await self._client.submit_alpha(result.alpha_id)
                console.print(
                    f"  {'✓' if success else '✗'} {result.alpha_id} "
                    f"(sharpe={result.metrics.sharpe:.2f})"
                )

        # Step 8: Reflect on all results
        console.print("[yellow]Step 7[/] Reflecting on results...")
        hypothesis_map = {r.alpha_id: idea_map.get(r.expression, "") for r in results if r}
        non_none = [r for r in results if r is not None]
        await self._reflector.reflect_batch(non_none, hypotheses=hypothesis_map)

        return report

    # ── helpers ───────────────────────────────────────────────────────────────

    async def _fetch_known_fields(self, dataset: str, universe: str) -> set[str]:
        try:
            fields = await self._client.fetch_datafields(dataset, universe)
            return {f["id"] for f in fields}
        except Exception:
            return set()

    @staticmethod
    def _print_round_summary(report: RoundReport) -> None:
        table = Table(title=f"Round {report.round_num} Summary", show_header=True)
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        for k, v in report.to_dict().items():
            table.add_row(str(k), str(v))
        console.print(table)

    @staticmethod
    def _print_session_summary(reports: list[RoundReport]) -> None:
        total_qualified = sum(len(r.qualified_ids) for r in reports)
        total_simulated = sum(len([x for x in r.results if x]) for r in reports)
        overall_rate = total_qualified / total_simulated if total_simulated else 0
        console.rule("[bold green]Session Complete[/]")
        console.print(
            f"Total rounds: {len(reports)} | "
            f"Simulated: {total_simulated} | "
            f"Qualified: {total_qualified} | "
            f"Pass rate: {overall_rate:.1%}"
        )
