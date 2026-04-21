"""Orchestrator — dual-track alpha generation pipeline.

Two tracks run in parallel each round:
  - Explorer: IdeaAgent → ExprSynthAgent → strict novelty filter
  - Skeleton: SkeletonAgent → field_coverage novelty filter

TrackBandit allocates simulation budget between the two tracks.
After simulation, SkeletonExtractor deposits qualified+soft-qualified alphas
into SkeletonRegistry for future Skeleton-track use.

Usage:
    async with WQBClient() as client:
        orchestrator = Orchestrator(client=client, ...)
        reports = await orchestrator.run(
            direction="cross-sectional momentum with volume confirmation",
            dataset="pv1",
            universe="TOP1000",
            n_rounds=3,
            track_mode="hybrid",   # "explorer_only" | "skeleton_only" | "hybrid"
        )
"""

from __future__ import annotations

import asyncio
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
from alpha_agent.engine.skeleton_agent import SkeletonAgent, SkeletonVariant
from alpha_agent.engine.skeleton_extractor import SkeletonExtractor
from alpha_agent.engine.validator import ExprValidator
from alpha_agent.eval.qualifier import qualify as _qualify_func
from alpha_agent.knowledge.alpha_memory import AlphaMemory
from alpha_agent.knowledge.skeleton_registry import SkeletonRegistry
from alpha_agent.knowledge.vector_store import VectorStore
from alpha_agent.search.novelty import NoveltyScorer
from alpha_agent.search.track_bandit import TrackBandit

console = Console()

_TRACK_EXPLORER = "explorer"
_TRACK_SKELETON = "skeleton"


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
        self.soft_qualified_ids: list[str] = []
        self.skeletons_added: int = 0
        # per-track counters
        self.explorer_simulated: int = 0
        self.explorer_qualified: int = 0
        self.skeleton_simulated: int = 0
        self.skeleton_qualified: int = 0
        self.started_at = datetime.utcnow()
        self.ended_at: datetime | None = None

    @property
    def pass_rate(self) -> float:
        total = len([r for r in self.results if r is not None])
        if total == 0:
            return 0.0
        return round(len(self.qualified_ids) / total, 3)

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
            "soft_qualified": len(self.soft_qualified_ids),
            "skeletons_added": self.skeletons_added,
            "pass_rate": self.pass_rate,
            "explorer_simulated": self.explorer_simulated,
            "explorer_qualified": self.explorer_qualified,
            "skeleton_simulated": self.skeleton_simulated,
            "skeleton_qualified": self.skeleton_qualified,
            "duration_s": (
                (self.ended_at - self.started_at).total_seconds()
                if self.ended_at else None
            ),
        }


class Orchestrator:
    """Dual-track alpha generation pipeline with Explorer + Skeleton agents."""

    def __init__(
        self,
        client: WQBClient,
        vector_store: VectorStore,
        alpha_memory: AlphaMemory,
        skeleton_registry: SkeletonRegistry | None = None,
        operator_kb: OperatorKB | None = None,
        model: str | None = None,
        novelty_threshold: float | None = None,
        auto_submit: bool = False,
    ) -> None:
        self._client = client
        self._store = vector_store
        self._memory = alpha_memory
        self._registry = skeleton_registry or SkeletonRegistry()
        self._kb = operator_kb or OperatorKB()
        self._model = model or settings.llm_model
        self._novelty_threshold = novelty_threshold or settings.novelty_score_min
        self._auto_submit = auto_submit

        # Explorer-track agents
        self._idea_agent = IdeaAgent(vector_store, alpha_memory, model=self._model)
        self._synth_agent = ExprSynthAgent(self._kb, alpha_memory, model=self._model)
        self._validator = ExprValidator(self._kb)
        self._novelty = NoveltyScorer(vector_store, alpha_memory)
        self._reflector = Reflector(alpha_memory, model=self._model)

        # Skeleton-track agents
        self._skeleton_agent = SkeletonAgent(
            skeleton_registry=self._registry,
            vector_store=vector_store,
            model=self._model,
        )
        self._extractor = SkeletonExtractor(model=self._model)
        self._bandit = TrackBandit(skeleton_registry=self._registry)

    # ── public run ────────────────────────────────────────────────────────────

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
        track_mode: str = "hybrid",
    ) -> list[RoundReport]:
        """Execute N rounds of the dual-track generation pipeline.

        Args:
            direction:          Research theme / direction (for Explorer track).
            dataset:            WQB dataset ID.
            universe:           Universe (TOP1000, TOP3000, etc.).
            n_rounds:           Override settings.max_rounds.
            ideas_per_round:    Override settings.ideas_per_round.
            variants_per_idea:  Override settings.variants_per_idea.
            explore_exploit_bias: 0.0=explore, 1.0=exploit (Explorer track).
            dry_run:            Skip WQB simulation.
            track_mode:         "explorer_only" | "skeleton_only" | "hybrid".

        Returns:
            List of RoundReport, one per round.
        """
        n_rounds = n_rounds or settings.max_rounds
        ideas_n = ideas_per_round or settings.ideas_per_round
        variants_k = variants_per_idea or settings.variants_per_idea

        known_fields = await self._fetch_known_fields(dataset, universe)
        self._validator.set_known_fields(known_fields)
        self._synth_agent.set_known_fields(known_fields)
        self._skeleton_agent.set_known_fields(known_fields)

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
                track_mode=track_mode,
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
        track_mode: str,
    ) -> RoundReport:
        report = RoundReport(round_num=round_num, direction=direction)
        total_budget = ideas_n * variants_k

        # TrackBandit decides allocation
        if track_mode == "explorer_only":
            allocation = {_TRACK_EXPLORER: total_budget, _TRACK_SKELETON: 0}
        elif track_mode == "skeleton_only":
            allocation = {_TRACK_EXPLORER: 0, _TRACK_SKELETON: total_budget}
        else:
            allocation = self._bandit.allocate(total_budget)

        console.print(f"[dim]Allocation — explorer={allocation[_TRACK_EXPLORER]} | skeleton={allocation[_TRACK_SKELETON]}[/]")

        # Run both tracks concurrently
        explorer_task = self._run_explorer_track(
            direction=direction,
            dataset=dataset,
            ideas_n=ideas_n,
            variants_k=variants_k,
            bias=bias,
            budget=allocation[_TRACK_EXPLORER],
        ) if allocation[_TRACK_EXPLORER] > 0 else _empty_explorer_track()

        skeleton_task = self._run_skeleton_track(
            dataset=dataset,
            budget=allocation[_TRACK_SKELETON],
        ) if allocation[_TRACK_SKELETON] > 0 else _empty_skeleton_track()

        (explorer_exprs, explorer_idea_map), (skeleton_variants,) = await asyncio.gather(
            explorer_task,
            skeleton_task,
        )

        report.ideas = list(explorer_idea_map.values())
        report.expressions_generated = explorer_exprs + [v.expression for v in skeleton_variants]

        # Validate
        valid_explorer = [e for e in explorer_exprs if self._validator.validate(e).ok]
        valid_skeleton = [v for v in skeleton_variants if self._validator.validate(v.expression).ok]
        report.expressions_after_validation = [e for e in valid_explorer] + [v.expression for v in valid_skeleton]

        # Novelty filter — different modes per track
        novel_explorer = self._novelty.filter_novel(valid_explorer, threshold=self._novelty_threshold, mode="strict")
        novel_skeleton_variants = self._novelty_filter_skeleton(valid_skeleton)
        all_novel = novel_explorer + [v.expression for v in novel_skeleton_variants]
        report.expressions_after_novelty = all_novel

        if dry_run or not all_novel:
            if dry_run:
                console.print("[dim]  DRY RUN — skipping WQB simulation[/]")
            return report

        # WQB simulation
        console.print(f"[yellow]Simulating[/] {len(all_novel)} expressions on WQB...")
        results = await self._client.simulate_batch(all_novel, dataset=dataset, universe=universe)
        report.results = results

        # Build per-track expression sets for bookkeeping
        explorer_set = set(novel_explorer)
        skeleton_expr_map: dict[str, SkeletonVariant] = {v.expression: v for v in novel_skeleton_variants}

        # Store results and extract skeletons
        idea_map_reversed: dict[str, str] = {}
        for expr, hyp in explorer_idea_map.items():
            idea_map_reversed[expr] = hyp

        for result in results:
            if result is None:
                continue

            hypothesis_text = idea_map_reversed.get(result.expression, "")
            embedding = self._store.embed([result.expression])[0]
            track = _TRACK_SKELETON if result.expression in skeleton_expr_map else _TRACK_EXPLORER
            sk_variant = skeleton_expr_map.get(result.expression)

            qual_result = _qualify_func(result.metrics.model_dump())
            is_qualified = result.qualified
            is_soft = qual_result.soft_qualified

            # Determine skeleton_id for skeleton-track results
            skeleton_id = ""
            if sk_variant:
                skeleton_id = sk_variant.skeleton_id
                self._registry.record_attempt(
                    skeleton_id=skeleton_id,
                    expression=result.expression,
                    alpha_id=result.alpha_id,
                    qualified=is_qualified,
                    soft_qualified=is_soft,
                    sharpe=float(result.metrics.sharpe or 0),
                    field_combo=sk_variant.field_mapping,
                    param_combo=sk_variant.param_mapping,
                )

            self._memory.upsert_alpha(
                alpha_id=result.alpha_id,
                expression=result.expression,
                hypothesis=hypothesis_text,
                dataset=dataset,
                universe=universe,
                metrics=result.metrics.model_dump(),
                checks=result.metrics.checks,
                qualified=is_qualified,
                soft_qualified=is_soft,
                failure_reasons=result.failure_reasons,
                skeleton_id=skeleton_id,
                track=track,
                embedding=embedding,
            )

            # Update track stats for bandit
            if track == _TRACK_EXPLORER:
                report.explorer_simulated += 1
                if is_qualified:
                    report.explorer_qualified += 1
            else:
                report.skeleton_simulated += 1
                if is_qualified:
                    report.skeleton_qualified += 1

            if is_qualified:
                report.qualified_ids.append(result.alpha_id)
            if is_soft:
                report.soft_qualified_ids.append(result.alpha_id)

            # Extract skeleton from qualified/soft-qualified Explorer results
            if track == _TRACK_EXPLORER and (is_qualified or (settings.soft_enable_skeleton and is_soft)):
                await self._deposit_skeleton(result, hypothesis_text)
                report.skeletons_added += 1

        # Feed back to bandit
        self._bandit.update(_TRACK_EXPLORER, report.explorer_simulated, report.explorer_qualified)
        self._bandit.update(_TRACK_SKELETON, report.skeleton_simulated, report.skeleton_qualified)

        # Auto-submit
        if self._auto_submit and report.qualified_ids:
            qualified_results = [r for r in results if r and r.alpha_id in set(report.qualified_ids)]
            console.print(f"[yellow]Auto-submitting[/] {len(qualified_results)} alphas...")
            for r in qualified_results:
                ok = await self._client.submit_alpha(r.alpha_id)
                console.print(f"  {'✓' if ok else '✗'} {r.alpha_id}")

        # Reflect
        console.print("[yellow]Reflecting[/] on results...")
        hyp_map = {r.alpha_id: idea_map_reversed.get(r.expression, "") for r in results if r}
        await self._reflector.reflect_batch([r for r in results if r], hypotheses=hyp_map)

        return report

    # ── Explorer track ────────────────────────────────────────────────────────

    async def _run_explorer_track(
        self,
        direction: str,
        dataset: str,
        ideas_n: int,
        variants_k: int,
        bias: float,
        budget: int,
    ) -> tuple[list[str], dict[str, str]]:
        """Generate expressions via IdeaAgent + ExprSynthAgent."""
        console.print(f"[cyan][Explorer][/] Generating {ideas_n} ideas...")
        ideas = await self._idea_agent.generate_ideas(
            direction=direction,
            dataset=dataset,
            n=ideas_n,
            explore_exploit_bias=bias,
        )

        all_exprs: list[str] = []
        idea_map: dict[str, str] = {}
        all_fields_list = list(self._validator._known_fields)

        console.print(f"[cyan][Explorer][/] Synthesizing {variants_k} variants/idea...")
        for idea in ideas:
            variants = await self._synth_agent.synthesize(
                hypothesis=idea,
                all_fields=all_fields_list,
                k=variants_k,
            )
            for expr in variants:
                idea_map[expr] = idea.get("hypothesis", "")
            all_exprs.extend(variants)

        # Cap to budget
        all_exprs = all_exprs[:budget]
        idea_map = {k: v for k, v in idea_map.items() if k in set(all_exprs)}
        console.print(f"[cyan][Explorer][/] Generated {len(all_exprs)} expressions")
        return all_exprs, idea_map

    # ── Skeleton track ────────────────────────────────────────────────────────

    async def _run_skeleton_track(
        self,
        dataset: str,
        budget: int,
    ) -> tuple[list[SkeletonVariant]]:
        """Generate expressions via SkeletonAgent."""
        n_seeds = max(1, budget // settings.skeleton_variants_per_seed)
        console.print(f"[green][Skeleton][/] Generating variants from {n_seeds} seeds...")
        variants = await self._skeleton_agent.generate(
            dataset=dataset,
            k_per_skeleton=settings.skeleton_variants_per_seed,
            max_seeds=n_seeds,
            pick_strategy=settings.skeleton_pick_strategy,
        )
        # Cap to budget
        variants = variants[:budget]
        console.print(f"[green][Skeleton][/] Got {len(variants)} variants")
        return (variants,)

    def _novelty_filter_skeleton(
        self,
        variants: list[SkeletonVariant],
        threshold: float = 0.1,
    ) -> list[SkeletonVariant]:
        """Filter skeleton variants by field_coverage novelty."""
        result = []
        for v in variants:
            score = self._novelty.score_expression(
                expression=v.expression,
                mode="field_coverage",
                skeleton_id=v.skeleton_id,
                field_mapping=v.field_mapping,
            )
            if score >= threshold:
                result.append(v)
        return result

    # ── skeleton deposit ──────────────────────────────────────────────────────

    async def _deposit_skeleton(
        self,
        result: AlphaResult,
        hypothesis: str,
    ) -> None:
        """Extract skeleton from a successful alpha and store in registry."""
        try:
            template = await self._extractor.extract_with_hints(
                result.expression,
                hypothesis=hypothesis,
            )
            if template is None:
                return
            sk_id = self._registry.upsert(
                template_str=template.template_str,
                template_ast_json=template.template_ast_json,
                operators_used=template.operators_used,
                field_slots=template.field_slots,
                param_slots=template.param_slots,
                group_slots=template.group_slots,
                origin_alpha_id=result.alpha_id,
                origin_hypothesis=hypothesis,
            )
            self._memory.update_skeleton_link(result.alpha_id, sk_id)
        except Exception as e:
            console.print(f"[dim][Skeleton] Extraction failed: {e}[/]")

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


async def _empty_explorer_track() -> tuple[list[str], dict[str, str]]:
    """No-op coroutine for the Explorer track."""
    return [], {}


async def _empty_skeleton_track() -> tuple[list[SkeletonVariant]]:
    """No-op coroutine for the Skeleton track."""
    return ([],)
