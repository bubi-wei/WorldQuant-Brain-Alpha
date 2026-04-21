from __future__ import annotations

import asyncio
import json
from pathlib import Path

from alpha_agent.config import settings
from alpha_agent.data.operator_kb import OperatorKB
from alpha_agent.data.wqb_client import WQBClient
from alpha_agent.engine.orchestrator import Orchestrator
from alpha_agent.knowledge.alpha_memory import AlphaMemory
from alpha_agent.knowledge.skeleton_registry import SkeletonRegistry
from alpha_agent.knowledge.vector_store import VectorStore


async def run_one(dataset: str) -> dict:
    vector_store = VectorStore()
    alpha_memory = AlphaMemory()
    skeleton_registry = SkeletonRegistry(
        db_path=settings.duckdb_path.with_name("skeleton_registry.db")
    )
    operator_kb = OperatorKB()

    async with WQBClient() as client:
        orchestrator = Orchestrator(
            client=client,
            vector_store=vector_store,
            alpha_memory=alpha_memory,
            skeleton_registry=skeleton_registry,
            operator_kb=operator_kb,
            model=settings.llm_model,
            auto_submit=False,
        )
        async def _noop_reflect(*args, **kwargs) -> None:
            return None

        # Validation run: skip reflection LLM call to save time/cost.
        orchestrator._reflector.reflect_batch = _noop_reflect  # type: ignore[attr-defined]
        reports = await orchestrator.run(
            direction="skeleton_only_validation_run",
            dataset=dataset,
            universe="TOP1000",
            n_rounds=1,
            ideas_per_round=1,
            variants_per_idea=1,
            explore_exploit_bias=0.4,
            dry_run=False,
            track_mode="skeleton_only",
        )

    report = reports[0]
    result = {
        "dataset": dataset,
        "generated": len(report.expressions_generated),
        "after_validation": len(report.expressions_after_validation),
        "after_novelty": len(report.expressions_after_novelty),
        "simulated": len([r for r in report.results if r is not None]),
        "qualified": len(report.qualified_ids),
        "soft_qualified": len(report.soft_qualified_ids),
        "pass_rate": report.pass_rate,
        "qualified_ids": report.qualified_ids,
    }
    print(
        f"[{dataset}] generated={result['generated']} simulated={result['simulated']} "
        f"qualified={result['qualified']} pass_rate={result['pass_rate']}"
    )
    return result


async def main() -> None:
    settings.ensure_dirs()
    datasets = ["pv1", "fundamental6", "analyst4"]
    all_results = []
    for ds in datasets:
        all_results.append(await run_one(ds))

    out_path = Path("data/skeleton_only_validation.json")
    out_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved summary to: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
