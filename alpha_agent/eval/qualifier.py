"""Alpha qualification — configurable thresholds for pass/fail classification.

Extracted from brain_batch_alpha.py and made threshold-configurable
via settings. Supports both hard-filter and soft-filter modes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from alpha_agent.config import settings


@dataclass
class QualificationResult:
    qualified: bool
    soft_qualified: bool
    failure_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics_summary: dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        status = "PASS" if self.qualified else ("SOFT" if self.soft_qualified else "FAIL")
        return f"{status} | {', '.join(self.failure_reasons or ['all checks ok'])}"


def qualify(
    is_data: dict[str, Any],
    soft_sharpe_min: float | None = None,
) -> QualificationResult:
    """Classify an alpha's 'is' metric block as pass/fail/soft-pass.

    Args:
        is_data:         The 'is' dict from the WQB alpha detail endpoint.
        soft_sharpe_min: Optional lower Sharpe threshold for soft qualification.
                         Soft alphas are stored but not auto-submitted.

    Returns:
        QualificationResult
    """
    sharpe = float(is_data.get("sharpe", 0))
    fitness = float(is_data.get("fitness", 0))
    turnover = float(is_data.get("turnover", 0))
    ic_mean = float(is_data.get("margin", 0))
    checks = is_data.get("checks", [])

    failure_reasons: list[str] = []
    warnings: list[str] = []

    if sharpe < settings.qual_sharpe_min:
        failure_reasons.append(f"sharpe={sharpe:.3f} < {settings.qual_sharpe_min}")
    if fitness < settings.qual_fitness_min:
        failure_reasons.append(f"fitness={fitness:.3f} < {settings.qual_fitness_min}")
    if not (settings.qual_turnover_min <= turnover <= settings.qual_turnover_max):
        failure_reasons.append(
            f"turnover={turnover:.3f} ∉ [{settings.qual_turnover_min}, {settings.qual_turnover_max}]"
        )
    if ic_mean < settings.qual_ic_mean_min:
        failure_reasons.append(f"ic_mean={ic_mean:.3f} < {settings.qual_ic_mean_min}")

    for check in checks:
        result = check.get("result", "")
        if result == "FAIL":
            failure_reasons.append(
                f"{check.get('name')}: {check.get('value')} (limit {check.get('limit')})"
            )
        elif result == "PENDING":
            warnings.append(f"{check.get('name')} check still PENDING")

    qualified = len(failure_reasons) == 0

    # Soft qualification: relax Sharpe threshold by 20% optionally
    soft_sharpe = soft_sharpe_min if soft_sharpe_min is not None else settings.qual_sharpe_min * 0.8
    soft_reasons = [r for r in failure_reasons if not r.startswith("sharpe")]
    soft_sharpe_ok = sharpe >= soft_sharpe
    soft_qualified = qualified or (soft_sharpe_ok and len(soft_reasons) == 0)

    return QualificationResult(
        qualified=qualified,
        soft_qualified=soft_qualified,
        failure_reasons=failure_reasons,
        warnings=warnings,
        metrics_summary={
            "sharpe": sharpe,
            "fitness": fitness,
            "turnover": turnover,
            "ic_mean": ic_mean,
        },
    )
