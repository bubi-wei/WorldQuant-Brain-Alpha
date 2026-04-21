"""Skeleton Registry — DuckDB-backed store for FASTEXPR structural templates.

A "skeleton" is a FASTEXPR expression with leaf field references replaced by
positional placeholders ($X1, $X2, ...), numeric constants by window/threshold
placeholders ($W1, $W2, ...), and group-enum strings by $G1, $G2, ...

Skeletons are extracted automatically from successful (and soft-successful)
alpha simulations and used by SkeletonAgent to generate new expressions by
substituting concrete fields/parameters into the fixed structure.
"""

from __future__ import annotations

import json
import math
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb

from alpha_agent.config import settings

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS skeletons (
    skeleton_id         VARCHAR PRIMARY KEY,
    template_str        VARCHAR NOT NULL,
    template_ast_json   VARCHAR DEFAULT '{}',
    operators_used      VARCHAR[]  DEFAULT [],
    field_slots_json    VARCHAR DEFAULT '[]',
    param_slots_json    VARCHAR DEFAULT '[]',
    group_slots_json    VARCHAR DEFAULT '[]',
    origin_alpha_ids    VARCHAR DEFAULT '[]',
    origin_hypothesis   VARCHAR DEFAULT '',
    attempt_count       INTEGER DEFAULT 0,
    success_count       INTEGER DEFAULT 0,
    soft_success_count  INTEGER DEFAULT 0,
    avg_sharpe          FLOAT   DEFAULT 0.0,
    archived            BOOLEAN DEFAULT FALSE,
    last_used_at        TIMESTAMP,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS skeleton_instances (
    id            VARCHAR PRIMARY KEY,
    skeleton_id   VARCHAR NOT NULL,
    expression    VARCHAR NOT NULL,
    alpha_id      VARCHAR DEFAULT '',
    field_combo   VARCHAR DEFAULT '{}',
    param_combo   VARCHAR DEFAULT '{}',
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

_ARCHIVE_THRESHOLD_ATTEMPTS = 20
_ARCHIVE_THRESHOLD_RATE = 0.05


class SkeletonRecord:
    """Value object representing one skeleton row."""

    def __init__(self, row: dict[str, Any]) -> None:
        self.skeleton_id: str = row["skeleton_id"]
        self.template_str: str = row["template_str"]
        self.template_ast_json: str = row.get("template_ast_json", "{}")
        self.operators_used: list[str] = row.get("operators_used") or []
        self.field_slots: list[dict[str, Any]] = json.loads(row.get("field_slots_json") or "[]")
        self.param_slots: list[dict[str, Any]] = json.loads(row.get("param_slots_json") or "[]")
        self.group_slots: list[dict[str, Any]] = json.loads(row.get("group_slots_json") or "[]")
        self.origin_alpha_ids: list[str] = json.loads(row.get("origin_alpha_ids") or "[]")
        self.origin_hypothesis: str = row.get("origin_hypothesis", "")
        self.attempt_count: int = row.get("attempt_count", 0)
        self.success_count: int = row.get("success_count", 0)
        self.soft_success_count: int = row.get("soft_success_count", 0)
        self.avg_sharpe: float = row.get("avg_sharpe", 0.0)
        self.archived: bool = row.get("archived", False)

    @property
    def success_rate(self) -> float:
        if self.attempt_count == 0:
            return 0.0
        return self.success_count / self.attempt_count

    @property
    def soft_rate(self) -> float:
        if self.attempt_count == 0:
            return 0.0
        return (self.success_count + self.soft_success_count) / self.attempt_count


class SkeletonRegistry:
    """DuckDB-backed skeleton store with UCB-based seed selection."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._path = str(db_path or settings.duckdb_path.with_name("skeleton_registry.db"))
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._read_only = False
        try:
            self._conn = duckdb.connect(self._path)
        except Exception as e:
            message = str(e)
            if "Conflicting lock is held" in message:
                self._conn = duckdb.connect(self._path, read_only=True)
                self._read_only = True
                print(
                    "[SkeletonRegistry] DuckDB file is locked by another process. "
                    "Connected in read-only mode."
                )
            else:
                raise
        if not self._read_only:
            self._conn.execute(_CREATE_TABLE)

    def _ensure_writable(self) -> None:
        if self._read_only:
            raise RuntimeError(
                "SkeletonRegistry is in read-only mode due to a DuckDB lock. "
                "Stop other Python kernels/processes using the same DB and retry."
            )

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "SkeletonRegistry":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ── upsert ────────────────────────────────────────────────────────────────

    def upsert(
        self,
        template_str: str,
        template_ast_json: str = "{}",
        operators_used: list[str] | None = None,
        field_slots: list[dict[str, Any]] | None = None,
        param_slots: list[dict[str, Any]] | None = None,
        group_slots: list[dict[str, Any]] | None = None,
        origin_alpha_id: str = "",
        origin_hypothesis: str = "",
    ) -> str:
        """Insert or update a skeleton. Returns skeleton_id.

        If a skeleton with the same template_str already exists,
        merges the origin_alpha_id and returns the existing id.
        """
        self._ensure_writable()
        existing = self._conn.execute(
            "SELECT skeleton_id, origin_alpha_ids FROM skeletons WHERE template_str = ?",
            [template_str],
        ).fetchone()

        if existing:
            sk_id = existing[0]
            ids = json.loads(existing[1] or "[]")
            if origin_alpha_id and origin_alpha_id not in ids:
                ids.append(origin_alpha_id)
                self._conn.execute(
                    "UPDATE skeletons SET origin_alpha_ids = ? WHERE skeleton_id = ?",
                    [json.dumps(ids), sk_id],
                )
            return sk_id

        sk_id = str(uuid.uuid4())
        origin_ids = [origin_alpha_id] if origin_alpha_id else []
        self._conn.execute(
            """
            INSERT INTO skeletons
                (skeleton_id, template_str, template_ast_json, operators_used,
                 field_slots_json, param_slots_json, group_slots_json,
                 origin_alpha_ids, origin_hypothesis, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                sk_id, template_str, template_ast_json,
                operators_used or [],
                json.dumps(field_slots or []),
                json.dumps(param_slots or []),
                json.dumps(group_slots or []),
                json.dumps(origin_ids),
                origin_hypothesis,
                datetime.utcnow().isoformat(),
            ],
        )
        return sk_id

    # ── outcome tracking ──────────────────────────────────────────────────────

    def record_attempt(
        self,
        skeleton_id: str,
        expression: str,
        alpha_id: str = "",
        qualified: bool = False,
        soft_qualified: bool = False,
        sharpe: float = 0.0,
        field_combo: dict[str, Any] | None = None,
        param_combo: dict[str, Any] | None = None,
    ) -> None:
        """Record one instantiation attempt for a skeleton."""
        self._ensure_writable()
        self._conn.execute(
            "INSERT INTO skeleton_instances (id, skeleton_id, expression, alpha_id, field_combo, param_combo, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [str(uuid.uuid4()), skeleton_id, expression, alpha_id,
             json.dumps(field_combo or {}), json.dumps(param_combo or {}),
             datetime.utcnow().isoformat()],
        )

        # update aggregate stats
        row = self._conn.execute(
            "SELECT attempt_count, success_count, soft_success_count, avg_sharpe FROM skeletons WHERE skeleton_id = ?",
            [skeleton_id],
        ).fetchone()
        if not row:
            return

        attempts, successes, soft, avg = row
        new_attempts = attempts + 1
        new_successes = successes + int(qualified)
        new_soft = soft + int(soft_qualified and not qualified)
        if qualified and sharpe > 0:
            new_avg = (avg * successes + sharpe) / new_successes if new_successes else 0.0
        else:
            new_avg = avg

        self._conn.execute(
            """
            UPDATE skeletons SET
                attempt_count = ?,
                success_count = ?,
                soft_success_count = ?,
                avg_sharpe = ?,
                last_used_at = ?
            WHERE skeleton_id = ?
            """,
            [new_attempts, new_successes, new_soft, new_avg,
             datetime.utcnow().isoformat(), skeleton_id],
        )

        # auto-archive chronically failing skeletons
        if new_attempts >= _ARCHIVE_THRESHOLD_ATTEMPTS:
            rate = new_successes / new_attempts
            if rate < _ARCHIVE_THRESHOLD_RATE:
                self._conn.execute(
                    "UPDATE skeletons SET archived = TRUE WHERE skeleton_id = ?",
                    [skeleton_id],
                )

    # ── retrieval ─────────────────────────────────────────────────────────────

    def pick_seeds(
        self,
        strategy: str = "ucb",
        k: int = 5,
        exploration_constant: float = 2.0,
        exclude_archived: bool = True,
    ) -> list[SkeletonRecord]:
        """Return k skeletons to instantiate, sorted by strategy."""
        where = "WHERE archived = FALSE" if exclude_archived else ""
        rows = self._conn.execute(
            f"SELECT * FROM skeletons {where} ORDER BY created_at"
        ).fetchall()
        cols = [d[0] for d in self._conn.description]
        records = [SkeletonRecord(dict(zip(cols, r))) for r in rows]

        if not records:
            return []

        if strategy == "top_sharpe":
            records.sort(key=lambda r: r.avg_sharpe, reverse=True)
        elif strategy == "recent":
            pass  # already ordered by created_at
        else:
            # UCB1: reward = success_rate + c * sqrt(ln(N) / n_i)
            total = sum(r.attempt_count for r in records) or 1
            for rec in records:
                n = rec.attempt_count or 1
                rec._ucb = rec.success_rate + exploration_constant * math.sqrt(
                    math.log(total) / n
                )
            records.sort(key=lambda r: getattr(r, "_ucb", 0), reverse=True)

        return records[:k]

    def get(self, skeleton_id: str) -> SkeletonRecord | None:
        row = self._conn.execute(
            "SELECT * FROM skeletons WHERE skeleton_id = ?", [skeleton_id]
        ).fetchone()
        if not row:
            return None
        cols = [d[0] for d in self._conn.description]
        return SkeletonRecord(dict(zip(cols, row)))

    def all(self, include_archived: bool = False) -> list[SkeletonRecord]:
        where = "" if include_archived else "WHERE archived = FALSE"
        rows = self._conn.execute(f"SELECT * FROM skeletons {where}").fetchall()
        cols = [d[0] for d in self._conn.description]
        return [SkeletonRecord(dict(zip(cols, r))) for r in rows]

    def seen_field_combos(self, skeleton_id: str) -> list[dict[str, Any]]:
        """Return all previously tried field combinations for a skeleton."""
        rows = self._conn.execute(
            "SELECT field_combo FROM skeleton_instances WHERE skeleton_id = ?",
            [skeleton_id],
        ).fetchall()
        return [json.loads(r[0]) for r in rows if r[0] and r[0] != "{}"]

    def count(self, include_archived: bool = False) -> int:
        where = "" if include_archived else "WHERE archived = FALSE"
        return self._conn.execute(f"SELECT COUNT(*) FROM skeletons {where}").fetchone()[0]

    def stats(self) -> dict[str, Any]:
        total = self._conn.execute("SELECT COUNT(*) FROM skeletons").fetchone()[0]
        active = self._conn.execute("SELECT COUNT(*) FROM skeletons WHERE archived = FALSE").fetchone()[0]
        instances = self._conn.execute("SELECT COUNT(*) FROM skeleton_instances").fetchone()[0]
        return {"total": total, "active": active, "instances": instances}

    def to_dataframe(self) -> Any:
        import pandas as pd  # noqa: PLC0415
        return self._conn.execute("SELECT * FROM skeletons ORDER BY created_at DESC").df()
