"""Persistent alpha memory backed by DuckDB.

Stores all historical alpha expressions, their metrics, failure reasons,
and LLM reflections. Provides retrieval functions used by RAG prompting.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np

from alpha_agent.config import settings

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS alphas (
    id              VARCHAR PRIMARY KEY,
    expression      VARCHAR NOT NULL,
    hypothesis      VARCHAR DEFAULT '',
    dataset         VARCHAR DEFAULT '',
    universe        VARCHAR DEFAULT '',
    metrics_json    VARCHAR DEFAULT '{}',
    checks_json     VARCHAR DEFAULT '[]',
    qualified       BOOLEAN DEFAULT FALSE,
    failure_reasons VARCHAR DEFAULT '[]',
    reflection      VARCHAR DEFAULT '',
    embedding       FLOAT[],
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS directions (
    id          VARCHAR PRIMARY KEY,
    name        VARCHAR NOT NULL,
    description VARCHAR DEFAULT '',
    tries       INTEGER DEFAULT 0,
    wins        INTEGER DEFAULT 0,
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


class AlphaMemory:
    """DuckDB-backed store for alpha history, metrics, and reflections."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._path = str(db_path or settings.duckdb_path)
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._read_only = False
        self._ephemeral = False
        try:
            self._conn = duckdb.connect(self._path)
        except Exception as e:
            message = str(e)
            # If another live kernel/process holds a write lock, fall back to read-only
            # so notebooks can still run retrieval/analysis workflows.
            if "Conflicting lock is held" in message:
                try:
                    self._conn = duckdb.connect(self._path, read_only=True)
                    self._read_only = True
                    print(
                        "[AlphaMemory] DuckDB file is locked by another process. "
                        "Connected in read-only mode."
                    )
                except Exception:
                    # Some platforms still require a lock even for read-only open.
                    # Use an in-memory DB so notebook demos can continue running.
                    self._conn = duckdb.connect(":memory:")
                    self._read_only = True
                    self._ephemeral = True
                    print(
                        "[AlphaMemory] DuckDB file lock prevents file access. "
                        "Using in-memory read-only fallback for this session."
                    )
            else:
                raise
        self._conn.execute(_CREATE_TABLE)

    def _ensure_writable(self) -> None:
        if self._read_only:
            suffix = " (in-memory fallback)" if self._ephemeral else ""
            raise RuntimeError(
                "AlphaMemory is in read-only mode due to a DuckDB lock. "
                f"Stop other Python kernels/processes using the same DB and retry{suffix}."
            )

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "AlphaMemory":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ── insert / update ───────────────────────────────────────────────────────

    def upsert_alpha(
        self,
        *,
        alpha_id: str | None = None,
        expression: str,
        hypothesis: str = "",
        dataset: str = "",
        universe: str = "",
        metrics: dict[str, Any] | None = None,
        checks: list[dict[str, Any]] | None = None,
        qualified: bool = False,
        failure_reasons: list[str] | None = None,
        reflection: str = "",
        embedding: list[float] | None = None,
    ) -> str:
        self._ensure_writable()
        row_id = alpha_id or str(uuid.uuid4())
        self._conn.execute(
            """
            INSERT OR REPLACE INTO alphas
                (id, expression, hypothesis, dataset, universe,
                 metrics_json, checks_json, qualified, failure_reasons,
                 reflection, embedding, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                row_id,
                expression,
                hypothesis,
                dataset,
                universe,
                json.dumps(metrics or {}),
                json.dumps(checks or []),
                qualified,
                json.dumps(failure_reasons or []),
                reflection,
                embedding,
                datetime.utcnow().isoformat(),
            ],
        )
        return row_id

    def update_reflection(self, alpha_id: str, reflection: str) -> None:
        self._ensure_writable()
        self._conn.execute(
            "UPDATE alphas SET reflection = ? WHERE id = ?",
            [reflection, alpha_id],
        )

    def update_embedding(self, alpha_id: str, embedding: list[float]) -> None:
        self._ensure_writable()
        self._conn.execute(
            "UPDATE alphas SET embedding = ? WHERE id = ?",
            [embedding, alpha_id],
        )

    # ── retrieval ─────────────────────────────────────────────────────────────

    def similar(self, embedding: list[float], k: int = 5) -> list[dict[str, Any]]:
        """Return k most similar alphas by cosine similarity of their embeddings."""
        rows = self._conn.execute(
            "SELECT id, expression, hypothesis, metrics_json, qualified, reflection, embedding "
            "FROM alphas WHERE embedding IS NOT NULL"
        ).fetchall()

        if not rows:
            return []

        query_vec = np.array(embedding, dtype=np.float32)
        scored: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            emb = np.array(row[6], dtype=np.float32)
            norm = np.linalg.norm(query_vec) * np.linalg.norm(emb)
            sim = float(np.dot(query_vec, emb) / norm) if norm > 0 else 0.0
            scored.append((sim, {
                "id": row[0],
                "expression": row[1],
                "hypothesis": row[2],
                "metrics": json.loads(row[3]),
                "qualified": row[4],
                "reflection": row[5],
                "similarity": sim,
            }))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:k]]

    def top_by_metric(
        self,
        metric: str = "sharpe",
        k: int = 5,
        qualified_only: bool = True,
    ) -> list[dict[str, Any]]:
        """Return top-k alphas sorted by a specific metric extracted from metrics_json."""
        where = "WHERE qualified = TRUE" if qualified_only else ""
        rows = self._conn.execute(
            f"SELECT id, expression, hypothesis, metrics_json, qualified, reflection "
            f"FROM alphas {where}"
        ).fetchall()

        scored: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            metrics = json.loads(row[3])
            value = float(metrics.get(metric, 0.0))
            scored.append((value, {
                "id": row[0],
                "expression": row[1],
                "hypothesis": row[2],
                "metrics": metrics,
                "qualified": row[4],
                "reflection": row[5],
            }))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:k]]

    def failed_reasons_for(self, field: str, limit: int = 20) -> list[str]:
        """Return failure reason strings for alphas that contain a given field name."""
        rows = self._conn.execute(
            "SELECT failure_reasons FROM alphas "
            "WHERE qualified = FALSE AND expression LIKE ?",
            [f"%{field}%"],
        ).fetchall()

        reasons: list[str] = []
        for row in rows[:limit]:
            reasons.extend(json.loads(row[0]))
        return list(dict.fromkeys(reasons))  # deduplicate preserving order

    def recent(self, n: int = 20) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT id, expression, hypothesis, metrics_json, qualified, reflection, created_at "
            "FROM alphas ORDER BY created_at DESC LIMIT ?",
            [n],
        ).fetchall()
        return [
            {
                "id": r[0], "expression": r[1], "hypothesis": r[2],
                "metrics": json.loads(r[3]), "qualified": r[4],
                "reflection": r[5], "created_at": r[6],
            }
            for r in rows
        ]

    def stats(self) -> dict[str, Any]:
        total = self._conn.execute("SELECT COUNT(*) FROM alphas").fetchone()[0]
        qualified = self._conn.execute(
            "SELECT COUNT(*) FROM alphas WHERE qualified = TRUE"
        ).fetchone()[0]
        return {
            "total": total,
            "qualified": qualified,
            "pass_rate": round(qualified / total, 3) if total > 0 else 0.0,
        }

    def to_dataframe(self) -> Any:
        """Return all alphas as a pandas DataFrame for Notebook analysis."""
        import pandas as pd  # noqa: PLC0415
        return self._conn.execute(
            "SELECT id, expression, hypothesis, dataset, universe, "
            "metrics_json, qualified, failure_reasons, reflection, created_at "
            "FROM alphas ORDER BY created_at DESC"
        ).df()

    # ── direction tracking (for bandit) ──────────────────────────────────────

    def record_direction_outcome(self, direction_name: str, won: bool) -> None:
        self._ensure_writable()
        self._conn.execute(
            """
            INSERT INTO directions (id, name, tries, wins, updated_at)
            VALUES (gen_random_uuid()::varchar, ?, 1, ?, CURRENT_TIMESTAMP)
            ON CONFLICT (name) DO UPDATE SET
                tries = directions.tries + 1,
                wins  = directions.wins + excluded.wins,
                updated_at = CURRENT_TIMESTAMP
            """,
            [direction_name, int(won)],
        )

    def direction_stats(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT name, tries, wins FROM directions ORDER BY wins DESC"
        ).fetchall()
        return [{"name": r[0], "tries": r[1], "wins": r[2]} for r in rows]
