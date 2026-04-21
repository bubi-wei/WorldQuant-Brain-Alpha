"""Global configuration — reads from .env via pydantic-settings."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    _project_root: Path = Path(__file__).resolve().parent.parent

    model_config = SettingsConfigDict(
        env_file=str(_project_root / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── WQB credentials ──────────────────────────────────────
    wqb_username: str = Field(default="", alias="WQB_USERNAME")
    wqb_password: str = Field(default="", alias="WQB_PASSWORD")

    # ── LLM ──────────────────────────────────────────────────
    llm_model: str = Field(default="openai/gpt-4o", alias="LLM_MODEL")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")

    # ── Embeddings ────────────────────────────────────────────
    embed_model: str = Field(default="all-MiniLM-L6-v2", alias="EMBED_MODEL")

    # ── Storage ───────────────────────────────────────────────
    chroma_persist_dir: Path = Field(default=_project_root / "data" / "chroma", alias="CHROMA_PERSIST_DIR")
    duckdb_path: Path = Field(default=_project_root / "data" / "alpha_memory.db", alias="DUCKDB_PATH")

    # ── WQB simulation defaults ───────────────────────────────
    wqb_instrument_type: str = Field(default="EQUITY", alias="WQB_INSTRUMENT_TYPE")
    wqb_region: str = Field(default="USA", alias="WQB_REGION")
    wqb_delay: int = Field(default=1, alias="WQB_DELAY")
    wqb_decay: int = Field(default=0, alias="WQB_DECAY")
    wqb_neutralization: str = Field(default="SUBINDUSTRY", alias="WQB_NEUTRALIZATION")
    wqb_truncation: float = Field(default=0.08, alias="WQB_TRUNCATION")
    wqb_concurrency: int = Field(default=3, alias="WQB_CONCURRENCY")

    # ── Qualification thresholds ──────────────────────────────
    qual_sharpe_min: float = Field(default=1.5, alias="QUAL_SHARPE_MIN")
    qual_fitness_min: float = Field(default=1.0, alias="QUAL_FITNESS_MIN")
    qual_turnover_min: float = Field(default=0.1, alias="QUAL_TURNOVER_MIN")
    qual_turnover_max: float = Field(default=0.9, alias="QUAL_TURNOVER_MAX")
    qual_ic_mean_min: float = Field(default=0.02, alias="QUAL_IC_MEAN_MIN")

    # ── Generation loop ───────────────────────────────────────
    ideas_per_round: int = Field(default=5, alias="IDEAS_PER_ROUND")
    variants_per_idea: int = Field(default=3, alias="VARIANTS_PER_IDEA")
    max_rounds: int = Field(default=10, alias="MAX_ROUNDS")
    novelty_score_min: float = Field(default=0.3, alias="NOVELTY_SCORE_MIN")

    # ── Dual-track skeleton system ────────────────────────────────────────────
    track_explorer_ratio: float = Field(default=0.3, alias="TRACK_EXPLORER_RATIO")
    skeleton_min_seeds: int = Field(default=3, alias="SKELETON_MIN_SEEDS")
    skeleton_variants_per_seed: int = Field(default=5, alias="SKELETON_VARIANTS_PER_SEED")
    soft_enable_skeleton: bool = Field(default=True, alias="SOFT_ENABLE_SKELETON")
    skeleton_pick_strategy: str = Field(default="ucb", alias="SKELETON_PICK_STRATEGY")
    explorer_floor: float = Field(default=0.1, alias="EXPLORER_FLOOR")

    @model_validator(mode="after")
    def _normalize_storage_paths(self) -> "Settings":
        """Resolve relative storage paths against project root for stable notebook behavior."""
        if not self.chroma_persist_dir.is_absolute():
            self.chroma_persist_dir = (self._project_root / self.chroma_persist_dir).resolve()
        if not self.duckdb_path.is_absolute():
            self.duckdb_path = (self._project_root / self.duckdb_path).resolve()
        return self

    def ensure_dirs(self) -> None:
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        self.duckdb_path.parent.mkdir(parents=True, exist_ok=True)


# Module-level singleton — import this everywhere
settings = Settings()
