"""Async WorldQuant Brain API client.

Refactored from brain_batch_alpha.py with:
- httpx.AsyncClient for concurrent simulations
- asyncio.Semaphore for rate-limiting
- Automatic Retry-After polling
- 401 re-authentication with tenacity
- Structured pydantic AlphaResult objects
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from alpha_agent.config import settings

API_BASE = "https://api.worldquantbrain.com"
_SIMULATION_POST_MAX_ATTEMPTS = 4
_SIMULATION_POLL_MAX_WAIT_SECONDS = 30 * 60
_SIMULATION_POLL_FALLBACK_SLEEP_SECONDS = 3.0
_REQUEST_MAX_ATTEMPTS = 4
_REQUEST_RETRY_BASE_SECONDS = 1.5


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("alpha_agent.wqb_client")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    log_path = Path(__file__).resolve().parents[2] / "data" / "wqb_client.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


logger = _build_logger()


# ── Pydantic response models ──────────────────────────────────────────────────

class AlphaMetrics(BaseModel):
    sharpe: float = 0.0
    fitness: float = 0.0
    turnover: float = 0.0
    margin: float = 0.0       # IC Mean in WQB terminology
    checks: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def ic_mean(self) -> float:
        return self.margin


class AlphaResult(BaseModel):
    alpha_id: str
    expression: str
    dataset: str
    settings_json: dict[str, Any]
    metrics: AlphaMetrics
    qualified: bool
    failure_reasons: list[str] = Field(default_factory=list)
    simulated_at: datetime = Field(default_factory=datetime.utcnow)

    def to_memory_row(self) -> dict[str, Any]:
        return {
            "id": self.alpha_id,
            "expression": self.expression,
            "dataset": self.dataset,
            "metrics_json": self.metrics.model_dump_json(),
            "checks_json": json.dumps(self.metrics.checks),
            "qualified": self.qualified,
            "failure_reasons": json.dumps(self.failure_reasons),
            "simulated_at": self.simulated_at.isoformat(),
        }


class AuthError(Exception):
    pass


# ── Client ────────────────────────────────────────────────────────────────────

class WQBClient:
    """Async WQB API client with concurrency control and auto-retry."""

    def __init__(
        self,
        username: str | None = None,
        password: str | None = None,
        concurrency: int | None = None,
    ) -> None:
        self._username = username or settings.wqb_username
        self._password = password or settings.wqb_password
        self._concurrency = concurrency or settings.wqb_concurrency
        self._semaphore = asyncio.Semaphore(self._concurrency)
        self._client: httpx.AsyncClient | None = None

    def _create_http_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=API_BASE,
            timeout=httpx.Timeout(60.0),
            # HTTP/2 may intermittently trigger RemoteProtocolError with some proxies/gateways.
            http2=False,
            # Default to ignore system proxy variables for stability.
            trust_env=settings.wqb_trust_env,
        )

    # ── context manager ───────────────────────────────────────────────────────

    async def __aenter__(self) -> "WQBClient":
        self._client = self._create_http_client()
        await self._authenticate()
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._client:
            await self._client.aclose()

    async def _refresh_session(self) -> None:
        """Recreate HTTP client and re-authenticate after transport-level failures."""
        old_client = self._client
        self._client = self._create_http_client()
        try:
            await self._authenticate()
        finally:
            if old_client is not None:
                await old_client.aclose()

    # ── authentication ────────────────────────────────────────────────────────

    async def _authenticate(self) -> None:
        assert self._client is not None
        resp = await self._client.post(
            "/authentication",
            auth=(self._username, self._password),
        )
        if resp.status_code not in (200, 201):
            raise AuthError(f"WQB authentication failed: HTTP {resp.status_code} — {resp.text}")

    @retry(
        retry=retry_if_exception_type(AuthError),
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        assert self._client is not None
        last_error: Exception | None = None
        for attempt in range(1, _REQUEST_MAX_ATTEMPTS + 1):
            try:
                resp = await self._client.request(method, path, **kwargs)
                if resp.status_code == 401:
                    await self._authenticate()
                    resp = await self._client.request(method, path, **kwargs)
                return resp
            except (httpx.TimeoutException, httpx.TransportError, httpx.RemoteProtocolError) as e:
                last_error = e
                if attempt == _REQUEST_MAX_ATTEMPTS:
                    break
                backoff = min(10.0, _REQUEST_RETRY_BASE_SECONDS * attempt)
                logger.warning(
                    f"[WQBClient] Request transient error "
                    f"attempt={attempt}/{_REQUEST_MAX_ATTEMPTS} method={method} path={path} "
                    f"error={type(e).__name__}: {e}; retry_in={backoff}s"
                )
                try:
                    await self._refresh_session()
                except Exception as refresh_error:
                    logger.warning(
                        f"[WQBClient] Session refresh failed "
                        f"error={type(refresh_error).__name__}: {refresh_error}"
                    )
                await asyncio.sleep(backoff)

        assert last_error is not None
        raise last_error

    # ── data fields ───────────────────────────────────────────────────────────

    async def fetch_datafields(
        self,
        dataset_id: str,
        universe: str = "TOP3000",
        delay: int = 1,
        field_type: str = "MATRIX",
        page_size: int = 50,
    ) -> list[dict[str, Any]]:
        """Fetch all data fields for a dataset, filtered by type."""
        base_params = {
            "instrumentType": settings.wqb_instrument_type,
            "region": settings.wqb_region,
            "delay": str(delay),
            "universe": universe,
            "dataset.id": dataset_id,
            "limit": page_size,
            "offset": 0,
        }

        first = await self._request("GET", "/data-fields", params={**base_params, "offset": 0})
        first.raise_for_status()
        total = first.json()["count"]

        tasks = [
            self._request("GET", "/data-fields", params={**base_params, "offset": offset})
            for offset in range(0, total, page_size)
        ]
        pages: list[httpx.Response] = []
        batch_size = 10
        for i in range(0, len(tasks), batch_size):
            batch_pages = await asyncio.gather(*tasks[i:i + batch_size])
            pages.extend(batch_pages)

        all_fields: list[dict[str, Any]] = []
        for page in pages:
            if page.status_code == 200:
                results = page.json().get("results", [])
                if field_type:
                    results = [f for f in results if f.get("type") == field_type]
                all_fields.extend(results)

        return all_fields

    # ── simulation ────────────────────────────────────────────────────────────

    async def simulate_one(
        self,
        expression: str,
        dataset: str,
        universe: str = "TOP3000",
    ) -> AlphaResult | None:
        """Simulate a single alpha expression and return structured result."""
        payload = self._build_payload(expression, universe)
        async with self._semaphore:
            return await self._run_simulation(expression, dataset, payload)

    async def simulate_batch(
        self,
        expressions: list[str],
        dataset: str,
        universe: str = "TOP3000",
    ) -> list[AlphaResult | None]:
        """Simulate multiple expressions concurrently (rate-limited by semaphore)."""
        tasks = [self.simulate_one(expr, dataset, universe) for expr in expressions]
        return list(await asyncio.gather(*tasks, return_exceptions=False))

    def _build_payload(self, expression: str, universe: str) -> dict[str, Any]:
        return {
            "type": "REGULAR",
            "settings": {
                "instrumentType": settings.wqb_instrument_type,
                "region": settings.wqb_region,
                "universe": universe,
                "delay": settings.wqb_delay,
                "decay": settings.wqb_decay,
                "neutralization": settings.wqb_neutralization,
                "truncation": settings.wqb_truncation,
                "pasteurization": "ON",
                "unitHandling": "VERIFY",
                "nanHandling": "ON",
                "language": "FASTEXPR",
                "visualization": False,
            },
            "regular": expression,
        }

    async def _run_simulation(
        self,
        expression: str,
        dataset: str,
        payload: dict[str, Any],
    ) -> AlphaResult | None:
        assert self._client is not None
        try:
            sim_resp: httpx.Response | None = None
            for attempt in range(1, _SIMULATION_POST_MAX_ATTEMPTS + 1):
                sim_resp = await self._request("POST", "/simulations", json=payload)
                if sim_resp.status_code == 201:
                    break

                retry_after = float(
                    sim_resp.headers.get("Retry-After", _SIMULATION_POLL_FALLBACK_SLEEP_SECONDS)
                )
                should_retry = sim_resp.status_code in {429, 500, 502, 503, 504}
                logger.warning(
                    f"[WQBClient] POST /simulations failed "
                    f"attempt={attempt}/{_SIMULATION_POST_MAX_ATTEMPTS} "
                    f"status={sim_resp.status_code} expr_full={expression!r} "
                    f"body={sim_resp.text[:240]!r}"
                )
                if not should_retry or attempt == _SIMULATION_POST_MAX_ATTEMPTS:
                    return None
                await asyncio.sleep(max(1.0, retry_after))

            if sim_resp is None or sim_resp.status_code != 201:
                return None

            progress_url = sim_resp.headers.get("Location", "")
            if not progress_url:
                logger.warning(
                    f"[WQBClient] Missing simulation progress URL "
                    f"expr_full={expression!r}"
                )
                return None

            alpha_id = await self._poll_simulation(progress_url)
            if not alpha_id:
                logger.warning(
                    f"[WQBClient] Simulation polling timed out "
                    f"expr_full={expression!r}"
                )
                return None

            detail_resp = await self._request("GET", f"/alphas/{alpha_id}")
            for _ in range(5):
                if detail_resp.status_code != 202:
                    break
                await asyncio.sleep(1)
                detail_resp = await self._request("GET", f"/alphas/{alpha_id}")

            if detail_resp.status_code == 202:
                logger.warning(
                    f"[WQBClient] Alpha detail still pending after retries "
                    f"alpha_id={alpha_id} expr_full={expression!r}"
                )
                return None

            if detail_resp.status_code != 200:
                logger.warning(
                    f"[WQBClient] GET /alphas/{alpha_id} failed "
                    f"status={detail_resp.status_code} expr_full={expression!r} "
                    f"body={detail_resp.text[:240]!r}"
                )
                return None

            detail_resp.raise_for_status()
            alpha_data = detail_resp.json()

            if "is" not in alpha_data:
                logger.warning(
                    f"[WQBClient] Alpha detail missing 'is' payload "
                    f"alpha_id={alpha_id} expr_full={expression!r} "
                    f"keys={list(alpha_data.keys())[:12]}"
                )
                return None

            return self._parse_result(alpha_id, expression, dataset, payload, alpha_data)

        except Exception as e:
            logger.exception(
                f"[WQBClient] Simulation exception expr_full={expression!r} "
                f"error={type(e).__name__}: {e}"
            )
            return None

    async def _poll_simulation(self, url: str) -> str | None:
        """Poll until simulation completes; returns alpha_id on success."""
        started = asyncio.get_event_loop().time()
        transient_failures = 0
        while True:
            if (asyncio.get_event_loop().time() - started) >= _SIMULATION_POLL_MAX_WAIT_SECONDS:
                return None

            try:
                resp = await self._request("GET", url)
            except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPError) as e:
                transient_failures += 1
                backoff = min(
                    10.0,
                    _SIMULATION_POLL_FALLBACK_SLEEP_SECONDS * max(1, transient_failures),
                )
                logger.warning(
                    f"[WQBClient] Poll transient error count={transient_failures} "
                    f"error={type(e).__name__}: {e}"
                )
                await asyncio.sleep(backoff)
                continue

            retry_after = float(resp.headers.get("Retry-After", 0))

            # WQB/gateway may intermittently return upstream errors while job is still queued.
            if resp.status_code in {429, 500, 502, 503, 504}:
                transient_failures += 1
                wait_s = max(retry_after, _SIMULATION_POLL_FALLBACK_SLEEP_SECONDS)
                logger.warning(
                    f"[WQBClient] Poll transient status count={transient_failures} "
                    f"status={resp.status_code} wait={wait_s}s"
                )
                await asyncio.sleep(wait_s)
                continue

            # For other non-200 responses, stop polling this simulation.
            if resp.status_code != 200:
                logger.warning(
                    f"[WQBClient] Poll non-retriable status={resp.status_code} "
                    f"body={resp.text[:240]!r}"
                )
                return None

            if retry_after == 0:
                try:
                    body = resp.json()
                except ValueError:
                    logger.warning("[WQBClient] Poll response is not valid JSON.")
                    await asyncio.sleep(_SIMULATION_POLL_FALLBACK_SLEEP_SECONDS)
                    continue
                return body.get("alpha")
            await asyncio.sleep(max(retry_after, _SIMULATION_POLL_FALLBACK_SLEEP_SECONDS))

    @staticmethod
    def _parse_result(
        alpha_id: str,
        expression: str,
        dataset: str,
        payload: dict[str, Any],
        alpha_data: dict[str, Any],
    ) -> AlphaResult:
        is_data = alpha_data.get("is", {})
        metrics = AlphaMetrics(
            sharpe=float(is_data.get("sharpe", 0)),
            fitness=float(is_data.get("fitness", 0)),
            turnover=float(is_data.get("turnover", 0)),
            margin=float(is_data.get("margin", 0)),
            checks=is_data.get("checks", []),
        )

        failure_reasons: list[str] = []
        qualified = True

        if metrics.sharpe < settings.qual_sharpe_min:
            qualified = False
            failure_reasons.append(f"sharpe={metrics.sharpe:.3f} < {settings.qual_sharpe_min}")
        if metrics.fitness < settings.qual_fitness_min:
            qualified = False
            failure_reasons.append(f"fitness={metrics.fitness:.3f} < {settings.qual_fitness_min}")
        if not (settings.qual_turnover_min <= metrics.turnover <= settings.qual_turnover_max):
            qualified = False
            failure_reasons.append(
                f"turnover={metrics.turnover:.3f} not in "
                f"[{settings.qual_turnover_min}, {settings.qual_turnover_max}]"
            )
        if metrics.ic_mean < settings.qual_ic_mean_min:
            qualified = False
            failure_reasons.append(f"ic_mean={metrics.ic_mean:.3f} < {settings.qual_ic_mean_min}")

        for check in metrics.checks:
            if check.get("result") == "FAIL":
                qualified = False
                failure_reasons.append(
                    f"check_fail:{check.get('name')} value={check.get('value')} limit={check.get('limit')}"
                )

        return AlphaResult(
            alpha_id=alpha_id,
            expression=expression,
            dataset=dataset,
            settings_json=payload.get("settings", {}),
            metrics=metrics,
            qualified=qualified,
            failure_reasons=failure_reasons,
        )

    # ── submission ────────────────────────────────────────────────────────────

    async def submit_alpha(self, alpha_id: str) -> bool:
        """Submit an alpha and poll until completion. Returns True on success."""
        submit_url = f"/alphas/{alpha_id}/submit"
        for attempt in range(5):
            resp = await self._request("POST", submit_url)
            if resp.status_code == 201:
                break
            if resp.status_code in (400, 403):
                return False
            await asyncio.sleep(3 * (attempt + 1))
        else:
            return False

        for _ in range(60):
            resp = await self._request("GET", submit_url)
            retry_after = float(resp.headers.get("Retry-After", 0))
            if retry_after == 0:
                return resp.status_code == 200
            await asyncio.sleep(retry_after)
        return False
