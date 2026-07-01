"""API-Sports API-Football v3 HTTP client with cache and retries."""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.data.providers.api_football.cache import (
    APIFootballCache,
    TTL_COVERAGE,
    TTL_LIVE_FIXTURE,
    TTL_METADATA,
    TTL_PAST_FIXTURE_LIST,
    TTL_PERMANENT,
)
from src.data.providers.api_football.paths import CACHE_ROOT

log = logging.getLogger(__name__)

AUTH_ERROR_CODES = frozenset({401, 403})


class APIFootballClientError(Exception):
    def __init__(self, status: int, message: str, endpoint: str = ""):
        self.status = status
        self.endpoint = endpoint
        super().__init__(f"APIFootball {status} [{endpoint}]: {message}")


class APIFootballClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://v3.football.api-sports.io",
        cache_dir: str | Path | None = None,
        rate_limit_sleep_seconds: float = 0.25,
        max_retries: int = 3,
        timeout_seconds: int = 30,
        force_refresh: bool = False,
    ):
        self.api_key = api_key or self._resolve_api_key()
        self.base_url = base_url.rstrip("/")
        self.rate_limit_sleep_seconds = rate_limit_sleep_seconds
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.force_refresh = force_refresh
        self.cache = APIFootballCache(cache_dir or CACHE_ROOT)
        self._session = self._build_session()
        self.endpoint_request_counts: dict[str, int] = {}
        self.failed_requests: list[dict[str, Any]] = []

    @staticmethod
    def _resolve_api_key() -> str:
        try:
            from dotenv import load_dotenv
            load_dotenv(override=True)
        except ImportError:
            pass
        for name in ("APIFOOTBALL_KEY", "API_FOOTBALL_KEY", "API_SPORTS_KEY"):
            val = os.environ.get(name, "").strip()
            if val:
                return val
        try:
            from config import APIFOOTBALL_KEY
            return (APIFOOTBALL_KEY or "").strip()
        except ImportError:
            return ""

    def _build_session(self) -> requests.Session:
        retry = Retry(
            total=self.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        session = requests.Session()
        session.mount("https://", HTTPAdapter(max_retries=retry))
        return session

    def _headers(self) -> dict[str, str]:
        return {"x-apisports-key": self.api_key}

    def _bump_endpoint(self, endpoint: str) -> None:
        self.endpoint_request_counts[endpoint] = self.endpoint_request_counts.get(endpoint, 0) + 1

    def _ttl_for_endpoint(self, endpoint: str, params: dict[str, Any]) -> int | None:
        if endpoint in ("/fixtures/events", "/fixtures/statistics", "/fixtures/players",
                        "/fixtures/lineups", "/injuries"):
            return TTL_PERMANENT
        if endpoint == "/fixtures":
            if params.get("live"):
                return TTL_LIVE_FIXTURE
            if params.get("date") or params.get("from"):
                return TTL_PAST_FIXTURE_LIST
            return TTL_PAST_FIXTURE_LIST
        if endpoint in ("/leagues", "/teams", "/countries", "/status"):
            return TTL_METADATA
        if endpoint.startswith("/coverage") or endpoint == "/coachs":
            return TTL_COVERAGE
        return TTL_METADATA

    def _request(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        *,
        ttl_seconds: int | None = None,
    ) -> list[Any] | dict[str, Any]:
        if not self.api_key:
            raise APIFootballClientError(401, "API key not configured", endpoint)

        params = {k: v for k, v in (params or {}).items() if v is not None}
        ttl = ttl_seconds if ttl_seconds is not None else self._ttl_for_endpoint(endpoint, params)

        if not self.force_refresh:
            cached = self.cache.get(endpoint, params, ttl_seconds=ttl, force_refresh=False)
            if cached is not None:
                return cached["response"]

        url = f"{self.base_url}{endpoint}"
        last_exc: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                log.info("API request endpoint=%s params=%s attempt=%s", endpoint, params, attempt + 1)
                self._bump_endpoint(endpoint)
                resp = self._session.get(
                    url,
                    params=params,
                    headers=self._headers(),
                    timeout=self.timeout_seconds,
                )
                if resp.status_code in AUTH_ERROR_CODES:
                    raise APIFootballClientError(resp.status_code, "authentication failed", endpoint)
                if resp.status_code == 429:
                    wait = self.rate_limit_sleep_seconds * (2 ** attempt) + 1
                    log.warning("rate limited endpoint=%s sleeping %.2fs", endpoint, wait)
                    time.sleep(wait)
                    continue
                if not resp.ok:
                    raise APIFootballClientError(resp.status_code, resp.text[:300], endpoint)

                payload = resp.json()
                if not isinstance(payload, dict):
                    raise APIFootballClientError(0, "malformed JSON response", endpoint)

                errors = payload.get("errors") or {}
                if isinstance(errors, dict) and errors:
                    raise APIFootballClientError(400, str(errors), endpoint)

                response = payload.get("response")
                if response is None:
                    raise APIFootballClientError(0, "missing response field", endpoint)

                self.cache.put(endpoint, params, response, ttl_seconds=ttl, valid=True)
                if self.rate_limit_sleep_seconds > 0:
                    time.sleep(self.rate_limit_sleep_seconds)
                return response

            except APIFootballClientError:
                raise
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_exc = exc
                wait = self.rate_limit_sleep_seconds * (2 ** attempt) + 0.5
                log.warning("network error endpoint=%s: %s; retry in %.2fs", endpoint, exc, wait)
                time.sleep(wait)

        self.failed_requests.append({"endpoint": endpoint, "params": params, "error": str(last_exc)})
        raise APIFootballClientError(0, f"request failed after retries: {last_exc}", endpoint)

    # ── Endpoint methods ──────────────────────────────────────────────

    def get_status(self) -> Any:
        return self._request("/status", {})

    def get_countries(self, **params: Any) -> Any:
        return self._request("/countries", params)

    def get_leagues(self, **params: Any) -> Any:
        return self._request("/leagues", params)

    def get_teams(self, **params: Any) -> Any:
        return self._request("/teams", params)

    def get_fixtures(self, **params: Any) -> Any:
        return self._request("/fixtures", params)

    def get_fixture_events(self, fixture_id: int) -> Any:
        return self._request("/fixtures/events", {"fixture": fixture_id})

    def get_fixture_statistics(self, fixture_id: int) -> Any:
        return self._request("/fixtures/statistics", {"fixture": fixture_id})

    def get_fixture_players(self, fixture_id: int) -> Any:
        return self._request("/fixtures/players", {"fixture": fixture_id})

    def get_fixture_lineups(self, fixture_id: int) -> Any:
        return self._request("/fixtures/lineups", {"fixture": fixture_id})

    def get_injuries(self, **params: Any) -> Any:
        return self._request("/injuries", params)

    def get_standings(self, **params: Any) -> Any:
        return self._request("/standings", params)

    def get_team_statistics(self, **params: Any) -> Any:
        return self._request("/teams/statistics", params)

    def get_coachs(self, **params: Any) -> Any:
        return self._request("/coachs", params)

    def get_players(self, **params: Any) -> Any:
        return self._request("/players", params)

    def get_player_squads(self, **params: Any) -> Any:
        return self._request("/players/squads", params)

    def cache_stats(self) -> dict[str, int]:
        return self.cache.stats()

    def endpoint_stats(self) -> dict[str, int]:
        return dict(self.endpoint_request_counts)
