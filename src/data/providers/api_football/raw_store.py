"""Raw data lake persistence for API-Football."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.data.providers.api_football.cache import APIFootballCache
from src.data.providers.api_football.models import RawResponseWrapper
from src.data.providers.api_football.paths import RAW_ROOT
from training_store import atomic_write_json

log = logging.getLogger(__name__)


class APIFootballRawStore:
    def __init__(self, root: Path | None = None):
        self.root = root or RAW_ROOT
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        for sub in (
            "fixtures/by_team",
            "fixtures/by_fixture",
            "events",
            "statistics",
            "players",
            "lineups",
            "injuries",
            "standings",
            "teams",
            "leagues",
            "coverage",
        ):
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    def save_wrapper(self, kind: str, key: str, wrapper: RawResponseWrapper | dict[str, Any]) -> Path:
        payload = wrapper.to_dict() if isinstance(wrapper, RawResponseWrapper) else wrapper
        if kind == "fixtures_by_team":
            path = self.root / "fixtures" / "by_team" / f"team_{key}.json"
        elif kind == "fixtures_by_fixture":
            path = self.root / "fixtures" / "by_fixture" / f"fixture_{key}.json"
        elif kind in ("events", "statistics", "players", "lineups", "injuries"):
            path = self.root / kind / f"fixture_{key}.json"
        elif kind == "standings":
            path = self.root / "standings" / f"{key}.json"
        elif kind == "teams":
            path = self.root / "teams" / f"team_{key}.json"
        elif kind == "leagues":
            path = self.root / "leagues" / f"{key}.json"
        elif kind == "coverage":
            path = self.root / "coverage" / key
        else:
            raise ValueError(f"unknown raw kind: {kind}")

        atomic_write_json(path, payload)
        return path

    def save_endpoint_response(
        self,
        kind: str,
        key: str,
        *,
        endpoint: str,
        params: dict[str, Any],
        api_response: Any,
        success: bool = True,
        error: str | None = None,
    ) -> Path:
        cache_key = APIFootballCache.make_cache_key(endpoint, params)
        if success:
            wrapper = RawResponseWrapper.success_wrapper(
                endpoint=endpoint,
                params=params,
                cache_key=cache_key,
                api_response=api_response,
            )
        else:
            wrapper = RawResponseWrapper.failure_wrapper(
                endpoint=endpoint,
                params=params,
                cache_key=cache_key,
                error=error or "unknown error",
            )
        return self.save_wrapper(kind, key, wrapper)

    def load_endpoint(self, kind: str, key: str) -> dict[str, Any] | None:
        if kind == "fixtures_by_fixture":
            path = self.root / "fixtures" / "by_fixture" / f"fixture_{key}.json"
        elif kind in ("events", "statistics", "players", "lineups", "injuries"):
            path = self.root / kind / f"fixture_{key}.json"
        else:
            return None
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def append_missing_endpoint_log(self, entry: dict[str, Any]) -> Path:
        path = self.root / "coverage" / "missing_endpoint_log.json"
        existing: list[dict[str, Any]] = []
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                existing = data if isinstance(data, list) else data.get("entries", [])
            except (json.JSONDecodeError, OSError):
                existing = []
        existing.append(entry)
        atomic_write_json(path, {"provider": "api_football", "entries": existing})
        return path

    def save_coverage_report(self, name: str, report: dict[str, Any]) -> Path:
        return self.save_wrapper("coverage", name, {
            "provider": "api_football",
            **report,
        })
