"""TTL-aware response cache for API-Football."""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.data.providers.api_football.paths import CACHE_ROOT

log = logging.getLogger(__name__)

# TTL seconds by data class
TTL_PERMANENT = None
TTL_PAST_FIXTURE_LIST = 30 * 86400
TTL_FUTURE_FIXTURE_LIST = 3600
TTL_LIVE_FIXTURE = 30
TTL_METADATA = 30 * 86400
TTL_COVERAGE = 30 * 86400


class APIFootballCache:
  def __init__(self, cache_dir: Path | str | None = None):
    self.cache_dir = Path(cache_dir or CACHE_ROOT)
    self.cache_dir.mkdir(parents=True, exist_ok=True)
    self.hits = 0
    self.misses = 0

  @staticmethod
  def make_cache_key(endpoint: str, params: dict[str, Any]) -> str:
    payload = json.dumps({"endpoint": endpoint, "params": params}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

  def _path_for_key(self, cache_key: str) -> Path:
    return self.cache_dir / f"{cache_key}.json"

  def get(
    self,
    endpoint: str,
    params: dict[str, Any],
    *,
    ttl_seconds: int | None = TTL_METADATA,
    force_refresh: bool = False,
  ) -> dict[str, Any] | None:
    key = self.make_cache_key(endpoint, params)
    path = self._path_for_key(key)
    if force_refresh or not path.exists():
      self.misses += 1
      return None
    try:
      envelope = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
      self.misses += 1
      return None
    if not envelope.get("valid"):
      self.misses += 1
      return None
    expires_at = envelope.get("expires_at")
    if ttl_seconds is not None and expires_at:
      exp = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
      if datetime.now(timezone.utc) > exp:
        self.misses += 1
        return None
    self.hits += 1
    log.debug("cache hit endpoint=%s key=%s", endpoint, key[:12])
    return envelope

  def put(
    self,
    endpoint: str,
    params: dict[str, Any],
    response: Any,
    *,
    ttl_seconds: int | None = TTL_METADATA,
    valid: bool = True,
  ) -> str:
    key = self.make_cache_key(endpoint, params)
    now = datetime.now(timezone.utc)
    expires_at = None
    if ttl_seconds is not None:
      expires_at = (now + timedelta(seconds=ttl_seconds)).isoformat()
    envelope = {
      "provider": "api_football",
      "endpoint": endpoint,
      "params": params,
      "cache_key": key,
      "created_at": now.isoformat(),
      "expires_at": expires_at,
      "valid": valid,
      "response": response,
    }
    path = self._path_for_key(key)
    path.write_text(json.dumps(envelope, indent=2, default=str), encoding="utf-8")
    log.debug("cache put endpoint=%s key=%s valid=%s", endpoint, key[:12], valid)
    return key

  def stats(self) -> dict[str, int]:
    return {"hits": self.hits, "misses": self.misses}
