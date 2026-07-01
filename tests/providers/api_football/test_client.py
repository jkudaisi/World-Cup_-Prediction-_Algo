from __future__ import annotations

import logging

import pytest

from src.data.providers.api_football.cache import APIFootballCache


class TestCacheKey:
    def test_stable_cache_key(self):
        k1 = APIFootballCache.make_cache_key("/fixtures", {"team": 26, "season": 2022})
        k2 = APIFootballCache.make_cache_key("/fixtures", {"season": 2022, "team": 26})
        assert k1 == k2
        assert len(k1) == 64


class TestCacheTTL:
    def test_miss_then_hit(self, tmp_path):
        cache = APIFootballCache(tmp_path)
        params = {"fixture": 1}
        assert cache.get("/fixtures/events", params) is None
        cache.put("/fixtures/events", params, [{"type": "Goal"}], ttl_seconds=3600)
        hit = cache.get("/fixtures/events", params, ttl_seconds=3600)
        assert hit is not None
        assert hit["response"][0]["type"] == "Goal"
        assert cache.stats()["hits"] == 1

    def test_invalid_not_returned(self, tmp_path):
        cache = APIFootballCache(tmp_path)
        cache.put("/fixtures", {"id": 1}, [], valid=False)
        assert cache.get("/fixtures", {"id": 1}) is None


class TestNoKeyInLogs:
    def test_api_key_not_logged(self, caplog):
        from src.data.providers.api_football.client import APIFootballClient

        secret = "super-secret-key-12345"
        with caplog.at_level(logging.DEBUG):
            APIFootballClient(api_key=secret)
        for record in caplog.records:
            assert secret not in record.getMessage()
