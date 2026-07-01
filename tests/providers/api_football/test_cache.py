from __future__ import annotations

from src.data.providers.api_football.cache import APIFootballCache


def test_cache_key_order_independent():
    a = APIFootballCache.make_cache_key("/teams", {"search": "Brazil", "country": "Brazil"})
    b = APIFootballCache.make_cache_key("/teams", {"country": "Brazil", "search": "Brazil"})
    assert a == b
