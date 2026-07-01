from __future__ import annotations

from src.data.providers.api_football.models import RawResponseWrapper
from src.data.providers.api_football.raw_store import APIFootballRawStore


def test_wrapper_format(tmp_path):
    store = APIFootballRawStore(tmp_path)
    wrapper = RawResponseWrapper.failure_wrapper(
        endpoint="/fixtures/players",
        params={"fixture": 123456},
        cache_key="abc",
        error="endpoint returned no data",
    )
    path = store.save_wrapper("players", "123456", wrapper)
    assert path.exists()
    data = store.load_endpoint("players", "123456")
    assert data["success"] is False
    assert data["provider"] == "api_football"
    assert data["error"] == "endpoint returned no data"
    assert data["api_response"] is None


def test_success_wrapper(tmp_path):
    store = APIFootballRawStore(tmp_path)
    path = store.save_endpoint_response(
        "events",
        "99",
        endpoint="/fixtures/events",
        params={"fixture": 99},
        api_response=[{"type": "Goal"}],
        success=True,
    )
    assert path.exists()
    loaded = store.load_endpoint("events", "99")
    assert loaded["success"] is True
    assert loaded["response_count"] == 1
