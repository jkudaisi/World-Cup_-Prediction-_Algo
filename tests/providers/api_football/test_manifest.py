from __future__ import annotations

import json

from src.data.providers.api_football.manifest import (
    build_backfill_manifest,
    load_resume_state,
    save_resume_state,
)


def test_backfill_manifest_fields():
    m = build_backfill_manifest(teams_requested=48, teams_resolved=46)
    assert m["provider"] == "api_football"
    assert m["teams_requested"] == 48
    assert m["status"] == "running"


def test_resume_state_roundtrip(tmp_path, monkeypatch):
    import src.data.providers.api_football.manifest as man

    monkeypatch.setattr(man, "MANIFEST_ROOT", tmp_path)
    state = {"completed_fixture_ids": [1, 2], "pending_fixture_ids": [3]}
    man.save_resume_state(state)
    loaded = man.load_resume_state()
    assert loaded["completed_fixture_ids"] == [1, 2]
    assert loaded["pending_fixture_ids"] == [3]
