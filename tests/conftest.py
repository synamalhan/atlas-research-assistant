import socket

import pytest


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*a, **k):
        raise RuntimeError("network access is forbidden in tests")

    monkeypatch.setattr(socket.socket, "connect", blocked)


@pytest.fixture(autouse=True)
def llm_env(monkeypatch, tmp_path):
    monkeypatch.setenv("ATLAS_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("ATLAS_LLM_BACKEND", "fake")
    monkeypatch.delenv("ATLAS_LLM_MODEL", raising=False)
    monkeypatch.setattr("atlas.llm.BACKOFF", 0)
