"""Shared test configuration.

Disables retry backoff so the suite does not spend 15 seconds sleeping to
prove that retries wait. The backoff logic itself is unchanged.

Also replaces the Anthropic client with a fake for every test, so the suite
never touches the network, never needs a key, and never spends money.
"""

import json
import os
import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

os.environ["BACKOFF_DISABLED"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import api as api_module  # noqa: E402
from src import db as db_module  # noqa: E402
from src import llm  # noqa: E402
from src.models import Base  # noqa: E402

FAKE_REQUIREMENTS = {
    "title": "Backend Engineer",
    "company": "Example Corp",
    "required_skills": ["Python", "AWS"],
    "preferred_skills": ["Kubernetes"],
    "years_experience": 3,
    "location": "Remote",
    "salary_range": None,
}


class FakeMessages:
    """Stands in for client.beta.messages.

    Set `raises` to an exception (or a list, consumed one per call) to
    simulate API failures. Set `reply` to override what a call returns.
    """

    def __init__(self):
        self.calls = []
        self.raises = []
        self.reply = None

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.raises:
            raise self.raises.pop(0)
        return self.reply or make_response(json.dumps(FAKE_REQUIREMENTS))


def make_response(text, stop_reason="end_turn", model=None, usage=(1200, 150)):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        stop_details=None,
        model=model or llm.MODEL,
        usage=SimpleNamespace(input_tokens=usage[0], output_tokens=usage[1]),
    )


@pytest.fixture(autouse=True)
def fake_anthropic(monkeypatch):
    messages = FakeMessages()
    client = SimpleNamespace(beta=SimpleNamespace(messages=messages))
    monkeypatch.setattr(llm, "get_client", lambda: client)
    # Behave as if a key is configured, regardless of the real environment.
    monkeypatch.setattr(llm, "is_offline", lambda: False)
    return messages


@pytest.fixture
def client(monkeypatch):
    """App wired to a throwaway in-memory database.

    StaticPool keeps the single connection alive so the same :memory:
    database is visible to the lifespan handler and the request handlers.
    """
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestSession = sessionmaker(bind=engine, autoflush=False)
    Base.metadata.create_all(engine)

    monkeypatch.setattr(db_module, "engine", engine)
    monkeypatch.setattr(db_module, "SessionLocal", TestSession)
    monkeypatch.setattr(api_module, "SessionLocal", TestSession)
    monkeypatch.setattr(db_module, "init_db", lambda: Base.metadata.create_all(engine))
    monkeypatch.setattr(api_module, "init_db", lambda: Base.metadata.create_all(engine))

    def override_session():
        s = TestSession()
        try:
            yield s
        finally:
            s.close()

    api_module.app.dependency_overrides[db_module.get_session] = override_session

    with TestClient(api_module.app) as c:
        c.session_factory = TestSession
        yield c

    api_module.app.dependency_overrides.clear()
