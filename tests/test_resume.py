"""Resume parsing tests.

The resume below is fabricated. Never put a real one in a fixture.

What matters: a resume is parsed once per distinct text (an identical
re-upload costs nothing), an offline placeholder never masks a later real
parse, and contact details have nowhere to land.
"""

import json

import anthropic
import httpx2
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from conftest import make_response
from src import llm
from src import resume as resume_module
from src.models import Base, Resume
from src.runner import StepFailed

FAKE_RESUME = """\
Jordan Avery | jordan.avery@example.com | (555) 010-0199

Backend Engineer, Brightline Logistics, 2021 - present
- Built order-routing services in Python and FastAPI on AWS
- Ran PostgreSQL migrations and Redis caching for 2M requests/day

Software Engineer, Harbor Analytics, 2018 - 2021
- Wrote ETL jobs in Python, deployed with Docker and Terraform

B.S. Computer Science, State University
"""

FAKE_PROFILE = {
    "skills": ["Python", "FastAPI", "AWS", "PostgreSQL", "Redis", "Docker", "Terraform"],
    "years_experience": 7,
    "titles": ["Backend Engineer", "Software Engineer"],
    "education": ["B.S. Computer Science"],
}

REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def profile_reply(profile=FAKE_PROFILE, model=None):
    return make_response(json.dumps(profile), model=model, usage=(900, 120))


@pytest.fixture
def session():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    yield s
    s.close()


# ------------------------------------------------------------- parsing


def test_upload_parses_and_stores_the_profile(session, fake_anthropic):
    fake_anthropic.reply = profile_reply()

    resume, parsed = resume_module.upload(session, FAKE_RESUME)

    assert parsed is True
    assert resume.profile == FAKE_PROFILE
    assert resume.model == llm.MODEL
    assert resume.cost_usd == pytest.approx((900 * 4 + 120 * 20) / 1e6)
    assert resume_module.current(session).id == resume.id

    call = fake_anthropic.calls[0]
    assert call["system"] == resume_module.RESUME_SYSTEM
    assert call["output_config"]["format"]["schema"] == resume_module.PROFILE_SCHEMA


def test_identical_reupload_is_not_parsed_or_billed_again(session, fake_anthropic):
    fake_anthropic.reply = profile_reply()
    first, _ = resume_module.upload(session, FAKE_RESUME)

    # same content, different whitespace: still the same resume
    again, parsed = resume_module.upload(session, "  " + FAKE_RESUME.replace("\n", "\r\n"))

    assert parsed is False
    assert again.id == first.id
    assert len(fake_anthropic.calls) == 1
    assert session.query(Resume).count() == 1


def test_changed_resume_is_parsed_and_becomes_current(session, fake_anthropic):
    fake_anthropic.reply = profile_reply()
    first, _ = resume_module.upload(session, FAKE_RESUME)

    updated = {**FAKE_PROFILE, "skills": FAKE_PROFILE["skills"] + ["Kubernetes"]}
    fake_anthropic.reply = profile_reply(updated)
    second, parsed = resume_module.upload(session, FAKE_RESUME + "\n- Learned Kubernetes")

    assert parsed is True
    assert second.id != first.id
    assert resume_module.current(session).profile["skills"][-1] == "Kubernetes"
    assert session.query(Resume).count() == 2   # history kept


def test_offline_placeholder_does_not_mask_a_later_real_parse(
    session, fake_anthropic, monkeypatch
):
    """Upload with no key, add a key, re-upload the same text: must parse."""
    monkeypatch.setattr(llm, "is_offline", lambda: True)
    placeholder, _ = resume_module.upload(session, FAKE_RESUME)
    assert placeholder.model == resume_module.OFFLINE
    assert placeholder.profile["skills"] == []
    assert fake_anthropic.calls == []

    monkeypatch.setattr(llm, "is_offline", lambda: False)
    fake_anthropic.reply = profile_reply()
    real, parsed = resume_module.upload(session, FAKE_RESUME)

    assert parsed is True
    assert real.profile == FAKE_PROFILE
    assert len(fake_anthropic.calls) == 1


def test_profile_schema_has_no_place_for_contact_details():
    schema = resume_module.PROFILE_SCHEMA
    assert set(schema["properties"]) == {"skills", "years_experience", "titles", "education"}
    assert schema["additionalProperties"] is False


def test_transient_error_stores_nothing(session, fake_anthropic):
    fake_anthropic.raises = [
        anthropic.RateLimitError(
            "slow down", response=httpx2.Response(429, request=REQUEST), body=None
        )
    ]
    with pytest.raises(StepFailed):
        resume_module.upload(session, FAKE_RESUME)
    assert session.query(Resume).count() == 0


# ---------------------------------------------------------------- HTTP


# client fixture: conftest.py


def test_put_resume_returns_profile_without_raw_text(client, fake_anthropic):
    fake_anthropic.reply = profile_reply()
    r = client.put("/resume", json={"text": FAKE_RESUME})

    assert r.status_code == 200
    body = r.json()
    assert body["profile"] == FAKE_PROFILE
    assert body["parsed"] is True
    assert body["chars"] == len(FAKE_RESUME)
    assert "raw_text" not in body
    assert "jordan.avery@example.com" not in r.text


def test_get_resume_404_until_uploaded(client, fake_anthropic):
    assert client.get("/resume").status_code == 404

    fake_anthropic.reply = profile_reply()
    client.put("/resume", json={"text": FAKE_RESUME})

    r = client.get("/resume")
    assert r.status_code == 200
    assert r.json()["profile"] == FAKE_PROFILE


def test_put_blank_resume_is_rejected(client, fake_anthropic):
    assert client.put("/resume", json={"text": "   \n "}).status_code == 422
    assert fake_anthropic.calls == []


def test_transient_api_error_returns_503(client, fake_anthropic):
    fake_anthropic.raises = [anthropic.APITimeoutError(request=REQUEST)]
    r = client.put("/resume", json={"text": FAKE_RESUME})

    assert r.status_code == 503
    assert client.get("/resume").status_code == 404


def test_refusal_returns_422(client, fake_anthropic):
    fake_anthropic.reply = make_response("", stop_reason="refusal")
    assert client.put("/resume", json={"text": FAKE_RESUME}).status_code == 422


def test_insights_total_includes_resume_spend(client, fake_anthropic):
    fake_anthropic.reply = profile_reply()
    client.put("/resume", json={"text": FAKE_RESUME})

    total = client.get("/insights").json()["total_cost_usd"]
    assert total == pytest.approx(round((900 * 4 + 120 * 20) / 1e6, 4))
