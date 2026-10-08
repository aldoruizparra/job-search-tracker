"""Tests for the extract step's Anthropic call.

The point here is the failure mapping. Transient API conditions must become
StepFailed so the runner retries them; everything else must propagate, because
retrying a bad request or a malformed response just pays for the same bug
three times.
"""

import json
from types import SimpleNamespace

import anthropic
import httpx2
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from conftest import FAKE_REQUIREMENTS, make_response
from src import pipeline
from src.models import Base, Job, JobStatus
from src.pipeline import STEPS, ModelRefused, extract
from src.runner import PipelineRunner, StepFailed

REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
JOB = SimpleNamespace(id="job-1")
PRIOR = {"text": "Backend Engineer at Example Corp. Python and AWS required."}


def status_error(cls, code):
    return cls("boom", response=httpx2.Response(code, request=REQUEST), body=None)


@pytest.fixture
def session():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    yield s
    s.close()


# ------------------------------------------------------------- happy path


def test_extract_returns_requirements_tokens_and_cost(fake_anthropic):
    result = extract(JOB, PRIOR)

    assert result["output"] == FAKE_REQUIREMENTS
    assert result["input_tokens"] == 1200
    assert result["output_tokens"] == 150
    # claude-opus-5-5: $4 in, $20 out per million
    assert result["cost_usd"] == pytest.approx((1200 * 4 + 150 * 20) / 1e6)

    call = fake_anthropic.calls[0]
    assert call["messages"][0]["content"] == PRIOR["text"]
    assert call["output_config"]["format"]["schema"] == pipeline.REQUIREMENTS_SCHEMA


def test_cost_is_priced_by_the_model_that_served_the_call(fake_anthropic):
    """A server-side fallback changes response.model; bill at that rate."""
    fake_anthropic.reply = make_response(
        json.dumps(FAKE_REQUIREMENTS), model="claude-opus-4-8"
    )
    result = extract(JOB, PRIOR)
    assert result["cost_usd"] == pytest.approx((1200 * 5 + 150 * 25) / 1e6)


def test_unpriced_model_fails_before_calling_the_api(fake_anthropic, monkeypatch):
    monkeypatch.setattr(pipeline, "EXTRACT_MODEL", "claude-unknown")
    with pytest.raises(ValueError, match="no pricing"):
        extract(JOB, PRIOR)
    assert fake_anthropic.calls == []


# ------------------------------------------------------ retryable failures


@pytest.mark.parametrize(
    "exc",
    [
        status_error(anthropic.RateLimitError, 429),
        status_error(anthropic.InternalServerError, 500),
        status_error(anthropic.InternalServerError, 529),
        anthropic.APITimeoutError(request=REQUEST),
        anthropic.APIConnectionError(request=REQUEST),
    ],
    ids=["rate_limit", "500", "overloaded", "timeout", "connection"],
)
def test_transient_api_errors_become_step_failed(fake_anthropic, exc):
    fake_anthropic.raises = [exc]
    with pytest.raises(StepFailed):
        extract(JOB, PRIOR)


# --------------------------------------------------- non-retryable failures


@pytest.mark.parametrize(
    "exc",
    [
        status_error(anthropic.BadRequestError, 400),
        status_error(anthropic.AuthenticationError, 401),
        status_error(anthropic.NotFoundError, 404),
    ],
    ids=["bad_request", "auth", "not_found"],
)
def test_client_errors_propagate_unretried(fake_anthropic, exc):
    fake_anthropic.raises = [exc]
    with pytest.raises(type(exc)):
        extract(JOB, PRIOR)


def test_malformed_response_is_a_bug_not_a_retry(fake_anthropic):
    fake_anthropic.reply = make_response("not json at all")
    with pytest.raises(json.JSONDecodeError):
        extract(JOB, PRIOR)


def test_truncated_response_is_not_retried(fake_anthropic):
    fake_anthropic.reply = make_response('{"title": "Back', stop_reason="max_tokens")
    with pytest.raises(ValueError, match="truncated"):
        extract(JOB, PRIOR)


def test_refusal_is_not_retried(fake_anthropic):
    fake_anthropic.reply = make_response("", stop_reason="refusal")
    with pytest.raises(ModelRefused):
        extract(JOB, PRIOR)


# ------------------------------------------------------- through the runner


def test_rate_limited_extract_retries_then_completes(fake_anthropic, session):
    """Two 429s, then success: one billed call, three recorded attempts."""
    fake_anthropic.raises = [
        status_error(anthropic.RateLimitError, 429),
        status_error(anthropic.RateLimitError, 429),
    ]
    job = Job(raw_text=PRIOR["text"])
    session.add(job)
    session.commit()

    result = PipelineRunner(session, STEPS).run(job.id)

    assert result.status == JobStatus.COMPLETE
    step = next(s for s in result.steps if s.step_name == "extract")
    assert step.attempts == 3
    assert len(fake_anthropic.calls) == 3
    assert result.total_input_tokens == 1200
    assert result.total_cost_usd == pytest.approx((1200 * 4 + 150 * 20) / 1e6)
    # store step copies the extracted fields onto the job
    assert result.title == "Backend Engineer"
    assert result.company == "Example Corp"
    assert result.requirements["required_skills"] == ["Python", "AWS"]
