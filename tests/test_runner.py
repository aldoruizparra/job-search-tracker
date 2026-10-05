"""Tests for the pipeline runner.

The important one is test_crash_resumes_without_rerunning_completed_steps.
That is the claim the whole project rests on, so it gets an explicit test.
"""

import sys
import os
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models import Base, Job, JobStatus, Step, StepStatus
from src.runner import PipelineRunner, StepFailed, make_idempotency_key


@pytest.fixture
def session_factory():
    """Fresh in-memory database per test.

    StaticPool keeps one connection alive so :memory: survives across
    sessions, which we need to simulate a process restart.
    """
    from sqlalchemy.pool import StaticPool

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


@pytest.fixture
def session(session_factory):
    s = session_factory()
    yield s
    s.close()


def make_step(name, calls, output=None, fail_times=0, crash=False, cost=0.0):
    """Build a handler that records calls and can fail or crash on demand."""
    state = {"failures": 0}

    def handler(job, prior):
        calls.append(name)
        if crash:
            raise SystemExit(f"simulated crash in {name}")
        if state["failures"] < fail_times:
            state["failures"] += 1
            raise StepFailed(f"{name} transient failure")
        return {
            "output": output or {"step": name},
            "input_tokens": 100,
            "output_tokens": 50,
            "cost_usd": cost,
        }

    return handler


# --------------------------------------------------------------------------


def test_happy_path_runs_every_step_once(session):
    calls = []
    steps = [
        ("ingest", make_step("ingest", calls)),
        ("extract", make_step("extract", calls)),
        ("compare", make_step("compare", calls)),
    ]

    job = Job(raw_text="posting")
    session.add(job)
    session.commit()

    result = PipelineRunner(session, steps).run(job.id)

    assert result.status == JobStatus.COMPLETE
    assert calls == ["ingest", "extract", "compare"]
    assert all(s.status == StepStatus.DONE for s in result.steps)


def test_crash_resumes_without_rerunning_completed_steps(session_factory):
    """The core guarantee: a crash is a pause, not a restart."""
    calls = []

    crashing = [
        ("ingest", make_step("ingest", calls)),
        ("extract", make_step("extract", calls, crash=True)),
        ("compare", make_step("compare", calls)),
    ]

    s1 = session_factory()
    job = Job(raw_text="posting")
    s1.add(job)
    s1.commit()
    job_id = job.id

    with pytest.raises(SystemExit):
        PipelineRunner(s1, crashing).run(job_id)
    s1.close()

    assert calls == ["ingest", "extract"]

    # fresh session == new process
    recovered = [
        ("ingest", make_step("ingest", calls)),
        ("extract", make_step("extract", calls)),
        ("compare", make_step("compare", calls)),
    ]
    s2 = session_factory()
    resumed = PipelineRunner(s2, recovered).resume_all()

    assert len(resumed) == 1
    assert resumed[0].status == JobStatus.COMPLETE

    # ingest must NOT appear a second time
    assert calls.count("ingest") == 1
    assert calls == ["ingest", "extract", "extract", "compare"]
    s2.close()


def test_transient_failure_is_retried(session):
    calls = []
    steps = [("extract", make_step("extract", calls, fail_times=2))]

    job = Job(raw_text="posting")
    session.add(job)
    session.commit()

    runner = PipelineRunner(session, steps, max_attempts=3)
    runner._backoff = staticmethod(lambda attempt: 0)  # no sleeping in tests

    result = runner.run(job.id)

    assert result.status == JobStatus.COMPLETE
    assert len(calls) == 3           # two failures, then success
    assert result.steps[0].attempts == 3


def test_exhausted_retries_goes_to_dead_letter(session):
    calls = []
    steps = [("extract", make_step("extract", calls, fail_times=99))]

    job = Job(raw_text="posting")
    session.add(job)
    session.commit()

    runner = PipelineRunner(session, steps, max_attempts=3)
    runner._backoff = staticmethod(lambda attempt: 0)

    result = runner.run(job.id)

    assert result.status == JobStatus.DEAD_LETTER
    assert result.steps[0].status == StepStatus.FAILED
    assert len(calls) == 3
    assert "transient failure" in result.steps[0].error


def test_dead_letter_job_is_not_silently_retried(session):
    """A failed job stays failed until a human looks at it."""
    calls = []
    steps = [("extract", make_step("extract", calls, fail_times=99))]

    job = Job(raw_text="posting")
    session.add(job)
    session.commit()

    runner = PipelineRunner(session, steps, max_attempts=2)
    runner._backoff = staticmethod(lambda attempt: 0)
    runner.run(job.id)

    before = len(calls)
    result = runner.run(job.id)       # run again

    assert result.status == JobStatus.DEAD_LETTER
    assert len(calls) == before        # handler was not called again


def test_cost_rolls_up_to_the_job(session):
    calls = []
    steps = [
        ("extract", make_step("extract", calls, cost=0.009)),
        ("compare", make_step("compare", calls, cost=0.006)),
    ]

    job = Job(raw_text="posting")
    session.add(job)
    session.commit()

    result = PipelineRunner(session, steps).run(job.id)

    assert result.total_cost_usd == pytest.approx(0.015)
    assert result.total_input_tokens == 200
    assert result.total_output_tokens == 100


def test_completed_job_is_not_rerun(session):
    calls = []
    steps = [("ingest", make_step("ingest", calls))]

    job = Job(raw_text="posting")
    session.add(job)
    session.commit()

    runner = PipelineRunner(session, steps)
    runner.run(job.id)
    runner.run(job.id)

    assert len(calls) == 1


def test_step_output_feeds_the_next_step(session):
    seen = {}

    def first(job, prior):
        return {"output": {"text": "hello"}}

    def second(job, prior):
        seen["prior"] = prior
        return {"output": {"ok": True}}

    job = Job(raw_text="posting")
    session.add(job)
    session.commit()

    PipelineRunner(session, [("a", first), ("b", second)]).run(job.id)

    assert seen["prior"] == {"text": "hello"}


def test_idempotency_key_is_stable_and_input_sensitive():
    a = make_idempotency_key("job1", "extract", {"prior": {"x": 1}})
    b = make_idempotency_key("job1", "extract", {"prior": {"x": 1}})
    c = make_idempotency_key("job1", "extract", {"prior": {"x": 2}})
    d = make_idempotency_key("job2", "extract", {"prior": {"x": 1}})

    assert a == b          # same inputs, same key
    assert a != c          # different input, different key
    assert a != d          # different job, different key


def test_resume_all_handles_multiple_stranded_jobs(session_factory):
    calls = []
    crashing = [("extract", make_step("extract", calls, crash=True))]

    s1 = session_factory()
    ids = []
    for _ in range(3):
        j = Job(raw_text="posting")
        s1.add(j)
        s1.commit()
        ids.append(j.id)
        with pytest.raises(SystemExit):
            PipelineRunner(s1, crashing).run(j.id)
    s1.close()

    s2 = session_factory()
    good = [("extract", make_step("extract", calls))]
    resumed = PipelineRunner(s2, good).resume_all()

    assert len(resumed) == 3
    assert all(j.status == JobStatus.COMPLETE for j in resumed)
    s2.close()
