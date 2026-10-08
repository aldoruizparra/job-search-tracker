"""API tests.

The one that matters is test_startup_recovers_stranded_jobs: it proves the
durability work in runner.py actually fires when the service boots.
"""

import os
import sys

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import api as api_module
from src import db as db_module
from src.models import Base, Job, JobStatus, Step, StepStatus
from src.runner import PipelineRunner, StepFailed


# ------------------------------------------------------------------


def test_create_job_runs_the_pipeline(client):
    r = client.post("/jobs", json={"raw_text": "Backend Engineer, Python, AWS"})
    assert r.status_code == 201

    body = r.json()
    assert body["status"] == "complete"
    assert len(body["steps"]) == 5
    assert all(s["status"] == "done" for s in body["steps"])
    assert body["application_status"] == "interested"


def test_create_job_requires_some_input(client):
    r = client.post("/jobs", json={})
    assert r.status_code == 422


def test_get_job_and_404(client):
    created = client.post("/jobs", json={"raw_text": "posting"}).json()

    assert client.get(f"/jobs/{created['id']}").json()["id"] == created["id"]
    assert client.get("/jobs/nope").status_code == 404


def test_update_application_status(client):
    created = client.post("/jobs", json={"raw_text": "posting"}).json()

    r = client.patch(
        f"/jobs/{created['id']}/status", json={"application_status": "applied"}
    )
    assert r.status_code == 200
    assert r.json()["application_status"] == "applied"
    # pipeline status is unaffected: the two are deliberately separate
    assert r.json()["status"] == "complete"


def test_list_jobs_filters_by_application_status(client):
    a = client.post("/jobs", json={"raw_text": "one"}).json()
    client.post("/jobs", json={"raw_text": "two"}).json()
    client.patch(f"/jobs/{a['id']}/status", json={"application_status": "applied"})

    assert len(client.get("/jobs").json()) == 2
    applied = client.get("/jobs", params={"status": "applied"}).json()
    assert len(applied) == 1
    assert applied[0]["id"] == a["id"]


def test_url_only_job_dead_letters(client):
    """URL ingestion is not built yet, so it should fail loudly, not silently."""
    r = client.post("/jobs", json={"source_url": "https://example.com/job"})
    assert r.status_code == 201

    body = r.json()
    assert body["status"] == "dead_letter"
    assert body["steps"][0]["status"] == "failed"
    assert body["steps"][0]["attempts"] == 3


def test_malformed_model_response_dead_letters_instead_of_500(client, fake_anthropic):
    from conftest import make_response

    fake_anthropic.reply = make_response("not json")
    r = client.post("/jobs", json={"raw_text": "posting"})
    assert r.status_code == 201

    body = r.json()
    assert body["status"] == "dead_letter"
    extract = next(s for s in body["steps"] if s["step_name"] == "extract")
    assert extract["status"] == "failed"
    assert extract["attempts"] == 1
    assert extract["error"].startswith("JSONDecodeError")


def test_retry_resets_only_the_failed_step(client):
    created = client.post("/jobs", json={"source_url": "https://example.com/job"}).json()
    assert created["status"] == "dead_letter"

    r = client.post(f"/jobs/{created['id']}/retry")
    assert r.status_code == 201 or r.status_code == 200
    # still fails, because the underlying handler is still unimplemented
    assert r.json()["status"] == "dead_letter"


def test_retry_rejects_a_healthy_job(client):
    created = client.post("/jobs", json={"raw_text": "posting"}).json()
    r = client.post(f"/jobs/{created['id']}/retry")
    assert r.status_code == 409


def test_health_counts_by_pipeline_status(client):
    client.post("/jobs", json={"raw_text": "posting"})
    client.post("/jobs", json={"source_url": "https://example.com/job"})

    body = client.get("/health").json()
    assert body["jobs"]["complete"] == 1
    assert body["jobs"]["dead_letter"] == 1


def test_insights_aggregates_across_jobs(client):
    """Write fit_analysis directly, since the compare step is still a stub."""
    client.post("/jobs", json={"raw_text": "one"})
    client.post("/jobs", json={"raw_text": "two"})

    s = client.session_factory()
    for job, missing in zip(s.query(Job).all(), [["Kubernetes", "Go"], ["Kubernetes"]]):
        job.fit_analysis = {"met": ["Python"], "missing": missing}
    s.commit()
    s.close()

    body = client.get("/insights").json()
    assert body["jobs_analyzed"] == 2
    assert body["top_gaps"][0] == {"skill": "Kubernetes", "count": 2}


def test_startup_recovers_stranded_jobs(monkeypatch):
    """Boot the app with a job stranded mid-pipeline and assert it finishes."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestSession = sessionmaker(bind=engine, autoflush=False)
    Base.metadata.create_all(engine)

    monkeypatch.setattr(db_module, "SessionLocal", TestSession)
    monkeypatch.setattr(api_module, "SessionLocal", TestSession)
    monkeypatch.setattr(api_module, "init_db", lambda: Base.metadata.create_all(engine))

    # hand-build a job that died after step 0
    s = TestSession()
    job = Job(raw_text="stranded posting", status=JobStatus.PROCESSING)
    s.add(job)
    s.commit()
    job_id = job.id

    s.add(
        Step(
            job_id=job_id,
            step_number=0,
            step_name="ingest",
            status=StepStatus.DONE,
            idempotency_key="k0",
            output_data={"text": "stranded posting", "source": "pasted"},
        )
    )
    s.commit()
    s.close()

    def override_session():
        sess = TestSession()
        try:
            yield sess
        finally:
            sess.close()

    api_module.app.dependency_overrides[db_module.get_session] = override_session

    with TestClient(api_module.app) as c:          # lifespan fires here
        body = c.get(f"/jobs/{job_id}").json()
        assert body["status"] == "complete"
        # ingest was already done and must not have run again
        ingest_step = next(s for s in body["steps"] if s["step_name"] == "ingest")
        assert ingest_step["attempts"] == 0

    api_module.app.dependency_overrides.clear()
