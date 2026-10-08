"""FastAPI layer.

The important line in this file is the resume_all() call in the lifespan
handler. Without it, a crash leaves jobs stranded in PROCESSING forever and
the durability work in runner.py never actually fires in production.
"""

import logging
from collections import Counter
from contextlib import asynccontextmanager
from typing import Optional

from datetime import datetime

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, field_validator, model_validator
from sqlalchemy.orm import Session

from . import resume as resume_module
from .db import SessionLocal, get_session, init_db
from .llm import ModelRefused
from .models import ApplicationStatus, Job, JobStatus, Resume
from .pipeline import STEPS
from .runner import PipelineRunner, StepFailed

log = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown.

    On startup we create tables and then recover anything stranded by a
    previous crash. This is the whole point of the project.
    """
    init_db()

    session = SessionLocal()
    try:
        recovered = PipelineRunner(session, STEPS).resume_all()
        if recovered:
            log.info("recovered stranded jobs on startup", extra={"count": len(recovered)})
    finally:
        session.close()

    yield


app = FastAPI(title="Job Search Tracker", lifespan=lifespan)


# ------------------------------------------------------------- schemas


class JobCreate(BaseModel):
    source_url: Optional[str] = None
    raw_text: Optional[str] = None

    @model_validator(mode="after")
    def need_one_input(self):
        if not self.source_url and not self.raw_text:
            raise ValueError("provide either source_url or raw_text")
        return self


class StepOut(BaseModel):
    step_number: int
    step_name: str
    status: str
    attempts: int
    duration_ms: Optional[int] = None
    cost_usd: float = 0.0
    error: Optional[str] = None


class JobOut(BaseModel):
    id: str
    title: Optional[str] = None
    company: Optional[str] = None
    status: str
    application_status: str
    fit_score: Optional[float] = None
    requirements: Optional[dict] = None
    fit_analysis: Optional[dict] = None
    total_cost_usd: float = 0.0
    steps: list[StepOut] = []


class StatusUpdate(BaseModel):
    application_status: ApplicationStatus


class ResumeIn(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def not_blank(cls, v):
        if not v.strip():
            raise ValueError("resume text is empty")
        return v


class ResumeOut(BaseModel):
    id: str
    profile: dict
    model: str
    chars: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    created_at: datetime
    parsed: Optional[bool] = None   # set on upload: False means cache hit


def resume_out(r: Resume, parsed: Optional[bool] = None) -> ResumeOut:
    # raw_text is deliberately not returned: it may hold contact details.
    return ResumeOut(
        id=r.id,
        profile=r.profile,
        model=r.model,
        chars=len(r.raw_text),
        input_tokens=r.input_tokens or 0,
        output_tokens=r.output_tokens or 0,
        cost_usd=r.cost_usd or 0.0,
        created_at=r.created_at,
        parsed=parsed,
    )


def to_out(job: Job) -> JobOut:
    return JobOut(
        id=job.id,
        title=job.title,
        company=job.company,
        status=job.status.value,
        application_status=job.application_status.value,
        fit_score=job.fit_score,
        requirements=job.requirements,
        fit_analysis=job.fit_analysis,
        total_cost_usd=job.total_cost_usd or 0.0,
        steps=[
            StepOut(
                step_number=s.step_number,
                step_name=s.step_name,
                status=s.status.value,
                attempts=s.attempts,
                duration_ms=s.duration_ms,
                cost_usd=s.cost_usd or 0.0,
                error=s.error,
            )
            for s in job.steps
        ],
    )


# ------------------------------------------------------------ endpoints


@app.post("/jobs", response_model=JobOut, status_code=201)
def create_job(body: JobCreate, session: Session = Depends(get_session)):
    """Submit a posting and run it through the pipeline.

    Synchronous for now: one user, a handful of postings a day. If this ever
    needs to handle bursts, the runner moves behind a queue and this returns
    202 with the job id instead.
    """
    job = Job(source_url=body.source_url, raw_text=body.raw_text)
    session.add(job)
    session.commit()

    PipelineRunner(session, STEPS).run(job.id)
    session.refresh(job)
    return to_out(job)


@app.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: str, session: Session = Depends(get_session)):
    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return to_out(job)


@app.get("/jobs", response_model=list[JobOut])
def list_jobs(
    status: Optional[ApplicationStatus] = Query(None),
    pipeline_status: Optional[JobStatus] = Query(None),
    session: Session = Depends(get_session),
):
    q = session.query(Job)
    if status:
        q = q.filter(Job.application_status == status)
    if pipeline_status:
        q = q.filter(Job.status == pipeline_status)
    return [to_out(j) for j in q.order_by(Job.created_at.desc()).all()]


@app.patch("/jobs/{job_id}/status", response_model=JobOut)
def update_status(job_id: str, body: StatusUpdate, session: Session = Depends(get_session)):
    """Update where YOU are with this job, not where the pipeline is."""
    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(404, "job not found")

    job.application_status = body.application_status
    session.commit()
    session.refresh(job)
    return to_out(job)


@app.post("/jobs/{job_id}/retry", response_model=JobOut)
def retry_job(job_id: str, session: Session = Depends(get_session)):
    """Manually re-run a dead-lettered job after fixing whatever broke.

    Clears the failed step so the runner will attempt it again. Completed
    steps stay completed, so this does not re-burn tokens on earlier work.
    """
    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    if job.status != JobStatus.DEAD_LETTER:
        raise HTTPException(409, f"job is {job.status.value}, not dead_letter")

    from .models import StepStatus

    for step in job.steps:
        if step.status == StepStatus.FAILED:
            step.status = StepStatus.PENDING
            step.attempts = 0
            step.error = None
    job.status = JobStatus.PROCESSING
    session.commit()

    PipelineRunner(session, STEPS).run(job.id)
    session.refresh(job)
    return to_out(job)


@app.put("/resume", response_model=ResumeOut)
def put_resume(body: ResumeIn, session: Session = Depends(get_session)):
    """Set the resume that every job is compared against.

    Parsed with one LLM call and stored. Re-uploading the same text returns
    the stored parse with parsed=false and costs nothing.

    No runner here, so no automatic retry: a transient API failure returns
    503 and the caller re-submits. This is a one-off, user-initiated call.
    """
    try:
        resume, parsed = resume_module.upload(session, body.text)
    except StepFailed as exc:
        raise HTTPException(503, f"model API unavailable, try again: {exc}")
    except ModelRefused as exc:
        raise HTTPException(422, str(exc))
    return resume_out(resume, parsed)


@app.get("/resume", response_model=ResumeOut)
def get_resume(session: Session = Depends(get_session)):
    resume = resume_module.current(session)
    if resume is None:
        raise HTTPException(404, "no resume uploaded")
    return resume_out(resume)


@app.get("/insights")
def insights(session: Session = Depends(get_session)):
    """Aggregate skill gaps across every posting tracked.

    This is the endpoint that makes the project worth using: not "what did
    this one job want" but "what do I keep missing across all of them".
    """
    jobs = session.query(Job).filter(Job.fit_analysis.isnot(None)).all()

    missing = Counter()
    met = Counter()
    for job in jobs:
        analysis = job.fit_analysis or {}
        missing.update(analysis.get("missing", []))
        met.update(analysis.get("met", []))

    return {
        "jobs_analyzed": len(jobs),
        "top_gaps": [{"skill": s, "count": c} for s, c in missing.most_common(15)],
        "strengths": [{"skill": s, "count": c} for s, c in met.most_common(15)],
        # resume parsing is LLM spend too, so it counts toward the total
        "total_cost_usd": round(
            sum(j.total_cost_usd or 0.0 for j in session.query(Job).all())
            + sum(r.cost_usd or 0.0 for r in session.query(Resume).all()),
            4,
        ),
    }


@app.get("/health")
def health(session: Session = Depends(get_session)):
    """Counts by pipeline status. Stranded jobs show up here as processing."""
    counts = {s.value: 0 for s in JobStatus}
    for job in session.query(Job).all():
        counts[job.status.value] += 1
    return {"status": "ok", "jobs": counts}
