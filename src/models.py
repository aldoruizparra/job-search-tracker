"""Database models for the job search tracker.

Two tables:
  Job  - one row per job posting you submit
  Step - one row per pipeline step for that job

The Step table is what makes crash-resume possible. Every step is written
to the database before it runs and updated after it finishes, so on restart
we can find exactly where we left off.
"""

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column, DateTime, Enum, Float, ForeignKey, Integer, String, Text, JSON
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def now_utc():
    return datetime.now(timezone.utc)


def new_id():
    return str(uuid.uuid4())


class JobStatus(str, enum.Enum):
    """Where a posting is in the processing pipeline."""
    PENDING = "pending"          # submitted, not processed yet
    PROCESSING = "processing"    # pipeline is running (or crashed mid-run)
    COMPLETE = "complete"        # all steps finished
    DEAD_LETTER = "dead_letter"  # a step exhausted its retries


class ApplicationStatus(str, enum.Enum):
    """Where YOU are with this job. Separate from pipeline status."""
    INTERESTED = "interested"
    APPLIED = "applied"
    SCREENING = "screening"
    INTERVIEWED = "interviewed"
    REJECTED = "rejected"
    OFFER = "offer"


class StepStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class Job(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, default=new_id)
    source_url = Column(String, nullable=True)
    raw_text = Column(Text, nullable=True)

    company = Column(String, nullable=True)
    title = Column(String, nullable=True)

    # filled in by the extract step
    requirements = Column(JSON, nullable=True)
    # filled in by the compare step
    fit_analysis = Column(JSON, nullable=True)
    fit_score = Column(Float, nullable=True)

    status = Column(Enum(JobStatus), default=JobStatus.PENDING, nullable=False)
    application_status = Column(
        Enum(ApplicationStatus), default=ApplicationStatus.INTERESTED, nullable=False
    )

    # cost tracking: the thing most agent projects skip
    total_input_tokens = Column(Integer, default=0)
    total_output_tokens = Column(Integer, default=0)
    total_cost_usd = Column(Float, default=0.0)

    created_at = Column(DateTime, default=now_utc)
    updated_at = Column(DateTime, default=now_utc, onupdate=now_utc)

    steps = relationship(
        "Step", back_populates="job", order_by="Step.step_number",
        cascade="all, delete-orphan"
    )


class Step(Base):
    """One step of the pipeline for one job.

    idempotency_key is the important column. It is derived from
    (job_id, step_name, attempt_input_hash), so if we retry a step after a
    timeout where the response was actually produced but lost, we can detect
    that the work was already done instead of paying for it twice.
    """
    __tablename__ = "steps"

    id = Column(String, primary_key=True, default=new_id)
    job_id = Column(String, ForeignKey("jobs.id"), nullable=False)

    step_number = Column(Integer, nullable=False)
    step_name = Column(String, nullable=False)

    status = Column(Enum(StepStatus), default=StepStatus.PENDING, nullable=False)
    idempotency_key = Column(String, nullable=False, unique=True)

    input_data = Column(JSON, nullable=True)
    output_data = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)

    attempts = Column(Integer, default=0)
    max_attempts = Column(Integer, default=3)

    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    cost_usd = Column(Float, default=0.0)
    duration_ms = Column(Integer, nullable=True)

    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=now_utc)

    job = relationship("Job", back_populates="steps")


class Resume(Base):
    """One parsed version of the user's resume. The newest row is current.

    Parsed once per distinct text, not once per job: content_hash lets an
    identical re-upload return the existing parse without paying for another
    LLM call. Older versions are kept so a past comparison can be traced to
    the resume it was made against.
    """
    __tablename__ = "resumes"

    id = Column(String, primary_key=True, default=new_id)
    raw_text = Column(Text, nullable=False)
    content_hash = Column(String, nullable=False, index=True)

    profile = Column(JSON, nullable=False)
    # "offline" when parsed without an API key; such a row is a placeholder
    # and must never satisfy the re-upload cache.
    model = Column(String, nullable=False)

    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    cost_usd = Column(Float, default=0.0)

    created_at = Column(DateTime, default=now_utc)
