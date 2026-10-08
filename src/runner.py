"""Pipeline runner.

The whole point of this file: a job is a sequence of steps, each step is
persisted before and after it runs, and the runner can be killed at any
moment and resume from the last completed step.

Design decisions worth knowing (put these in your README):

1. At-least-once, not exactly-once. True exactly-once across a process
   boundary and a third-party API is not achievable. Instead every step
   carries an idempotency key, so a duplicate execution is detectable and
   cheap rather than impossible.

2. Steps are resumed, not restarted. On startup we look for jobs stuck in
   PROCESSING and continue from the first step that is not DONE.

3. A step that exhausts max_attempts moves the whole job to DEAD_LETTER
   rather than disappearing, so failures stay inspectable.
"""

import hashlib
import json
import logging
import os
import random
import time
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from .models import Job, JobStatus, Step, StepStatus

log = logging.getLogger("runner")


def now_utc():
    return datetime.now(timezone.utc)


def make_idempotency_key(job_id: str, step_name: str, payload: dict) -> str:
    """Stable hash of (job, step, input).

    If the same step runs again with the same input, the key matches, which
    is how we detect a retry of work that may already have completed.
    """
    blob = json.dumps(payload, sort_keys=True, default=str)
    digest = hashlib.sha256(f"{job_id}:{step_name}:{blob}".encode()).hexdigest()
    return digest[:32]


class StepFailed(Exception):
    """Raised by a step handler when the attempt failed but is retryable."""


class PipelineRunner:
    """Executes a fixed sequence of steps against a job."""

    def __init__(self, session: Session, steps: list, max_attempts: int = 3):
        """
        steps: list of (step_name, handler) tuples, in execution order.
               Each handler has signature handler(job, prior_output) -> dict
               and may raise StepFailed to trigger a retry.
        """
        self.session = session
        self.steps = steps
        self.max_attempts = max_attempts

    # ----- public API ---------------------------------------------------

    def run(self, job_id: str) -> Job:
        """Run (or resume) the pipeline for one job."""
        job = self.session.get(Job, job_id)
        if job is None:
            raise ValueError(f"no job {job_id}")

        if job.status == JobStatus.COMPLETE:
            log.info("job already complete", extra={"job_id": job_id})
            return job

        job.status = JobStatus.PROCESSING
        self.session.commit()

        prior_output = None

        for index, (step_name, handler) in enumerate(self.steps):
            step = self._get_or_create_step(job, index, step_name, prior_output)

            # Resume: this step already finished on a previous run.
            if step.status == StepStatus.DONE:
                log.info(
                    "skipping completed step",
                    extra={"job_id": job_id, "step": step_name},
                )
                prior_output = step.output_data
                continue

            if step.status == StepStatus.FAILED:
                job.status = JobStatus.DEAD_LETTER
                self.session.commit()
                log.error(
                    "job in dead letter",
                    extra={"job_id": job_id, "step": step_name},
                )
                return job

            ok = self._execute_step(job, step, handler, prior_output)
            if not ok:
                job.status = JobStatus.DEAD_LETTER
                self.session.commit()
                return job

            prior_output = step.output_data

        job.status = JobStatus.COMPLETE
        self.session.commit()
        log.info("job complete", extra={"job_id": job_id})
        return job

    def resume_all(self) -> list:
        """Find every job stranded mid-pipeline and continue it.

        Call this on service startup. This is what turns a crash into a
        pause instead of a data loss.
        """
        stranded = (
            self.session.query(Job)
            .filter(Job.status == JobStatus.PROCESSING)
            .all()
        )
        log.info("resuming stranded jobs", extra={"count": len(stranded)})
        return [self.run(job.id) for job in stranded]

    # ----- internals ----------------------------------------------------

    def _get_or_create_step(self, job, index, step_name, prior_output):
        """Fetch the existing step row, or create it before any work happens.

        Writing the row BEFORE execution is what makes resume possible. If we
        crash during the handler, the row is already there in RUNNING state.
        """
        existing = (
            self.session.query(Step)
            .filter(Step.job_id == job.id, Step.step_number == index)
            .one_or_none()
        )
        if existing:
            return existing

        payload = {"prior": prior_output}
        step = Step(
            job_id=job.id,
            step_number=index,
            step_name=step_name,
            status=StepStatus.PENDING,
            idempotency_key=make_idempotency_key(job.id, step_name, payload),
            input_data=payload,
            max_attempts=self.max_attempts,
        )
        self.session.add(step)
        self.session.commit()
        return step

    def _execute_step(self, job, step, handler, prior_output) -> bool:
        """Run one step with retries. Returns False if it exhausted attempts."""
        while step.attempts < step.max_attempts:
            step.attempts += 1
            step.status = StepStatus.RUNNING
            step.started_at = now_utc()
            self.session.commit()

            start = time.monotonic()
            try:
                result = handler(job, prior_output)

                step.output_data = result.get("output")
                step.input_tokens = result.get("input_tokens", 0)
                step.output_tokens = result.get("output_tokens", 0)
                step.cost_usd = result.get("cost_usd", 0.0)
                step.duration_ms = int((time.monotonic() - start) * 1000)
                step.status = StepStatus.DONE
                step.finished_at = now_utc()

                # roll cost up to the job
                job.total_input_tokens += step.input_tokens
                job.total_output_tokens += step.output_tokens
                job.total_cost_usd += step.cost_usd

                self.session.commit()
                log.info(
                    "step done",
                    extra={
                        "job_id": job.id,
                        "step": step.step_name,
                        "attempt": step.attempts,
                        "duration_ms": step.duration_ms,
                        "cost_usd": step.cost_usd,
                    },
                )
                return True

            except StepFailed as exc:
                step.error = str(exc)
                step.duration_ms = int((time.monotonic() - start) * 1000)
                self.session.commit()

                log.warning(
                    "step failed, will retry",
                    extra={
                        "job_id": job.id,
                        "step": step.step_name,
                        "attempt": step.attempts,
                        "error": str(exc),
                    },
                )

                if step.attempts < step.max_attempts:
                    time.sleep(self._backoff(step.attempts))

            # Anything else is a bug: retrying it pays for the same failure
            # again. Dead-letter now, so the job is inspectable instead of
            # stranded in PROCESSING and re-run by resume_all() on every boot.
            # Exception, not BaseException: SystemExit and KeyboardInterrupt
            # are the process dying, and must leave the step RUNNING to resume.
            except Exception as exc:
                # The handler may have left the session mid-transaction.
                self.session.rollback()
                step.status = StepStatus.FAILED
                step.error = f"{type(exc).__name__}: {exc}"
                step.duration_ms = int((time.monotonic() - start) * 1000)
                step.finished_at = now_utc()
                self.session.commit()
                log.exception(
                    "step raised a non-retryable error",
                    extra={"job_id": job.id, "step": step.step_name},
                )
                return False

        step.status = StepStatus.FAILED
        step.finished_at = now_utc()
        self.session.commit()
        log.error(
            "step exhausted retries",
            extra={"job_id": job.id, "step": step.step_name},
        )
        return False

    @staticmethod
    def _backoff(attempt: int) -> float:
        """Exponential backoff with jitter.

        Jitter matters: without it, many clients retrying at the same moment
        hit the API in a synchronized wave and keep failing together.

        BACKOFF_DISABLED=1 turns the sleep off so the test suite does not
        spend 15 seconds proving that retries wait.
        """
        if os.getenv("BACKOFF_DISABLED") == "1":
            return 0.0
        base = min(2 ** attempt, 30)
        return base * (0.5 + random.random() * 0.5)
