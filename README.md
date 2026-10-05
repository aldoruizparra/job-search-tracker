# Job Search Tracker

A backend service that processes job postings through an LLM pipeline and tracks
applications. The engineering focus is durability: every pipeline step is
persisted before and after execution, so a crash resumes from the last completed
step instead of re-running expensive LLM calls.

**Status: in progress.** The persistence layer and pipeline runner are built and
tested. The API layer and real LLM steps are next.

## The problem

Tracking applications in a spreadsheet tells you nothing useful. I wanted
something that could extract structured requirements from a posting, compare
them against my resume, and then aggregate across everything I had applied to:
*which skills do I keep missing?*

The interesting engineering problem is that each pipeline step is an expensive,
slow, failure-prone LLM call. Most agent code assumes the happy path. This
assumes failure.

## Pipeline

```
ingest  →  extract  →  compare  →  score  →  store
```

| Step | What it does |
| --- | --- |
| `ingest` | Fetch a posting from a URL, or accept pasted text |
| `extract` | LLM call producing structured requirements as JSON |
| `compare` | Match requirements against a parsed resume |
| `score` | Fit score with reasoning, not just a number |
| `store` | Persist with application status |

Each step is a row in the database. Each row is written **before** the step runs
and updated after it finishes. That ordering is what makes resume possible.

## Durability model

The runner's core guarantee: **a crash is a pause, not a restart.**

- Every step row is created in `PENDING` before the handler executes.
- On startup, `resume_all()` finds jobs stranded in `PROCESSING` and continues
  from the first step that is not `DONE`.
- Completed steps are skipped, not re-run. No duplicate token spend.
- Each step carries an **idempotency key**, a stable hash of
  `(job_id, step_name, input)`, so a retry after a lost response is detectable.
- Transient failures retry with exponential backoff **and jitter**.
- A step that exhausts its attempts moves the job to `DEAD_LETTER` rather than
  disappearing, so failures stay inspectable.

## Design tradeoffs

**At-least-once, not exactly-once.** True exactly-once across a process boundary
and a third-party API is not achievable. Instead, every step carries an
idempotency key so duplicate execution is *detectable and cheap* rather than
impossible.

**Backoff has jitter.** Without it, many clients retrying simultaneously hit the
API in a synchronized wave and keep failing together.

**Dead-letter over silent failure.** A job that exhausts retries lands in an
inspectable state. Losing a failed job is worse than keeping a broken one.

**Pipeline status and application status are separate fields.** Whether the
extraction succeeded has nothing to do with whether I have applied yet.
Collapsing them into one enum is the obvious mistake and it bites later.

**SQLite for now.** Single user, single process, no concurrency pressure. The
SQLAlchemy models move to Postgres unchanged if that assumption breaks.

**Cost tracking is first-class.** Token usage and USD cost are recorded per step
and rolled up per job. LLM calls are the expensive part of this system and most
implementations never measure them.

## What is built

**`src/models.py`** — SQLAlchemy models.

- `Job`: the posting, extracted requirements, fit analysis, application status,
  and rolled-up token and cost totals.
- `Step`: one row per pipeline step with status, attempt count, idempotency key,
  per-step token usage, cost, and duration.

**`src/runner.py`** — the pipeline runner.

- `run(job_id)` executes or resumes a job, skipping completed steps.
- `resume_all()` recovers every job stranded mid-pipeline. Intended to be called
  on service startup.
- `_execute_step()` handles retries, backoff, cost rollup, and the transition to
  dead-letter.
- `make_idempotency_key()` produces a stable hash over job, step, and input.

**`tests/test_runner.py`** — 10 tests, all passing.

The one that matters is `test_crash_resumes_without_rerunning_completed_steps`.
It runs the pipeline, raises `SystemExit` inside step two, opens a **fresh
session to simulate a new process**, calls `resume_all()`, and then asserts the
first step was not executed a second time.

Also covered: retry on transient failure, dead-letter after exhausted attempts,
dead-letter jobs not silently retrying, cost rollup from steps to job, output
threading between steps, idempotency key stability, and resuming multiple
stranded jobs at once.

## Running it

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

```
10 passed
```

## What is next

- [ ] `src/api.py` — FastAPI endpoints, with `resume_all()` wired into startup
- [ ] Real LLM steps via the Anthropic API, replacing the test handlers
- [ ] URL ingestion with proper handling of timeouts and bot detection
- [ ] Resume parsing and the comparison step
- [ ] `GET /insights` — aggregate skill gaps across all tracked postings
- [ ] Structured JSON logging with `job_id` on every line
- [ ] `/metrics` endpoint: task counts by status, p95 step latency, token spend
- [ ] Dockerfile and compose setup

## Planned API

```
POST   /jobs                 submit a posting (URL or text)
GET    /jobs/{id}            extracted requirements and fit analysis
GET    /jobs?status=applied  filter the pipeline
PATCH  /jobs/{id}/status     update application status
GET    /insights             aggregate skill gaps across all postings
```

## Stack

Python, FastAPI, SQLAlchemy, SQLite, pytest, Anthropic API.
