# Job Search Tracker

A backend service that processes job postings through an LLM pipeline and tracks
applications. The engineering focus is durability: every pipeline step is
persisted before and after execution, so a crash resumes from the last completed
step instead of re-running expensive LLM calls.

**Status: in progress.** The persistence layer, pipeline runner, and HTTP API are
built and tested. The step handlers are stubs with their output shapes fixed;
real LLM calls are next.

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
| `store` | Write results back onto the job row |

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
  disappearing, so failures stay inspectable and can be retried deliberately.

The API wires `resume_all()` into the FastAPI lifespan handler. Without that
call, the durability work never actually fires in production and crashed jobs
sit stranded forever.

## API

```
POST   /jobs                 submit a posting (URL or text), runs the pipeline
GET    /jobs/{id}            requirements, fit analysis, and per-step detail
GET    /jobs?status=applied  filter by application status or pipeline status
PATCH  /jobs/{id}/status     update application status
POST   /jobs/{id}/retry      re-run a dead-lettered job
GET    /insights             aggregate skill gaps across all postings
GET    /health               job counts by pipeline status
```

`/insights` is the endpoint that makes this worth using. Not "what did this one
job want" but "what do I keep missing across all of them."

`/jobs/{id}/retry` resets only the **failed** step. Completed steps stay
completed, so retrying does not re-burn tokens on work that already succeeded.

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

**`POST /jobs` runs the pipeline synchronously.** One user, a handful of
postings a day, so the simpler thing is correct. If this ever needs to absorb
bursts, the runner moves behind a queue and the endpoint returns `202` with a
job id instead. The runner itself would not change.

**Step handlers are stubs with fixed output shapes.** `extract`, `compare`, and
`score` return the right structure with placeholder values. Swapping in real
Anthropic calls replaces their bodies only; the runner, API, and tests stay as
they are.

**SQLite for now.** Single user, single process, no concurrency pressure. The
SQLAlchemy models move to Postgres unchanged if that assumption breaks.

**Cost tracking is first-class.** Token usage and USD cost are recorded per step
and rolled up per job. LLM calls are the expensive part of this system and most
implementations never measure them.

## Layout

```
src/
  models.py     SQLAlchemy models: Job and Step
  runner.py     pipeline execution, resume, retry, dead-letter
  pipeline.py   the five step handlers
  db.py         engine and session setup
  api.py        FastAPI endpoints and startup recovery
tests/
  conftest.py       disables retry backoff so the suite runs in ~1s
  test_runner.py    10 tests on the execution engine
  test_api.py       11 tests on the HTTP layer
```

## Tests

**21 tests, all passing, ~1 second.**

Two of them carry the project:

`test_crash_resumes_without_rerunning_completed_steps` runs the pipeline, raises
`SystemExit` inside step two, opens a **fresh session to simulate a new
process**, calls `resume_all()`, and asserts the first step was not executed a
second time.

`test_startup_recovers_stranded_jobs` hand-builds a job stranded mid-pipeline,
boots the app through `TestClient` so the lifespan handler fires, and asserts the
job completed **and** that the already-done step still shows `attempts == 0`.
That is the same guarantee, proven at the HTTP layer rather than in isolation.

Also covered: retry on transient failure, dead-letter after exhausted attempts,
dead-letter jobs not silently retrying, cost rollup, output threading between
steps, idempotency key stability, multiple stranded jobs recovered at once,
request validation, status filtering, and insight aggregation.

## Running it

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

```
21 passed in 1.08s
```

Start the service:

```bash
python -m uvicorn src.api:app --reload
```

Interactive docs at `http://localhost:8000/docs`.

```bash
curl -X POST localhost:8000/jobs \
  -H 'Content-Type: application/json' \
  -d '{"raw_text": "Senior Backend Engineer. Python, AWS, Kubernetes. 5+ years."}'
```

## What is next

- [ ] Real LLM calls in `extract`, `compare`, and `score` via the Anthropic API
- [ ] URL ingestion with timeout and bot-detection handling
- [ ] Resume parsing to feed the comparison step
- [ ] Structured JSON logging with `job_id` on every line
- [ ] `/metrics`: job counts by status, p95 step latency, token spend
- [ ] Dockerfile and compose setup

## Stack

Python, FastAPI, SQLAlchemy, SQLite, pytest, Anthropic API.
[text](../Downloads/conftest.py)