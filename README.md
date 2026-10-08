# Job Search Tracker

A backend service that processes job postings through an LLM pipeline and tracks
applications. The engineering focus is durability: every pipeline step is
persisted before and after execution, so a crash resumes from the last completed
step instead of re-running expensive LLM calls.

**Status: in progress.** The persistence layer, pipeline runner, and HTTP API are
built and tested. `extract` and resume parsing make real Anthropic calls;
`compare` and `score` are still stubs with their output shapes fixed.

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
- A handler that raises anything other than `StepFailed` is treated as a bug:
  the step is marked `FAILED` on the first attempt and the job dead-letters
  immediately. Retrying a bug pays for the same failure three times. A process
  crash (`SystemExit`, Ctrl-C) is deliberately *not* caught, so the step stays
  `RUNNING` and resumes on the next boot.

The API wires `resume_all()` into the FastAPI lifespan handler. Without that
call, the durability work never actually fires in production and crashed jobs
sit stranded forever.

## API

```
POST   /jobs                 submit a posting (URL or text), runs the pipeline
GET    /jobs/{id}            requirements, fit analysis, and per-step detail
GET    /jobs?status=applied  filter by application status or pipeline status
PATCH  /jobs/{id}/status     update application status
PUT    /resume               set the resume every job is compared against
GET    /resume               the current parsed resume profile
POST   /jobs/{id}/retry      re-run a dead-lettered job
GET    /insights             aggregate skill gaps across all postings
GET    /health               job counts by pipeline status
```

`/insights` is the endpoint that makes this worth using. Not "what did this one
job want" but "what do I keep missing across all of them."

`PUT /resume` parses the resume into a skills profile with one LLM call and
stores it. Re-uploading the same text returns the stored parse with
`"parsed": false` and costs nothing. The response never echoes the raw text.

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

**The runner owns retries, not the SDK.** The Anthropic client is built with
`max_retries=0`. With the SDK default of 2, each runner attempt could hide three
API calls, so the `attempts` column would undercount what was actually paid and
the runner's jittered backoff would stack on top of the SDK's own.

**API errors are classified, not caught wholesale.** In `extract`, rate limits,
timeouts, connection errors, and 5xx (including 529 overloaded) become
`StepFailed` and are retried. A 400, 401, or 404, a refusal, a response
truncated at `max_tokens`, or output that does not parse all propagate. Retrying
those spends three attempts reproducing the same failure.

**The output shape is enforced by the API.** `extract` uses structured outputs
with a JSON schema matching the shape the `compare` step was written against,
so a well-formed response cannot drift from it.

**Cost is priced by the model that served the call.** Requests opt into
server-side refusal fallbacks, which can route a declined request to another
model. Cost is computed from `response.model`, not the requested one. An
unpriced model fails before the call rather than after paying for it.

**The resume is parsed once, not per job.** It belongs to the user, not to a
posting, so it is stored in its own table and reused by every comparison
instead of being a pipeline step that would re-bill the same call per job. A
whitespace-insensitive content hash detects identical re-uploads. An offline
placeholder parse never satisfies that check, so adding an API key later and
re-uploading produces a real parse instead of a cached empty one.

**Contact details have nowhere to land.** The resume profile schema has fields
for skills, years of experience, titles, and education only, so a name, email,
or phone number in the text cannot reach the database through the parse. The
raw text is stored (to re-parse later) but never returned by the API.

**Resume upload has no runner, so it does not retry.** It is a one-off,
user-initiated call: a transient API failure returns `503` and the caller
re-submits, rather than the request blocking through backoff.

**`compare` and `score` are still stubs with fixed output shapes.** Swapping in
real calls replaces their bodies only; the runner, API, and tests stay as they
are.

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
  llm.py        Anthropic client, error mapping, pricing, offline mode
  resume.py     resume parsing, stored once and reused by every job
  pipeline.py   the five step handlers
  db.py         engine and session setup
  api.py        FastAPI endpoints and startup recovery
tests/
  conftest.py       disables retry backoff, fakes the Anthropic client
  test_runner.py    12 tests on the execution engine
  test_api.py       12 tests on the HTTP layer
  test_extract.py   19 tests on extract error handling and offline mode
  test_resume.py    12 tests on resume parsing, caching, and the endpoints
```

## Tests

**55 tests, all passing, under a second.** No test touches the network: an
autouse fixture in `conftest.py` replaces the Anthropic client with a fake.

Two of them carry the project:

`test_crash_resumes_without_rerunning_completed_steps` runs the pipeline, raises
`SystemExit` inside step two, opens a **fresh session to simulate a new
process**, calls `resume_all()`, and asserts the first step was not executed a
second time.

`test_startup_recovers_stranded_jobs` hand-builds a job stranded mid-pipeline,
boots the app through `TestClient` so the lifespan handler fires, and asserts the
job completed **and** that the already-done step still shows `attempts == 0`.
That is the same guarantee, proven at the HTTP layer rather than in isolation.

`test_extract.py` covers the failure mapping for the real LLM call: each
transient API error becomes a retry, each client error, refusal, truncation,
and malformed response does not, and a rate-limited extract retried through the
real runner records three attempts but bills one call.

Also covered: handler bugs dead-lettering on the first attempt instead of being re-run on every restart, retry on transient failure, dead-letter after exhausted attempts,
dead-letter jobs not silently retrying, cost rollup, output threading between
steps, idempotency key stability, multiple stranded jobs recovered at once,
request validation, status filtering, and insight aggregation.

## Running it

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

```
55 passed in 0.45s
```

Start the service:

```bash
python -m uvicorn src.api:app --reload
```

With no API key set, the service runs in **offline mode**: `extract` returns
placeholder output (empty skill lists, null fields) at zero cost, so the whole
pipeline can be exercised without an Anthropic account. To make real calls:

```bash
cp .env.example .env               # then set ANTHROPIC_API_KEY
python -m uvicorn src.api:app --reload --env-file .env
```

Interactive docs at `http://localhost:8000/docs`.

```bash
curl -X POST localhost:8000/jobs \
  -H 'Content-Type: application/json' \
  -d '{"raw_text": "Senior Backend Engineer. Python, AWS, Kubernetes. 5+ years."}'
```

## What is next

- [x] Real LLM call in `extract` via the Anthropic API
- [ ] Real LLM calls in `compare` and `score`
- [ ] URL ingestion with timeout and bot-detection handling
- [x] Resume parsing to feed the comparison step
- [ ] Structured JSON logging with `job_id` on every line
- [ ] `/metrics`: job counts by status, p95 step latency, token spend
- [ ] Dockerfile and compose setup

## Stack

Python, FastAPI, SQLAlchemy, SQLite, pytest, Anthropic API.