"""Pipeline step handlers.

Every handler has the same contract:

    handler(job, prior_output) -> {
        "output":        dict,   # passed to the next step, persisted
        "input_tokens":  int,    # 0 for non-LLM steps
        "output_tokens": int,
        "cost_usd":      float,
    }

Raise StepFailed for anything retryable (timeout, rate limit, transient 5xx).
Let other exceptions propagate: those are bugs, not conditions to retry.

The extract/compare/score handlers are stubs right now. Weekend 2 replaces
their bodies with real Anthropic API calls. The runner does not change.
"""

import logging

from .runner import StepFailed

log = logging.getLogger("pipeline")


def no_cost(output: dict) -> dict:
    """Helper for steps that do not call an LLM."""
    return {"output": output, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}


# ---------------------------------------------------------------- ingest


def ingest(job, prior):
    """Get the posting text, either from pasted text or by fetching a URL."""
    if job.raw_text:
        return no_cost({"text": job.raw_text, "source": "pasted"})

    if not job.source_url:
        # Not retryable: no amount of trying fixes a job with no input.
        raise ValueError(f"job {job.id} has neither raw_text nor source_url")

    # Weekend 2: real fetch with timeout handling and bot-detection fallback.
    # Job boards block scrapers aggressively, so this is where StepFailed
    # will actually get raised in practice.
    raise StepFailed("URL ingestion not implemented yet, paste the text instead")


# --------------------------------------------------------------- extract


def extract(job, prior):
    """Pull structured requirements out of the posting text.

    Becomes an LLM call. Shape of the output is fixed now so the compare
    step can be written against it.
    """
    text = (prior or {}).get("text", "")
    if not text:
        raise StepFailed("no text to extract from")

    stub = {
        "title": None,
        "company": None,
        "required_skills": [],
        "preferred_skills": [],
        "years_experience": None,
        "location": None,
        "salary_range": None,
    }
    log.info("extract stub", extra={"job_id": job.id, "chars": len(text)})
    return no_cost(stub)


# --------------------------------------------------------------- compare


def compare(job, prior):
    """Match extracted requirements against the resume."""
    reqs = prior or {}
    stub = {
        "met": [],
        "partial": [],
        "missing": [],
        "notes": "comparison not implemented yet",
    }
    log.info("compare stub", extra={"job_id": job.id, "skills": len(reqs.get("required_skills", []))})
    return no_cost(stub)


# ----------------------------------------------------------------- score


def score(job, prior):
    """Produce a fit score with reasoning, not just a number."""
    analysis = prior or {}
    met = len(analysis.get("met", []))
    missing = len(analysis.get("missing", []))
    total = met + missing

    value = round(met / total, 2) if total else 0.0
    return no_cost({"fit_score": value, "reasoning": "scoring not implemented yet"})


# ----------------------------------------------------------------- store


def store(job, prior):
    """Write the final analysis back onto the Job row.

    Runs last so the job carries its results directly, without anyone
    having to walk the step history to find them.
    """
    result = prior or {}
    job.fit_score = result.get("fit_score")

    # pull the richer outputs off the steps that produced them
    by_name = {s.step_name: s.output_data for s in job.steps}
    reqs = by_name.get("extract") or {}
    job.requirements = reqs
    job.fit_analysis = by_name.get("compare")
    job.title = reqs.get("title") or job.title
    job.company = reqs.get("company") or job.company

    return no_cost({"stored": True})


# The pipeline, in order. The runner takes this list as-is.
STEPS = [
    ("ingest", ingest),
    ("extract", extract),
    ("compare", compare),
    ("score", score),
    ("store", store),
]
