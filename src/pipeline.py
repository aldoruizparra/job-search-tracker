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

extract makes a real Anthropic call. compare and score are still stubs with
their output shapes fixed; replacing them changes handler bodies only.
"""

import json
import logging
import os
from functools import lru_cache

import anthropic

from .runner import StepFailed

log = logging.getLogger("pipeline")

EXTRACT_MODEL = os.getenv("EXTRACT_MODEL", "claude-opus-5-5")

# USD per million tokens, (input, output). Cost is priced off response.model,
# not the requested model, because a server-side fallback can change which
# model actually served the call. Opus 5 and Opus 4.8 are the fallback targets.
PRICING = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}


class ModelRefused(Exception):
    """The model declined the request. Not retryable: same input, same answer."""


@lru_cache(maxsize=1)
def get_client() -> anthropic.Anthropic:
    """Shared client, built on first use so importing this module needs no key.

    max_retries=0 because the runner owns retries. With the SDK's default of
    2, every runner attempt would hide up to three API calls, and the
    attempts column on the step row would undercount what we actually paid.
    """
    return anthropic.Anthropic(max_retries=0, timeout=60.0)


def is_offline() -> bool:
    """True when no Anthropic credential is configured.

    Offline mode lets the service run end to end without a key (for demos,
    or anyone cloning the repo): LLM steps return placeholder output at zero
    cost instead of failing every job.
    """
    return not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    in_rate, out_rate = PRICING[model]
    return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000


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

    # Next: real fetch with timeout handling and bot-detection fallback.
    # Job boards block scrapers aggressively, so this is where StepFailed
    # will actually get raised in practice.
    raise StepFailed("URL ingestion not implemented yet, paste the text instead")


# --------------------------------------------------------------- extract


def _nullable(kind: str) -> dict:
    return {"anyOf": [{"type": kind}, {"type": "null"}]}


# Structured outputs constrain the response to this schema, so the shape the
# compare step was written against is enforced by the API, not by hope.
REQUIREMENTS_SCHEMA = {
    "type": "object",
    "properties": {
        "title": _nullable("string"),
        "company": _nullable("string"),
        "required_skills": {"type": "array", "items": {"type": "string"}},
        "preferred_skills": {"type": "array", "items": {"type": "string"}},
        "years_experience": _nullable("integer"),
        "location": _nullable("string"),
        "salary_range": _nullable("string"),
    },
    "required": [
        "title", "company", "required_skills", "preferred_skills",
        "years_experience", "location", "salary_range",
    ],
    "additionalProperties": False,
}

EXTRACT_SYSTEM = """\
You extract structured requirements from a job posting.

required_skills are skills the posting says a candidate must have.
preferred_skills are ones it calls nice-to-have, preferred, or a plus.
Name each skill as a short canonical term ("Kubernetes", not "experience
running Kubernetes in production") so the same skill matches across
postings. years_experience is the minimum number of years asked for.
salary_range is the range as written, including currency.

Use null for any field the posting does not state. Do not infer or guess."""


def extract(job, prior):
    """Pull structured requirements out of the posting text with one LLM call.

    Failure mapping, which is the part that matters:
      rate limit, timeout, connection error, 5xx  ->  StepFailed (retry)
      any other API error (400, 401, 404, ...)    ->  propagates (a bug)
      refusal                                     ->  ModelRefused (propagates)
      truncated or unparseable output             ->  propagates (a bug)
    """
    text = (prior or {}).get("text", "")
    if not text:
        raise StepFailed("no text to extract from")

    if is_offline():
        log.warning("extract running offline, no API key", extra={"job_id": job.id})
        return no_cost({
            "title": None,
            "company": None,
            "required_skills": [],
            "preferred_skills": [],
            "years_experience": None,
            "location": None,
            "salary_range": None,
        })

    # Fail before paying for a call we could not price.
    if EXTRACT_MODEL not in PRICING:
        raise ValueError(f"no pricing for model {EXTRACT_MODEL}")

    try:
        response = get_client().beta.messages.create(
            model=EXTRACT_MODEL,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={
                "effort": "low",
                "format": {"type": "json_schema", "schema": REQUIREMENTS_SCHEMA},
            },
            system=EXTRACT_SYSTEM,
            messages=[{"role": "user", "content": text}],
        )
    # APITimeoutError subclasses APIConnectionError, so this covers both.
    except (anthropic.RateLimitError, anthropic.APIConnectionError) as exc:
        raise StepFailed(f"{type(exc).__name__}: {exc}") from exc
    except anthropic.InternalServerError as exc:   # 5xx, including 529 overloaded
        raise StepFailed(f"server error {exc.status_code}: {exc}") from exc

    if response.stop_reason == "refusal":
        raise ModelRefused(f"extract refused: {response.stop_details}")
    if response.stop_reason == "max_tokens":
        raise ValueError("extract output truncated at max_tokens")

    body = next(b.text for b in response.content if b.type == "text")
    requirements = json.loads(body)

    usage = response.usage
    log.info(
        "extract done",
        extra={"job_id": job.id, "model": response.model, "chars": len(text)},
    )
    return {
        "output": requirements,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cost_usd": cost_usd(response.model, usage.input_tokens, usage.output_tokens),
    }


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
