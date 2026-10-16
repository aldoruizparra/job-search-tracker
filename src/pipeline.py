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

extract makes a real Anthropic call through llm.structured_call, which owns
the error mapping. compare and score are deterministic (no LLM): compare
matches skill names against the current resume, score turns that into a
number with reasoning.
"""

import logging

from sqlalchemy.orm import object_session

from . import llm
from . import resume as resume_module
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

    # Next: real fetch with timeout handling and bot-detection fallback.
    # Job boards block scrapers aggressively, so this is where StepFailed
    # will actually get raised in practice.
    raise StepFailed("URL ingestion not implemented yet, paste the text instead")


# --------------------------------------------------------------- extract


# Structured outputs constrain the response to this schema, so the shape the
# compare step was written against is enforced by the API, not by hope.
REQUIREMENTS_SCHEMA = {
    "type": "object",
    "properties": {
        "title": llm.nullable("string"),
        "company": llm.nullable("string"),
        "required_skills": {"type": "array", "items": {"type": "string"}},
        "preferred_skills": {"type": "array", "items": {"type": "string"}},
        "years_experience": llm.nullable("integer"),
        "location": llm.nullable("string"),
        "salary_range": llm.nullable("string"),
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
    """Pull structured requirements out of the posting text with one LLM call."""
    text = (prior or {}).get("text", "")
    if not text:
        raise StepFailed("no text to extract from")

    if llm.is_offline():
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

    result = llm.structured_call(EXTRACT_SYSTEM, text, REQUIREMENTS_SCHEMA)
    log.info(
        "extract done",
        extra={"job_id": job.id, "model": result["model"], "chars": len(text)},
    )
    return result


# --------------------------------------------------------------- compare


def _fold(skill: str) -> str:
    """Case- and whitespace-insensitive key for matching skill names.

    Deliberately exact otherwise: "k8s" does not match "Kubernetes" yet.
    Synonym normalization is its own piece of work.
    """
    return " ".join(skill.split()).casefold()


def compare(job, prior):
    """Match extracted requirements against the current resume.

    Deterministic, no LLM call, so it is free, works offline, and can be
    re-run whenever the resume changes (see PipelineRunner.rerun_from).

    With no usable resume (none uploaded, or only the offline placeholder,
    whose empty skill list would mark everything missing) the job still
    completes: the output carries the posting's priorities and no match.
    """
    reqs = prior or {}
    required = reqs.get("required_skills") or []
    preferred = reqs.get("preferred_skills") or []
    priorities = {
        "required": required,
        "preferred": preferred,
        "years_experience": reqs.get("years_experience"),
    }

    current = resume_module.current(object_session(job))
    if current is None or current.model == resume_module.OFFLINE:
        log.info("compare without resume", extra={"job_id": job.id})
        return no_cost({
            "mode": "no_resume",
            "resume_id": None,
            "met": [],
            "missing": [],
            "partial": [],
            "priorities": priorities,
            "notes": "no parsed resume yet; showing what the posting prioritizes",
        })

    have = {_fold(s) for s in current.profile.get("skills", [])}
    met_required = [s for s in required if _fold(s) in have]
    met_preferred = [s for s in preferred if _fold(s) in have]
    output = {
        "mode": "resume",
        "resume_id": current.id,
        # met and missing are what /insights aggregates across jobs
        "met": met_required + met_preferred,
        "missing": [s for s in required if _fold(s) not in have],
        "met_required": met_required,
        "met_preferred": met_preferred,
        "missing_preferred": [s for s in preferred if _fold(s) not in have],
        "partial": [],   # reserved for synonym and fuzzy matches
        "resume_years": current.profile.get("years_experience"),
        "priorities": priorities,
        "notes": "exact skill-name match, case-insensitive",
    }
    log.info(
        "compare done",
        extra={"job_id": job.id, "resume_id": current.id, "met": len(output["met"])},
    )
    return no_cost(output)


# ----------------------------------------------------------------- score


def _join(items: list) -> str:
    return ", ".join(items) if items else "none listed"


def score(job, prior):
    """Produce a fit score with reasoning, not just a number.

    The score is the share of *required* skills the resume shows. Preferred
    skills and years go in the reasoning, not the number, so the number
    stays easy to interpret. No score at all (None, not 0.0) when there is
    no resume or the posting lists no required skills: a 0 would read as
    "terrible fit" when the truth is "not measurable".
    """
    analysis = prior or {}
    p = analysis.get("priorities") or {}
    years = p.get("years_experience")
    years_text = f"{years}+ years" if years is not None else "years not stated"

    if analysis.get("mode") != "resume":
        reasoning = (
            f"No resume to compare against yet. This posting prioritizes: "
            f"must have {_join(p.get('required', []))}; "
            f"nice to have {_join(p.get('preferred', []))}; {years_text}."
        )
        return no_cost({"fit_score": None, "reasoning": reasoning})

    required = p.get("required", [])
    preferred = p.get("preferred", [])
    met_req = analysis.get("met_required", [])
    parts = []
    if required:
        parts.append(f"Matches {len(met_req)} of {len(required)} required skills")
        if analysis.get("missing"):
            parts.append(f"missing {_join(analysis['missing'])}")
    else:
        parts.append("Posting lists no required skills")
    if preferred:
        parts.append(f"{len(analysis.get('met_preferred', []))} of {len(preferred)} preferred")
    resume_years = analysis.get("resume_years")
    if years is not None and resume_years is not None:
        parts.append(f"asks {years}+ years, resume shows {resume_years}")

    value = round(len(met_req) / len(required), 2) if required else None
    return no_cost({"fit_score": value, "reasoning": "; ".join(parts) + "."})


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
