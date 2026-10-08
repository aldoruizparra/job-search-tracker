"""Resume parsing.

A resume is parsed into a skills profile once, stored, and reused by every
job's compare step. It is not a pipeline step: it belongs to the user, not to
a posting, and parsing it per job would pay for the same call every time.

The profile schema has no fields for name, email, phone, or address, so
contact details never reach the database even when they are in the text.
"""

import hashlib
import logging

from sqlalchemy.orm import Session

from . import llm
from .models import Resume

log = logging.getLogger("resume")

PROFILE_SCHEMA = {
    "type": "object",
    "properties": {
        "skills": {"type": "array", "items": {"type": "string"}},
        "years_experience": llm.nullable("integer"),
        "titles": {"type": "array", "items": {"type": "string"}},
        "education": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["skills", "years_experience", "titles", "education"],
    "additionalProperties": False,
}

RESUME_SYSTEM = """\
You extract a skills profile from a resume so it can be matched against job
postings.

skills: every technical skill, tool, language, framework, platform, and
practice the resume shows. Name each as a short canonical term ("Kubernetes",
not "deployed services to Kubernetes clusters") so it matches the same skill
named in a posting.
years_experience: total years of professional experience, from the dates
given. null if the dates do not support a number.
titles: job titles held, most recent first.
education: degrees and certifications, one short entry each.

Do not include names, contact details, or employer addresses. Use only what
the resume states."""

OFFLINE = "offline"


def content_hash(text: str) -> str:
    """Whitespace-insensitive hash, so re-pasting with different line endings
    or trailing spaces does not count as a new resume."""
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def current(session: Session) -> Resume | None:
    """The resume compare should match against: the newest one stored."""
    return session.query(Resume).order_by(Resume.created_at.desc()).first()


def upload(session: Session, text: str) -> tuple[Resume, bool]:
    """Parse and store a resume. Returns (resume, parsed).

    parsed is False when the text matches the current resume and that parse
    was real: nothing is called and nothing is billed. An offline placeholder
    never short-circuits, so adding an API key later and re-uploading the same
    text produces a real parse.

    Raises StepFailed for transient API errors (the caller decides whether to
    retry) and lets everything else propagate, as llm.structured_call does.
    """
    digest = content_hash(text)
    latest = current(session)
    if latest and latest.content_hash == digest and latest.model != OFFLINE:
        log.info("resume unchanged, reusing parse", extra={"resume_id": latest.id})
        return latest, False

    if llm.is_offline():
        log.warning("resume parsed offline, no API key")
        result = {
            "output": {"skills": [], "years_experience": None, "titles": [], "education": []},
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "model": OFFLINE,
        }
    else:
        result = llm.structured_call(RESUME_SYSTEM, text, PROFILE_SCHEMA)

    resume = Resume(
        raw_text=text,
        content_hash=digest,
        profile=result["output"],
        model=result["model"],
        input_tokens=result["input_tokens"],
        output_tokens=result["output_tokens"],
        cost_usd=result["cost_usd"],
    )
    session.add(resume)
    session.commit()
    log.info(
        "resume parsed",
        extra={"resume_id": resume.id, "model": resume.model, "cost_usd": resume.cost_usd},
    )
    return resume, True
