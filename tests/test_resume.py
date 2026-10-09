"""Resume parsing tests.

The resume below is fabricated, and test PDFs are generated from it. Never
put a real resume in a fixture.

What matters: a resume is parsed once per distinct text (an identical
re-upload costs nothing, PDF or text), an offline placeholder never masks a
later real parse, a PDF without a text layer is rejected rather than stored
empty, and contact details have nowhere to land.
"""

import io
import json

import anthropic
import httpx2
import pypdf
import pytest
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from conftest import make_response
from src import llm
from src import resume as resume_module
from src.models import Base, Resume
from src.runner import StepFailed

FAKE_RESUME = """\
Jordan Avery | jordan.avery@example.com | (555) 010-0199

Backend Engineer, Brightline Logistics, 2021 - present
- Built order-routing services in Python and FastAPI on AWS
- Ran PostgreSQL migrations and Redis caching for 2M requests/day

Software Engineer, Harbor Analytics, 2018 - 2021
- Wrote ETL jobs in Python, deployed with Docker and Terraform

B.S. Computer Science, State University
"""

FAKE_PROFILE = {
    "skills": ["Python", "FastAPI", "AWS", "PostgreSQL", "Redis", "Docker", "Terraform"],
    "years_experience": 7,
    "titles": ["Backend Engineer", "Software Engineer"],
    "education": ["B.S. Computer Science"],
}

REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def profile_reply(profile=FAKE_PROFILE, model=None):
    return make_response(json.dumps(profile), model=model, usage=(900, 120))


@pytest.fixture
def session():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    yield s
    s.close()


# ------------------------------------------------------------- parsing


def test_upload_parses_and_stores_the_profile(session, fake_anthropic):
    fake_anthropic.reply = profile_reply()

    resume, parsed = resume_module.upload(session, FAKE_RESUME)

    assert parsed is True
    assert resume.profile == FAKE_PROFILE
    assert resume.model == llm.MODEL
    assert resume.cost_usd == pytest.approx((900 * 4 + 120 * 20) / 1e6)
    assert resume_module.current(session).id == resume.id

    call = fake_anthropic.calls[0]
    assert call["system"] == resume_module.RESUME_SYSTEM
    assert call["output_config"]["format"]["schema"] == resume_module.PROFILE_SCHEMA


def test_identical_reupload_is_not_parsed_or_billed_again(session, fake_anthropic):
    fake_anthropic.reply = profile_reply()
    first, _ = resume_module.upload(session, FAKE_RESUME)

    # same content, different whitespace: still the same resume
    again, parsed = resume_module.upload(session, "  " + FAKE_RESUME.replace("\n", "\r\n"))

    assert parsed is False
    assert again.id == first.id
    assert len(fake_anthropic.calls) == 1
    assert session.query(Resume).count() == 1


def test_changed_resume_is_parsed_and_becomes_current(session, fake_anthropic):
    fake_anthropic.reply = profile_reply()
    first, _ = resume_module.upload(session, FAKE_RESUME)

    updated = {**FAKE_PROFILE, "skills": FAKE_PROFILE["skills"] + ["Kubernetes"]}
    fake_anthropic.reply = profile_reply(updated)
    second, parsed = resume_module.upload(session, FAKE_RESUME + "\n- Learned Kubernetes")

    assert parsed is True
    assert second.id != first.id
    assert resume_module.current(session).profile["skills"][-1] == "Kubernetes"
    assert session.query(Resume).count() == 2   # history kept


def test_offline_placeholder_does_not_mask_a_later_real_parse(
    session, fake_anthropic, monkeypatch
):
    """Upload with no key, add a key, re-upload the same text: must parse."""
    monkeypatch.setattr(llm, "is_offline", lambda: True)
    placeholder, _ = resume_module.upload(session, FAKE_RESUME)
    assert placeholder.model == resume_module.OFFLINE
    assert placeholder.profile["skills"] == []
    assert fake_anthropic.calls == []

    # still offline: re-uploading reuses the placeholder, no duplicate row
    same, parsed = resume_module.upload(session, FAKE_RESUME)
    assert parsed is False
    assert same.id == placeholder.id
    assert session.query(Resume).count() == 1

    monkeypatch.setattr(llm, "is_offline", lambda: False)
    fake_anthropic.reply = profile_reply()
    real, parsed = resume_module.upload(session, FAKE_RESUME)

    assert parsed is True
    assert real.profile == FAKE_PROFILE
    assert len(fake_anthropic.calls) == 1


def test_profile_schema_has_no_place_for_contact_details():
    schema = resume_module.PROFILE_SCHEMA
    assert set(schema["properties"]) == {"skills", "years_experience", "titles", "education"}
    assert schema["additionalProperties"] is False


def test_transient_error_stores_nothing(session, fake_anthropic):
    fake_anthropic.raises = [
        anthropic.RateLimitError(
            "slow down", response=httpx2.Response(429, request=REQUEST), body=None
        )
    ]
    with pytest.raises(StepFailed):
        resume_module.upload(session, FAKE_RESUME)
    assert session.query(Resume).count() == 0


# ----------------------------------------------------------------- PDFs


def make_pdf(text: str) -> bytes:
    """A real PDF with `text` in its text layer, one line per line."""
    writer = pypdf.PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})
    })
    ops = ["BT", "/F1 10 Tf", "50 750 Td", "13 TL"]
    for line in text.splitlines():
        escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        ops.append(f"({escaped}) Tj T*")
    ops.append("ET")
    stream = DecodedStreamObject()
    stream.set_data("\n".join(ops).encode("latin-1"))
    page[NameObject("/Contents")] = writer._add_object(stream)
    return _write(writer)


def blank_pdf() -> bytes:
    """Stands in for a scanned resume: pages, but no text layer."""
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=612, height=792)
    return _write(writer)


def _write(writer) -> bytes:
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def test_pdf_text_is_extracted():
    text = resume_module.extract_pdf_text(make_pdf(FAKE_RESUME))
    assert "Backend Engineer, Brightline Logistics, 2021 - present" in text
    assert "PostgreSQL" in text


def test_pdf_with_same_content_hashes_like_the_text():
    """Re-exporting or pasting the same resume must not trigger a new parse."""
    text = resume_module.extract_pdf_text(make_pdf(FAKE_RESUME))
    assert resume_module.content_hash(text) == resume_module.content_hash(FAKE_RESUME)


@pytest.mark.parametrize(
    "data,reason",
    [
        (b"PK\x03\x04 this is a docx", "not a PDF"),
        (blank_pdf(), "scanned"),
        (b"%PDF-1.7\ngarbage that is not a real pdf body", "could not be read"),
        (b"%PDF-" + b"0" * (resume_module.MAX_PDF_BYTES + 1), "over 5 MB"),
    ],
    ids=["not_pdf", "scanned", "corrupt", "too_big"],
)
def test_unusable_pdfs_are_rejected_with_a_reason(data, reason):
    with pytest.raises(resume_module.UnreadableResume, match=reason):
        resume_module.extract_pdf_text(data)


def test_password_protected_pdf_is_rejected():
    reader = pypdf.PdfReader(io.BytesIO(make_pdf(FAKE_RESUME)))
    writer = pypdf.PdfWriter(clone_from=reader)
    writer.encrypt("secret")
    with pytest.raises(resume_module.UnreadableResume, match="password"):
        resume_module.extract_pdf_text(_write(writer))


# --------------------------------------------------------------- reparse


def test_reparse_ignores_the_cache_and_keeps_history(session, fake_anthropic):
    fake_anthropic.reply = profile_reply()
    first, _ = resume_module.upload(session, FAKE_RESUME)

    again = resume_module.reparse(session)

    assert again.id != first.id
    assert again.content_hash == first.content_hash
    assert len(fake_anthropic.calls) == 2          # asked for, so billed
    assert resume_module.current(session).id == again.id
    assert session.query(Resume).count() == 2


def test_reparse_with_nothing_uploaded_returns_none(session, fake_anthropic):
    assert resume_module.reparse(session) is None
    assert fake_anthropic.calls == []


# ---------------------------------------------------------------- HTTP


# client fixture: conftest.py


def put_text(client, text):
    return client.put("/resume", data={"text": text})


def put_pdf(client, data, name="resume.pdf"):
    return client.put("/resume", files={"file": (name, data, "application/pdf")})


def test_put_resume_returns_profile_without_raw_text(client, fake_anthropic):
    fake_anthropic.reply = profile_reply()
    r = put_text(client, FAKE_RESUME)

    assert r.status_code == 200
    body = r.json()
    assert body["profile"] == FAKE_PROFILE
    assert body["parsed"] is True
    assert body["chars"] == len(FAKE_RESUME)
    assert "raw_text" not in body
    assert "jordan.avery@example.com" not in r.text


def test_put_pdf_resume_parses_its_text(client, fake_anthropic):
    fake_anthropic.reply = profile_reply()
    r = put_pdf(client, make_pdf(FAKE_RESUME))

    assert r.status_code == 200
    assert r.json()["profile"] == FAKE_PROFILE
    sent = fake_anthropic.calls[0]["messages"][0]["content"]
    assert "Brightline Logistics" in sent


def test_pdf_then_same_text_is_a_free_cache_hit(client, fake_anthropic):
    fake_anthropic.reply = profile_reply()
    put_pdf(client, make_pdf(FAKE_RESUME))

    r = put_text(client, FAKE_RESUME)
    assert r.json()["parsed"] is False
    assert len(fake_anthropic.calls) == 1


def test_scanned_pdf_is_rejected_and_nothing_stored(client, fake_anthropic):
    r = put_pdf(client, blank_pdf())

    assert r.status_code == 422
    assert "scanned" in r.json()["detail"]
    assert fake_anthropic.calls == []
    assert client.get("/resume").status_code == 404


@pytest.mark.parametrize("payload", ["neither", "both"])
def test_put_needs_exactly_one_of_file_or_text(client, fake_anthropic, payload):
    if payload == "neither":
        r = client.put("/resume")
    else:
        r = client.put(
            "/resume",
            data={"text": FAKE_RESUME},
            files={"file": ("r.pdf", make_pdf(FAKE_RESUME), "application/pdf")},
        )
    assert r.status_code == 422
    assert "exactly one" in r.json()["detail"]
    assert fake_anthropic.calls == []


def test_get_resume_404_until_uploaded(client, fake_anthropic):
    assert client.get("/resume").status_code == 404

    fake_anthropic.reply = profile_reply()
    put_text(client, FAKE_RESUME)

    r = client.get("/resume")
    assert r.status_code == 200
    assert r.json()["profile"] == FAKE_PROFILE


def test_put_blank_resume_is_rejected(client, fake_anthropic):
    r = put_text(client, "   \n ")
    assert r.status_code == 422
    assert "empty" in r.json()["detail"]
    assert fake_anthropic.calls == []


def test_transient_api_error_returns_503(client, fake_anthropic):
    fake_anthropic.raises = [anthropic.APITimeoutError(request=REQUEST)]
    r = put_text(client, FAKE_RESUME)

    assert r.status_code == 503
    assert client.get("/resume").status_code == 404


def test_refusal_returns_422(client, fake_anthropic):
    fake_anthropic.reply = make_response("", stop_reason="refusal")
    r = put_text(client, FAKE_RESUME)
    assert r.status_code == 422
    assert "refused" in r.json()["detail"]


def test_insights_total_includes_resume_spend(client, fake_anthropic):
    fake_anthropic.reply = profile_reply()
    put_text(client, FAKE_RESUME)

    total = client.get("/insights").json()["total_cost_usd"]
    assert total == pytest.approx(round((900 * 4 + 120 * 20) / 1e6, 4))


def test_reparse_endpoint_turns_offline_placeholder_into_real_parse(
    client, fake_anthropic, monkeypatch
):
    """The path the user will actually take: upload with no key, add one,
    reparse without uploading again."""
    monkeypatch.setattr(llm, "is_offline", lambda: True)
    assert put_text(client, FAKE_RESUME).json()["model"] == resume_module.OFFLINE
    assert client.post("/resume/reparse").status_code == 409   # still no key

    monkeypatch.setattr(llm, "is_offline", lambda: False)
    fake_anthropic.reply = profile_reply()
    r = client.post("/resume/reparse")

    assert r.status_code == 200
    assert r.json()["profile"] == FAKE_PROFILE
    assert client.get("/resume").json()["model"] == llm.MODEL


def test_reparse_endpoint_404_without_a_resume(client, fake_anthropic):
    assert client.post("/resume/reparse").status_code == 404
