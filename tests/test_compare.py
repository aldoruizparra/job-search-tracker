"""compare and score, and re-comparing jobs after the resume changes.

The postings and resume profiles here are fabricated. The default fake
extract output (conftest.FAKE_REQUIREMENTS) asks for Python and AWS, prefers
Kubernetes, and wants 3+ years.
"""

import json

from conftest import FAKE_REQUIREMENTS, make_response
from src import llm, skills
from src.models import Job

PROFILE = {
    "skills": ["python", "Docker", "Kubernetes"],   # lowercase on purpose
    "years_experience": 2,
    "titles": ["Software Engineer"],
    "education": ["B.S. Computer Science"],
}


def upload_resume(client, fake_anthropic, profile=PROFILE, text="Fabricated resume text"):
    """Upload a resume whose parse returns `profile`, then restore the
    default fake reply so later extract calls get posting requirements."""
    fake_anthropic.reply = make_response(json.dumps(profile))
    r = client.put("/resume", data={"text": text})
    fake_anthropic.reply = None
    assert r.status_code == 200
    return r.json()


def post_job(client, text="Backend Engineer at Example Corp"):
    r = client.post("/jobs", json={"raw_text": text})
    assert r.status_code == 201
    return r.json()


def step(job_body, name):
    return next(s for s in job_body["steps"] if s["step_name"] == name)


# ---------------------------------------------------------- no resume


def test_job_without_resume_completes_with_priorities_and_no_score(client, fake_anthropic):
    body = post_job(client)

    assert body["status"] == "complete"
    assert body["fit_score"] is None               # not measurable, not 0
    analysis = body["fit_analysis"]
    assert analysis["mode"] == "no_resume"
    assert analysis["resume_id"] is None
    assert analysis["missing"] == []               # nothing to count as a gap
    assert analysis["priorities"] == {
        "required": ["Python", "AWS"],
        "preferred": ["Kubernetes"],
        "years_experience": 3,
    }


def test_no_resume_reasoning_summarizes_what_the_posting_wants(client, fake_anthropic):
    s = client.session_factory()
    body = post_job(client)
    job = s.get(Job, body["id"])
    score_out = next(st for st in job.steps if st.step_name == "score").output_data
    s.close()

    assert "must have Python, AWS" in score_out["reasoning"]
    assert "nice to have Kubernetes" in score_out["reasoning"]
    assert "3+ years" in score_out["reasoning"]


def test_offline_placeholder_resume_counts_as_no_resume(client, fake_anthropic, monkeypatch):
    """An empty placeholder would otherwise mark every skill as missing."""
    monkeypatch.setattr(llm, "is_offline", lambda: True)
    client.put("/resume", data={"text": "Fabricated resume text"})
    monkeypatch.setattr(llm, "is_offline", lambda: False)

    analysis = post_job(client)["fit_analysis"]
    assert analysis["mode"] == "no_resume"
    assert analysis["missing"] == []


# -------------------------------------------------------- with resume


def test_compare_matches_skills_case_insensitively(client, fake_anthropic):
    resume = upload_resume(client, fake_anthropic)
    body = post_job(client)

    analysis = body["fit_analysis"]
    assert analysis["mode"] == "resume"
    assert analysis["resume_id"] == resume["id"]
    assert analysis["met_required"] == ["Python"]    # posting's spelling kept
    assert analysis["missing"] == ["AWS"]
    assert analysis["met_preferred"] == ["Kubernetes"]
    assert analysis["missing_preferred"] == []
    assert analysis["met"] == ["Python", "Kubernetes"]
    assert body["fit_score"] == 0.5                  # 1 of 2 required


def test_score_reasoning_explains_the_number(client, fake_anthropic):
    upload_resume(client, fake_anthropic)
    body = post_job(client)

    s = client.session_factory()
    job = s.get(Job, body["id"])
    reasoning = next(st for st in job.steps if st.step_name == "score").output_data["reasoning"]
    s.close()

    assert "Matches 1 of 2 required skills" in reasoning
    assert "missing AWS" in reasoning
    assert "1 of 1 preferred" in reasoning
    assert "asks 3+ years, resume shows 2" in reasoning


def test_posting_without_required_skills_has_no_score(client, fake_anthropic):
    upload_resume(client, fake_anthropic)
    fake_anthropic.reply = make_response(json.dumps({**FAKE_REQUIREMENTS, "required_skills": []}))

    body = post_job(client)
    assert body["fit_analysis"]["mode"] == "resume"
    assert body["fit_score"] is None


# ---------------------------------------------------------- recompare


def test_recompare_updates_old_jobs_without_rebilling_extract(client, fake_anthropic):
    job = post_job(client)                           # no resume yet
    assert job["fit_score"] is None
    extract_calls = len(fake_anthropic.calls)

    upload_resume(client, fake_anthropic)
    calls_after_upload = len(fake_anthropic.calls)
    assert calls_after_upload == extract_calls + 1  # just the resume parse

    r = client.post("/jobs/recompare")
    assert r.status_code == 200
    assert r.json()["recompared"] == [job["id"]]
    assert r.json()["already_current"] == 0

    after = client.get(f"/jobs/{job['id']}").json()
    assert after["status"] == "complete"
    assert after["fit_score"] == 0.5
    assert after["fit_analysis"]["mode"] == "resume"
    assert len(fake_anthropic.calls) == calls_after_upload   # extract not called
    assert step(after, "extract")["attempts"] == 1
    assert step(after, "ingest")["attempts"] == 1


def test_recompare_twice_does_nothing_the_second_time(client, fake_anthropic):
    post_job(client)
    upload_resume(client, fake_anthropic)
    client.post("/jobs/recompare")

    r = client.post("/jobs/recompare").json()
    assert r["recompared"] == []
    assert r["already_current"] == 1


def test_new_resume_makes_jobs_stale_again(client, fake_anthropic):
    upload_resume(client, fake_anthropic)
    job = post_job(client)
    assert job["fit_score"] == 0.5

    upload_resume(
        client, fake_anthropic,
        profile={**PROFILE, "skills": PROFILE["skills"] + ["AWS"]},
        text="Fabricated resume text, now with AWS",
    )
    r = client.post("/jobs/recompare").json()

    assert r["recompared"] == [job["id"]]
    assert client.get(f"/jobs/{job['id']}").json()["fit_score"] == 1.0


def test_recompare_needs_a_parsed_resume(client, fake_anthropic):
    post_job(client)
    r = client.post("/jobs/recompare")
    assert r.status_code == 409


def test_recompare_skips_dead_lettered_jobs(client, fake_anthropic):
    dead = client.post("/jobs", json={"source_url": "https://example.com/job"}).json()
    assert dead["status"] == "dead_letter"
    upload_resume(client, fake_anthropic)

    r = client.post("/jobs/recompare").json()
    assert r["recompared"] == []
    assert client.get(f"/jobs/{dead['id']}").json()["status"] == "dead_letter"


# ----------------------------------------------------------- insights


def test_insights_ignores_jobs_compared_without_a_resume(client, fake_anthropic):
    post_job(client)                                 # no_resume comparison
    upload_resume(client, fake_anthropic)
    post_job(client, text="Another fabricated posting")

    body = client.get("/insights").json()
    assert body["jobs_analyzed"] == 1
    assert body["top_gaps"] == [{"skill": "AWS", "count": 1}]


# ------------------------------------------------- skill normalization


def test_synonyms_match_and_are_reported_canonically(client, fake_anthropic):
    upload_resume(client, fake_anthropic, profile={**PROFILE, "skills": ["Kubernetes", "golang"]})
    fake_anthropic.reply = make_response(json.dumps(
        {**FAKE_REQUIREMENTS, "required_skills": ["k8s", "Go"], "preferred_skills": []}
    ))

    body = post_job(client)
    assert body["fit_analysis"]["met_required"] == ["Kubernetes", "Go"]
    assert body["fit_score"] == 1.0


def test_broader_resume_skill_is_partial_and_not_counted(client, fake_anthropic):
    upload_resume(client, fake_anthropic, profile={**PROFILE, "skills": ["AWS"]})
    fake_anthropic.reply = make_response(json.dumps(
        {**FAKE_REQUIREMENTS, "required_skills": ["AWS Lambda", "AWS"], "preferred_skills": []}
    ))

    body = post_job(client)
    analysis = body["fit_analysis"]
    assert analysis["met_required"] == ["AWS"]
    assert analysis["partial_required"] == [{"skill": "AWS Lambda", "have": "AWS"}]
    assert analysis["missing"] == []
    assert body["fit_score"] == 0.5

    s = client.session_factory()
    job = s.get(Job, body["id"])
    reasoning = next(st for st in job.steps if st.step_name == "score").output_data["reasoning"]
    s.close()
    assert "only broadly: AWS Lambda (resume shows AWS)" in reasoning


def test_insights_counts_spelling_variants_as_one_skill(client, fake_anthropic):
    post_job(client)
    post_job(client, text="Another fabricated posting")
    s = client.session_factory()
    for job, missing in zip(s.query(Job).all(), [["k8s", "Go"], ["Kubernetes"]]):
        job.fit_analysis = {"mode": "resume", "met": [], "missing": missing}
    s.commit()
    s.close()

    gaps = client.get("/insights").json()["top_gaps"]
    assert gaps[0] == {"skill": "Kubernetes", "count": 2}


def test_recompare_picks_up_a_changed_skills_table(client, fake_anthropic):
    """Editing skills.py must not need a new resume to take effect."""
    upload_resume(client, fake_anthropic)
    job = post_job(client)

    s = client.session_factory()
    row = s.get(Job, job["id"])
    row.fit_analysis = {**row.fit_analysis, "skills_version": "older-table"}
    s.commit()
    s.close()

    r = client.post("/jobs/recompare").json()
    assert r["recompared"] == [job["id"]]
    after = client.get(f"/jobs/{job['id']}").json()
    assert after["fit_analysis"]["skills_version"] == skills.VERSION
