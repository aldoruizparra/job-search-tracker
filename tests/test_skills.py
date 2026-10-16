"""Skill-name normalization.

The tables in src/skills.py are data, so most of these tests guard their
integrity: an ambiguous alias or a parent cycle must fail loudly at import,
not silently produce wrong matches.
"""

import pytest

from src import skills


# ------------------------------------------------------------ canonical


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("k8s", "Kubernetes"),
        ("K8S", "Kubernetes"),
        ("kubernetes", "Kubernetes"),
        ("Postgres", "PostgreSQL"),
        ("golang", "Go"),
        ("NodeJS", "Node.js"),
        ("node js", "Node.js"),
        ("node-js", "Node.js"),
        ("dotnet", ".NET"),
        ("REST", "REST APIs"),
        ("restful apis", "REST APIs"),
        ("aws lambda", "AWS Lambda"),
        ("Amazon Web Services", "AWS"),
    ],
)
def test_spellings_collapse_to_one_canonical_name(raw, expected):
    assert skills.canonical(raw) == expected


def test_c_family_stays_distinct():
    """# and + are significant; stripping them would merge three languages."""
    assert {skills.canonical(x) for x in ["C", "C++", "C#"]} == {"C", "C++", "C#"}
    assert skills.canonical("csharp") == "C#"
    assert skills.canonical("cpp") == "C++"


def test_unknown_skill_passes_through_tidied():
    assert skills.canonical("  Some   Niche Tool ") == "Some Niche Tool"


def test_canonical_list_dedupes_and_keeps_order():
    assert skills.canonical_list(["k8s", "Python", "Kubernetes", "py"]) == ["Kubernetes", "Python"]


def test_combined_names_split_only_when_every_part_is_known():
    assert skills.canonical_list(["HTML/CSS"]) == ["HTML", "CSS"]
    assert skills.canonical_list(["CI/CD"]) == ["CI/CD"]              # known as a whole
    assert skills.canonical_list(["HTML/Frobnicate"]) == ["HTML/Frobnicate"]


# ---------------------------------------------------------------- match


def test_narrower_skill_on_resume_meets_broader_requirement():
    result = skills.match(["AWS", "SQL", "JavaScript"], ["AWS Lambda", "Postgres", "Next.js"])
    assert result["met"] == ["AWS", "SQL", "JavaScript"]   # Next.js -> React -> JavaScript
    assert result["missing"] == []


def test_broader_skill_on_resume_is_only_partial():
    result = skills.match(["Amazon S3"], ["AWS"])
    assert result["met"] == []
    assert result["partial"] == [{"skill": "Amazon S3", "have": "AWS"}]
    assert result["missing"] == []


def test_match_reports_canonical_names():
    result = skills.match(["k8s", "golang"], ["Kubernetes"])
    assert result == {"met": ["Kubernetes"], "partial": [], "missing": ["Go"]}


# ------------------------------------------------------ table integrity


def test_ambiguous_alias_is_rejected():
    with pytest.raises(ValueError, match="would mean both"):
        skills._build_lookup({"Kubernetes": ["kube"], "Kubeflow": ["kube"]})


def test_parent_must_be_a_known_skill():
    with pytest.raises(ValueError, match="not in ALIASES"):
        skills._check_parents({"Django": "Pyton"}, {"Django": [], "Python": []})


def test_parent_cycle_is_rejected():
    with pytest.raises(ValueError, match="cycle"):
        skills._check_parents({"A": "B", "B": "A"}, {"A": [], "B": []})


def test_shipped_tables_pass_their_own_checks():
    skills._build_lookup(skills.ALIASES)
    skills._check_parents(skills.PARENTS, skills.ALIASES)


def test_version_changes_when_the_tables_change():
    before = skills._version(skills.ALIASES, skills.PARENTS)
    changed = {**skills.ALIASES, "Zig": []}
    assert skills._version(changed, skills.PARENTS) != before
    assert skills._version(skills.ALIASES, skills.PARENTS) == before   # stable
