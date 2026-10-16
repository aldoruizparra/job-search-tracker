"""Skill-name normalization. Deterministic, no LLM.

Postings and resumes name the same skill differently ("k8s", "Kubernetes",
"Postgres", "PostgreSQL"). Every comparison and every count across postings
goes through canonical() so those collapse into one skill.

Raw LLM outputs are stored as returned and normalized when used, never
rewritten. Improving these tables therefore costs nothing to apply: no
posting or resume is re-parsed. VERSION is a hash of the tables, recorded on
each comparison, so POST /jobs/recompare can find comparisons made under an
older table.

Two tables:
  ALIASES  canonical name -> other spellings. Matching ignores case,
           whitespace, hyphens, underscores, and dots, so "Node.js",
           "NodeJS", and "node js" need no separate entries. "#" and "+"
           are significant: C, C++, and C# stay distinct.
  PARENTS  skill -> the broader skill it implies. Having AWS Lambda means
           having AWS, so a posting asking for AWS is met. The reverse is
           only a partial match: having AWS does not mean having Lambda.

To add a skill: put it in ALIASES (with an empty list if it has no other
spellings but should have a fixed display form), and in PARENTS if it
implies a broader one. The import-time checks below reject ambiguity.
"""

import hashlib
import json
import re

ALIASES = {
    # languages
    "Python": ["py", "python3"],
    "JavaScript": ["js", "ecmascript"],
    "TypeScript": ["ts"],
    "Go": ["golang"],
    "C++": ["cpp"],
    "C#": ["csharp", "c sharp"],
    "Java": [],
    "Kotlin": [],
    "Rust": [],
    "Ruby": [],
    "Swift": [],
    "SQL": [],
    "Bash": ["shell", "shell scripting"],
    "HTML": ["html5"],
    "CSS": ["css3"],
    # frameworks and libraries
    "Node.js": ["node"],
    "Express": ["express.js"],
    "React": ["react.js"],
    "Next.js": [],
    "Vue.js": ["vue"],
    "Django": [],
    "Flask": [],
    "FastAPI": [],
    "Spring Boot": ["spring"],
    ".NET": ["dotnet", ".net core"],
    "Ruby on Rails": ["rails", "ror"],
    "PyTorch": ["torch"],
    "TensorFlow": [],
    "scikit-learn": ["sklearn"],
    "pandas": [],
    "NumPy": [],
    # data stores and data platforms
    "PostgreSQL": ["postgres", "postgre", "psql"],
    "MySQL": [],
    "SQLite": [],
    "MongoDB": ["mongo"],
    "Redis": [],
    "Elasticsearch": [],
    "Apache Kafka": ["kafka"],
    "Apache Spark": ["spark", "pyspark"],
    "Apache Airflow": ["airflow"],
    "Snowflake": [],
    "BigQuery": ["google bigquery"],
    "dbt": [],
    # cloud
    "AWS": ["amazon web services"],
    "GCP": ["google cloud", "google cloud platform"],
    "Azure": ["microsoft azure"],
    "AWS Lambda": ["lambda"],
    "Amazon EC2": ["ec2", "aws ec2"],
    "Amazon S3": ["s3", "aws s3"],
    "Amazon DynamoDB": ["dynamodb", "dynamo", "aws dynamodb"],
    "Amazon CloudWatch": ["cloudwatch", "aws cloudwatch"],
    "AWS CloudFormation": ["cloudformation"],
    "AWS CDK": ["cdk"],
    "AWS Glue": ["glue"],
    "Amazon Athena": ["athena", "aws athena"],
    "AWS Step Functions": ["step functions"],
    "Amazon EventBridge": ["eventbridge", "aws eventbridge"],
    "AWS CloudTrail": ["cloudtrail"],
    "Amazon Bedrock": ["bedrock", "aws bedrock"],
    # infrastructure and practices
    "Kubernetes": ["k8s", "kube"],
    "Helm": [],
    "Docker": [],
    "Terraform": ["hashicorp terraform"],
    "Ansible": [],
    "CI/CD": ["cicd", "ci cd"],
    "GitHub Actions": ["gh actions"],
    "Jenkins": [],
    "Git": [],
    "Linux": [],
    "REST APIs": ["rest", "rest api", "restful", "restful apis", "restful api"],
    "GraphQL": [],
    "gRPC": [],
    "Microservices": ["microservice architecture"],
    # machine learning
    "Machine Learning": ["ml"],
    "NLP": ["natural language processing"],
    "LLMs": ["llm", "large language models", "large language model"],
    "RAG": ["retrieval augmented generation", "rag knowledge bases"],
    "MCP": ["mcp servers", "model context protocol"],
    # other tools
    "Unity": ["unity3d"],
    "Power BI": ["microsoft power bi"],
}

PARENTS = {
    "TypeScript": "JavaScript",
    "Node.js": "JavaScript",
    "Express": "Node.js",
    "React": "JavaScript",
    "Next.js": "React",
    "Vue.js": "JavaScript",
    "Django": "Python",
    "Flask": "Python",
    "FastAPI": "Python",
    "Spring Boot": "Java",
    "Ruby on Rails": "Ruby",
    "PostgreSQL": "SQL",
    "MySQL": "SQL",
    "SQLite": "SQL",
    "Helm": "Kubernetes",
    "AWS Lambda": "AWS",
    "Amazon EC2": "AWS",
    "Amazon S3": "AWS",
    "Amazon DynamoDB": "AWS",
    "Amazon CloudWatch": "AWS",
    "AWS CloudFormation": "AWS",
    "AWS CDK": "AWS",
    "AWS Glue": "AWS",
    "Amazon Athena": "AWS",
    "AWS Step Functions": "AWS",
    "Amazon EventBridge": "AWS",
    "AWS CloudTrail": "AWS",
    "Amazon Bedrock": "AWS",
}


def key(skill: str) -> str:
    """Lookup key: ignores case, whitespace, hyphens, underscores, dots."""
    return re.sub(r"[\s\-_.]+", "", skill.casefold())


def _build_lookup(aliases: dict) -> dict:
    lookup = {}
    for canon, others in aliases.items():
        for name in [canon, *others]:
            k = key(name)
            if k in lookup and lookup[k] != canon:
                raise ValueError(f"{name!r} would mean both {lookup[k]!r} and {canon!r}")
            lookup[k] = canon
    return lookup


def _check_parents(parents: dict, aliases: dict) -> None:
    for child, parent in parents.items():
        for name in (child, parent):
            if name not in aliases:
                raise ValueError(f"PARENTS names {name!r}, which is not in ALIASES")
        seen, node = {child}, parent
        while node in parents:
            node = parents[node]
            if node in seen:
                raise ValueError(f"PARENTS has a cycle through {child!r}")
            seen.add(node)


def _version(aliases: dict, parents: dict) -> str:
    blob = json.dumps([aliases, parents], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


_LOOKUP = _build_lookup(ALIASES)
_check_parents(PARENTS, ALIASES)
VERSION = _version(ALIASES, PARENTS)


def canonical(skill: str) -> str:
    """Canonical display name. Unknown skills pass through, tidied."""
    tidy = " ".join(skill.split())
    return _LOOKUP.get(key(tidy), tidy)


def _split_combined(skill: str) -> list:
    """"HTML/CSS" -> ["HTML", "CSS"], but only when the whole name is unknown
    and every part is known. "CI/CD" is itself known, so it stays whole."""
    if key(skill) in _LOOKUP or "/" not in skill:
        return [skill]
    parts = [p for p in skill.split("/") if p.strip()]
    if len(parts) > 1 and all(key(p) in _LOOKUP for p in parts):
        return parts
    return [skill]


def canonical_list(skills: list) -> list:
    """Canonical names in first-seen order, duplicates removed by key."""
    out, seen = [], set()
    for raw in skills:
        for s in _split_combined(raw):
            c = canonical(s)
            if key(c) not in seen:
                seen.add(key(c))
                out.append(c)
    return out


def ancestors(skill: str) -> list:
    """Broader skills `skill` implies, nearest first. Takes a canonical name."""
    out, node = [], skill
    while node in PARENTS:
        node = PARENTS[node]
        out.append(node)
    return out


def match(wanted: list, have: list) -> dict:
    """Classify each wanted skill against the skills a resume shows.

    met      the resume has it, or has something narrower that implies it
             (asks AWS, resume has AWS Lambda)
    partial  the resume has only something broader (asks AWS Lambda, resume
             has AWS). Reported, but not counted as met.
    missing  neither
    All names are canonical.
    """
    owned = canonical_list(have)
    owned_keys = {key(s) for s in owned}
    implied = set(owned_keys)
    for s in owned:
        implied.update(key(a) for a in ancestors(s))

    result = {"met": [], "partial": [], "missing": []}
    for skill in canonical_list(wanted):
        if key(skill) in implied:
            result["met"].append(skill)
            continue
        broader = next((a for a in ancestors(skill) if key(a) in owned_keys), None)
        if broader:
            result["partial"].append({"skill": skill, "have": broader})
        else:
            result["missing"].append(skill)
    return result
