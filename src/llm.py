"""Anthropic client and the one structured LLM call every feature goes through.

Kept separate from pipeline.py because not every LLM call is a pipeline step:
resume parsing happens once per resume, not once per job.

The failure mapping lives here, once:
  rate limit, timeout, connection error, 5xx  ->  StepFailed (retryable)
  any other API error (400, 401, 404, ...)    ->  propagates (a bug)
  refusal                                     ->  ModelRefused (propagates)
  truncated or unparseable output             ->  propagates (a bug)
"""

import json
import logging
import os
from functools import lru_cache

import anthropic

from .runner import StepFailed

log = logging.getLogger("llm")

MODEL = os.getenv("LLM_MODEL", "claude-opus-5-5")

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
    or anyone cloning the repo): LLM calls are skipped and callers return
    placeholder output at zero cost instead of failing.
    """
    return not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    in_rate, out_rate = PRICING[model]
    return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000


def nullable(kind: str) -> dict:
    """JSON schema for a field that is `kind` or null."""
    return {"anyOf": [{"type": kind}, {"type": "null"}]}


def structured_call(system: str, text: str, schema: dict) -> dict:
    """One Messages call constrained to `schema`.

    Returns the step handler shape plus the model that served the call:
    {"output", "input_tokens", "output_tokens", "cost_usd", "model"}.
    Never logs `text`: it may be a resume.
    """
    # Fail before paying for a call we could not price.
    if MODEL not in PRICING:
        raise ValueError(f"no pricing for model {MODEL}")

    try:
        response = get_client().beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={
                "effort": "low",
                "format": {"type": "json_schema", "schema": schema},
            },
            system=system,
            messages=[{"role": "user", "content": text}],
        )
    # APITimeoutError subclasses APIConnectionError, so this covers both.
    except (anthropic.RateLimitError, anthropic.APIConnectionError) as exc:
        raise StepFailed(f"{type(exc).__name__}: {exc}") from exc
    except anthropic.InternalServerError as exc:   # 5xx, including 529 overloaded
        raise StepFailed(f"server error {exc.status_code}: {exc}") from exc

    if response.stop_reason == "refusal":
        raise ModelRefused(f"model refused: {response.stop_details}")
    if response.stop_reason == "max_tokens":
        raise ValueError("output truncated at max_tokens")

    body = next(b.text for b in response.content if b.type == "text")
    usage = response.usage
    return {
        "output": json.loads(body),
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cost_usd": cost_usd(response.model, usage.input_tokens, usage.output_tokens),
        "model": response.model,
    }
