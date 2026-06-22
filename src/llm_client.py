"""Shared LLM client with retry logic."""

import json
import logging
import os
import time
from openai import OpenAI, APIError, APITimeoutError, RateLimitError

log = logging.getLogger(__name__)

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
    return _client


def get_model() -> str:
    return os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")


def chat_json(messages: list[dict], schema: dict | None = None, max_retries: int = 3) -> dict:
    """Call the LLM and parse JSON response, with retry on transient failures."""
    client = get_client()
    model = get_model()

    kwargs = dict(model=model, messages=messages, temperature=0.0)

    if schema:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "response", "strict": True, "schema": schema}
        }
    else:
        kwargs["response_format"] = {"type": "json_object"}

    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(**kwargs)
            if not response.choices:
                raise ValueError("LLM returned no choices (possible refusal)")
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from LLM")
            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")
            return parsed
        except (APITimeoutError, RateLimitError, APIError, ValueError) as e:
            last_error = e
            wait = 2 ** attempt
            log.warning("LLM call failed (attempt %d/%d): %s — retrying in %ds", attempt + 1, max_retries, e, wait)
            time.sleep(wait)
        except json.JSONDecodeError as e:
            if schema and attempt == 0:
                log.warning("Strict JSON schema failed, falling back to json_object mode")
                kwargs["response_format"] = {"type": "json_object"}
                continue
            last_error = e
            log.error("LLM returned invalid JSON: %s", e)
            raise

    raise RuntimeError(f"LLM call failed after {max_retries} retries: {last_error}")


def chat_text(messages: list[dict], max_retries: int = 3) -> str:
    """Call the LLM and return plain text response, with retry."""
    client = get_client()
    model = get_model()

    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, temperature=0.0
            )
            if not response.choices:
                raise ValueError("LLM returned no choices (possible refusal)")
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from LLM")
            return content
        except (APITimeoutError, RateLimitError, APIError, ValueError) as e:
            last_error = e
            wait = 2 ** attempt
            log.warning("LLM call failed (attempt %d/%d): %s — retrying in %ds", attempt + 1, max_retries, e, wait)
            time.sleep(wait)

    raise RuntimeError(f"LLM text call failed after {max_retries} retries: {last_error}")


def get_embedding_model() -> str:
    return os.environ.get("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small")


def embed(texts: list[str], max_retries: int = 3) -> list[list[float]]:
    """Embed a batch of texts. Returns list of embedding vectors."""
    if not texts:
        return []

    client = get_client()
    model = get_embedding_model()

    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(model=model, input=texts)
            return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
        except (APITimeoutError, RateLimitError, APIError) as e:
            last_error = e
            wait = 2 ** attempt
            log.warning("Embedding call failed (attempt %d/%d): %s — retrying in %ds", attempt + 1, max_retries, e, wait)
            time.sleep(wait)

    raise RuntimeError(f"Embedding call failed after {max_retries} retries: {last_error}")


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
