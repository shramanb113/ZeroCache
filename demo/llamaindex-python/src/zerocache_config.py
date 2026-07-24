
"""Shared config for every script in this demo.

Mirrors the ZEROCACHE_DEMO_KEY env var convention already used by
demo/mastra/test-embed.mjs and demo/langchain-ts, so the same exported key
works across every demo app in this repo.
"""

import os

ZEROCACHE_HOST = os.environ.get("ZEROCACHE_HOST", "http://127.0.0.1:8080")
PROVIDER = "gemini"
# NOT "text-embedding-004" -- that's the model zerocache-adapters-gemini's
# own test fixtures use, but it (and embedding-001) were deprecated and
# removed from Google's real API around early 2026, returning a 404 at
# generativelanguage.googleapis.com. Confirmed live: gemini-embedding-001
# is the current, supported text-embedding model. Not a Zerocache bug --
# the fixture's model string was simply never revalidated against Google's
# live catalog, and neither was mine the first time.
MODEL = "gemini-embedding-001"
API_KEY = os.environ.get("ZEROCACHE_DEMO_KEY")

# The openai Python SDK (which llama-index-embeddings-openai wraps) appends
# the endpoint name itself -- client.post("/embeddings", ...) -- onto
# whatever base_url it's given, identical to the Node SDK's behavior
# (confirmed against openai/_client.py and openai/resources/embeddings.py).
# Zerocache's contract is POST /{provider}/v1/embeddings, so base_url has to
# be host + "/gemini/v1" for that "/embeddings" append to land correctly.
ZEROCACHE_GEMINI_BASE_URL = f"{ZEROCACHE_HOST}/{PROVIDER}/v1"


def require_api_key() -> str:
    if not API_KEY:
        raise RuntimeError(
            "ZEROCACHE_DEMO_KEY is not set. Export a real Gemini API key as "
            "ZEROCACHE_DEMO_KEY before running this script -- Zerocache is "
            "bring-your-own-key, it holds no provider credentials of its own."
        )
    return API_KEY
