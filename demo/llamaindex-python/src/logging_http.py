"""httpx event-hook equivalent of demo/langchain-ts's logging-fetch.ts.

OpenAIEmbedding (llama-index-embeddings-openai) accepts a custom
http_client: httpx.Client -- confirmed against the installed package's
_get_credential_kwargs (zerocache-http/src/... no, llama_index/embeddings/
openai/base.py). Wiring a response event hook is the only way to see
Zerocache's X-Zerocache-Hits / X-Zerocache-Misses headers from inside a
LlamaIndex app; the Embedding interface itself never surfaces response
metadata.
"""

import time

import httpx


def make_logging_http_client(label: str) -> httpx.Client:
    start_times: dict[int, float] = {}

    def on_request(request: httpx.Request) -> None:
        start_times[id(request)] = time.monotonic()

    def on_response(response: httpx.Response) -> None:
        start = start_times.pop(id(response.request), None)
        elapsed_ms = (time.monotonic() - start) * 1000 if start is not None else -1
        hits = response.headers.get("x-zerocache-hits")
        misses = response.headers.get("x-zerocache-misses")
        suffix = f" -- hits={hits} misses={misses}" if hits is not None or misses is not None else ""
        print(f"[{label}] {response.status_code} in {elapsed_ms:.0f}ms{suffix}")

    return httpx.Client(event_hooks={"request": [on_request], "response": [on_response]})
