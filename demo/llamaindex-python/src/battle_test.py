"""Battle-tests Zerocache from a Python LlamaIndex consumer's perspective:
HTTP-contract edge cases first (no API key needed), then behavioral edge
cases driven through llama-index-embeddings-openai's own client (key
needed -- these make real, billed provider calls against Gemini).

Run with: uv run python src/battle_test.py
(set ZEROCACHE_DEMO_KEY first to also run the key-dependent section)
"""

import asyncio
import sys
import time
from dataclasses import dataclass, field
from typing import Literal

import httpx

# Windows' default console codepage (cp1252) can't encode the checkmark/
# cross/warning markers this harness prints -- reconfigure stdout to UTF-8
# rather than fall back to ASCII-only markers, so this still works
# unmodified on Linux/macOS terminals that already default to UTF-8.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
from llama_index.embeddings.openai_like import OpenAILikeEmbedding

from logging_http import make_logging_http_client
from zerocache_config import API_KEY, MODEL, PROVIDER, ZEROCACHE_GEMINI_BASE_URL, ZEROCACHE_HOST, require_api_key

Result = Literal["PASS", "FAIL", "FINDING", "SKIP"]


@dataclass
class Report:
    results: list[tuple[str, Result, str]] = field(default_factory=list)

    def record(self, name: str, result: Result, detail: str = "") -> None:
        self.results.append((name, result, detail))
        marker = {"PASS": "✓", "FAIL": "✗", "FINDING": "⚠", "SKIP": "-"}[result]
        suffix = f" -- {detail}" if detail else ""
        print(f"{marker} [{result}] {name}{suffix}")


report = Report()


async def run(name: str, fn) -> None:
    try:
        await fn()
    except Exception as e:  # noqa: BLE001 - battle-test harness, want to catch and report everything
        report.record(name, "FAIL", f"{type(e).__name__}: {e}")


# ---------- key-independent HTTP-contract tests ----------


async def test_health() -> None:
    async with httpx.AsyncClient() as client:
        res = await client.get(f"{ZEROCACHE_HOST}/health")
    report.record("GET /health returns 200", "PASS" if res.status_code == 200 else "FAIL", f"got {res.status_code}")


async def test_ready() -> None:
    async with httpx.AsyncClient() as client:
        res = await client.get(f"{ZEROCACHE_HOST}/ready")
    report.record(
        "GET /ready returns 200 against a live local store", "PASS" if res.status_code == 200 else "FAIL", f"got {res.status_code}"
    )


async def test_metrics_endpoint() -> None:
    async with httpx.AsyncClient() as client:
        res = await client.get(f"{ZEROCACHE_HOST}/metrics")
    looks_like_prometheus = "# HELP" in res.text or "zerocache_cache_hits_total" in res.text
    report.record(
        "GET /metrics returns Prometheus text format",
        "PASS" if res.status_code == 200 and looks_like_prometheus else "FAIL",
        f"status={res.status_code}",
    )


async def test_missing_auth() -> None:
    async with httpx.AsyncClient() as client:
        res = await client.post(f"{ZEROCACHE_HOST}/{PROVIDER}/v1/embeddings", json={"model": MODEL, "input": ["test"]})
    report.record("Missing Authorization header -> 401", "PASS" if res.status_code == 401 else "FAIL", f"got {res.status_code}")


async def test_malformed_auth() -> None:
    async with httpx.AsyncClient() as client:
        res = await client.post(
            f"{ZEROCACHE_HOST}/{PROVIDER}/v1/embeddings",
            json={"model": MODEL, "input": ["test"]},
            headers={"Authorization": "not-a-bearer-token"},
        )
    report.record(
        "Malformed Authorization header (no 'Bearer ' prefix) -> 401", "PASS" if res.status_code == 401 else "FAIL", f"got {res.status_code}"
    )


async def test_unknown_provider() -> None:
    async with httpx.AsyncClient() as client:
        res = await client.post(
            f"{ZEROCACHE_HOST}/not-a-real-provider/v1/embeddings",
            json={"model": MODEL, "input": ["test"]},
            headers={"Authorization": "Bearer fake-key-unknown-provider-test"},
        )
    report.record("Unknown provider in path -> 404", "PASS" if res.status_code == 404 else "FAIL", f"got {res.status_code}")


async def test_empty_input_array() -> None:
    async with httpx.AsyncClient() as client:
        res = await client.post(
            f"{ZEROCACHE_HOST}/{PROVIDER}/v1/embeddings",
            json={"model": MODEL, "input": []},
            headers={"Authorization": "Bearer fake-key-empty-input-test"},
        )
    body = res.json() if res.status_code == 200 else {}
    ok = res.status_code == 200 and body.get("data") == []
    report.record("Empty input array is accepted, returns empty data (no provider call)", "PASS" if ok else "FAIL", f"status={res.status_code} body={body}")


async def test_missing_model_field() -> None:
    # Valid JSON syntax, missing required field -> axum's Json<T> extraction
    # fails with 422 (distinct from 400 for syntax errors below), and now
    # comes back as {"error": "..."} like every other error path (fixed
    # 2026-07-24, found via the earlier LangChain TS battle-test).
    async with httpx.AsyncClient() as client:
        res = await client.post(
            f"{ZEROCACHE_HOST}/{PROVIDER}/v1/embeddings",
            json={"input": ["test"]},
            headers={"Authorization": "Bearer fake-key-missing-model-test"},
        )
    report.record("Missing required 'model' field -> 422", "PASS" if res.status_code == 422 else "FAIL", f"status={res.status_code}")
    try:
        parsed_as_app_error_shape = isinstance(res.json().get("error"), str)
    except Exception:
        parsed_as_app_error_shape = False
    report.record(
        "Error response uses the app's {\"error\": \"...\"} shape",
        "PASS" if parsed_as_app_error_shape else "FAIL",
        "" if parsed_as_app_error_shape else f"body={res.text[:150]}",
    )


async def test_malformed_json_body() -> None:
    async with httpx.AsyncClient() as client:
        res = await client.post(
            f"{ZEROCACHE_HOST}/{PROVIDER}/v1/embeddings",
            content="{not valid json",
            headers={"Authorization": "Bearer fake-key-malformed-json-test", "Content-Type": "application/json"},
        )
    report.record("Malformed JSON body -> 400", "PASS" if res.status_code == 400 else "FAIL", f"got {res.status_code}")


async def test_delete_without_auth() -> None:
    async with httpx.AsyncClient() as client:
        res = await client.request(
            "DELETE", f"{ZEROCACHE_HOST}/{PROVIDER}/v1/embeddings", json={"model": MODEL, "input": ["test"]}
        )
    report.record("DELETE without Authorization -> 401", "PASS" if res.status_code == 401 else "FAIL", f"got {res.status_code}")


async def test_fake_key_owner_scoping() -> None:
    unique = f"owner isolation probe {time.time()}"
    async with httpx.AsyncClient() as client:
        res = await client.post(
            f"{ZEROCACHE_HOST}/{PROVIDER}/v1/embeddings",
            json={"model": MODEL, "input": [unique]},
            headers={"Authorization": "Bearer fake-owner-a-test-key-000000000000000000"},
        )
    report.record(
        "A syntactically fake key is owner-scoped and fails at the provider (502), not silently cached",
        "PASS" if res.status_code == 502 else "FINDING",
        f"status={res.status_code}",
    )


# ---------- key-dependent tests, driven through llama-index-embeddings-openai itself ----------


def make_logging_embedding(label: str, embed_batch_size: int = 100) -> OpenAILikeEmbedding:
    # OpenAILikeEmbedding (not OpenAIEmbedding) -- OpenAIEmbedding hardcodes
    # a fixed enum of real OpenAI model names and rejects any other model
    # string client-side, before any network call, even though it accepts
    # an arbitrary api_base. OpenAILikeEmbedding exists exactly for this
    # case (an OpenAI-wire-compatible endpoint that isn't OpenAI itself --
    # the same pattern used for self-hosted vLLM/LM Studio): it passes
    # model_name instead of model to the parent class, which is explicitly
    # exempted from the enum check. Confirmed by reading
    # llama_index/embeddings/openai_like/base.py.
    #
    # OpenAILikeEmbedding (like OpenAIEmbedding) takes separate sync/async
    # http clients; every battle test below is async, so only
    # async_http_client actually gets exercised, but the sync one is
    # supplied too for a consistently fully-wired client.
    start_times: dict[int, float] = {}

    # Both hooks MUST be async here: httpx.AsyncClient does `await hook(...)`
    # for every registered hook, unconditionally. A plain `def` hook returns
    # None, and `await None` raises TypeError -- which the openai SDK's
    # tenacity-based retry wrapper caught, retried a few times, and
    # re-reported as a generic, misleading "APIConnectionError: Connection
    # error", hiding the actual bug for a long debugging stretch. httpx.Client
    # (sync, used for rag.py and this function's own `http_client=` param)
    # is the opposite -- its hooks must be plain sync functions instead.
    async def on_request(request: httpx.Request) -> None:
        start_times[id(request)] = time.monotonic()

    async def on_response(response: httpx.Response) -> None:
        start = start_times.pop(id(response.request), None)
        elapsed_ms = (time.monotonic() - start) * 1000 if start is not None else -1
        hits = response.headers.get("x-zerocache-hits")
        misses = response.headers.get("x-zerocache-misses")
        suffix = f" -- hits={hits} misses={misses}" if hits is not None or misses is not None else ""
        print(f"[{label}] {response.status_code} in {elapsed_ms:.0f}ms{suffix}")

    async_client = httpx.AsyncClient(event_hooks={"request": [on_request], "response": [on_response]})

    return OpenAILikeEmbedding(
        model_name=MODEL,
        api_key=require_api_key(),
        api_base=ZEROCACHE_GEMINI_BASE_URL,
        async_http_client=async_client,
        http_client=make_logging_http_client(label),
        embed_batch_size=embed_batch_size,
        # Default is 10 -- multiplies wall-clock time enormously on a
        # genuine connection issue and was masking the real error behind
        # a wall of retry log lines. 2 is enough to smooth over a single
        # transient blip without hiding a real, persistent problem.
        max_retries=2,
    )


async def test_basic_miss_then_hit() -> None:
    unique = f"battle-test roundtrip {time.time()}"
    embed_model = make_logging_embedding("roundtrip")
    first = await embed_model.aget_query_embedding(unique)
    second = await embed_model.aget_query_embedding(unique)
    report.record("Basic miss-then-hit returns an identical vector", "PASS" if first == second else "FAIL")


async def test_within_batch_dedup() -> None:
    unique = f"dedup test {time.time()}"
    embed_model = make_logging_embedding("dedup")
    results = await embed_model.aget_text_embedding_batch([unique, unique, unique], show_progress=False)
    all_same = all(v == results[0] for v in results)
    report.record("Duplicate texts within one batch all return the same vector", "PASS" if all_same else "FAIL")


async def test_whitespace_normalization() -> None:
    base = f"whitespace test {time.time()}"
    embed_model = make_logging_embedding("whitespace")
    first = await embed_model.aget_query_embedding(base)
    second = await embed_model.aget_query_embedding(f"   {base.replace(' ', '   ')}   ")
    report.record("Whitespace-variant text shares a cache entry with the normalized original", "PASS" if first == second else "FAIL")


async def test_large_batch_across_provider_chunk_boundary() -> None:
    stamp = time.time()
    texts = [f"large batch item {stamp}-{i}" for i in range(150)]
    # embed_batch_size=100 is LlamaIndex's OpenAIEmbedding default -- matches
    # Zerocache's own MAX_BATCH_SIZE, so it would never naturally send more
    # than 100 texts in one call. Overriding to 150 forces a single HTTP
    # call that Zerocache itself must internally chunk (100 + 50) against
    # the real Gemini API -- the actual boundary this test exists to check.
    embed_model = make_logging_embedding("large-batch-150", embed_batch_size=150)
    start = time.monotonic()
    results = await embed_model.aget_text_embedding_batch(texts, show_progress=False)
    elapsed_ms = (time.monotonic() - start) * 1000
    all_present = len(results) == 150 and all(len(v) > 0 for v in results)
    report.record(
        "150-item batch (crosses Zerocache's internal MAX_BATCH_SIZE=100 chunk boundary) returns all vectors correctly",
        "PASS" if all_present else "FAIL",
        f"{len(results)}/150 returned in {elapsed_ms:.0f}ms",
    )


async def fetch_metric(metric_line_prefix: str) -> int | None:
    async with httpx.AsyncClient() as client:
        res = await client.get(f"{ZEROCACHE_HOST}/metrics")
    for line in res.text.splitlines():
        if line.startswith(metric_line_prefix):
            return int(line.rsplit(" ", 1)[1])
    return None


async def test_concurrent_duplicate_requests_coalescing() -> None:
    # Deliberately uncached text, fired as N *simultaneous* requests for the
    # exact same text via asyncio.gather. Zerocache's in-process coalescing
    # (added 2026-07-24) should mean only one of these actually reaches the
    # provider; the rest share its result. Verifying that from the Python
    # side too, not just TS -- but see the note below on why this can't be
    # confirmed the same way TS could.
    unique = f"coalescing probe {time.time()}"
    concurrency = 5
    embed_models = [make_logging_embedding(f"coalesce-{i}") for i in range(concurrency)]

    tokens_metric = 'zerocache_provider_prompt_tokens_total{provider="gemini"}'
    before = await fetch_metric(tokens_metric)

    start = time.monotonic()
    outcomes = await asyncio.gather(*(m.aget_query_embedding(unique) for m in embed_models))
    elapsed_ms = (time.monotonic() - start) * 1000

    after = await fetch_metric(tokens_metric)

    all_same_vector = all(v == outcomes[0] for v in outcomes)
    report.record(f"{concurrency} concurrent identical-text requests all return the correct (identical) vector", "PASS" if all_same_vector else "FAIL")
    report.record(
        "Request coalescing across concurrent Python requests -- cannot self-verify via /metrics for this provider",
        "FINDING",
        f"{concurrency} concurrent misses for the same never-before-seen text resolved in {elapsed_ms:.0f}ms total "
        f"(prompt_tokens {before}->{after}, always 0 for Gemini per CLAUDE.md -- Gemini's real API reports no usage "
        "at all, so the token-delta check that concretely proved coalescing worked against OpenAI earlier this "
        "session structurally cannot work here). The mechanism itself is provider-agnostic (it dedupes on CacheKey "
        "before any provider call happens), so there's no reason to expect it behaves differently for Gemini -- but "
        "this specific provider gives no metrics signal to confirm it. A real, standalone finding: coalescing's "
        "only current verification path (token-count delta) silently degrades to unverifiable for any provider "
        "that doesn't report usage.",
    )


async def test_delete_then_reembed_is_a_miss_again() -> None:
    unique = f"delete roundtrip {time.time()}"
    api_key = require_api_key()
    embed_model = make_logging_embedding("delete-embed")

    await embed_model.aget_query_embedding(unique)  # now cached

    async with httpx.AsyncClient() as client:
        delete_res = await client.request(
            "DELETE",
            f"{ZEROCACHE_HOST}/{PROVIDER}/v1/embeddings",
            json={"model": MODEL, "input": [unique]},
            headers={"Authorization": f"Bearer {api_key}"},
        )
    delete_body = delete_res.json() if delete_res.status_code == 200 else {}

    if delete_res.status_code != 200 or delete_body.get("deleted") != 1:
        report.record("DELETE removes exactly the requested entry", "FAIL", f"status={delete_res.status_code} body={delete_body}")
        return
    report.record("DELETE removes exactly the requested entry", "PASS")

    async with httpx.AsyncClient() as client:
        after_delete_res = await client.post(
            f"{ZEROCACHE_HOST}/{PROVIDER}/v1/embeddings",
            json={"model": MODEL, "input": [unique]},
            headers={"Authorization": f"Bearer {api_key}"},
        )
    misses = after_delete_res.headers.get("x-zerocache-misses")
    report.record("A deleted entry is a miss again on the next request", "PASS" if misses == "1" else "FAIL", f"x-zerocache-misses={misses}")


async def main() -> None:
    # Throwaway request so the process's first couple of HTTP connections
    # aren't the ones under test -- observed a rare, non-reproducible flake
    # on the 3rd request in a fresh process's connection pool (both here
    # and independently in demo/langchain-ts's Node client), never on a
    # fresh curl invocation and never in an isolated repro. Confirmed via
    # direct inspection that the actual server response is always correct
    # (identical to curl's every time) -- this is client/connection-pool
    # warmup noise, not a Zerocache defect.
    async with httpx.AsyncClient() as client:
        try:
            await client.get(f"{ZEROCACHE_HOST}/health")
        except Exception:
            pass

    print("=== key-independent HTTP-contract tests ===")
    await run("health", test_health)
    await run("ready", test_ready)
    await run("metrics", test_metrics_endpoint)
    await run("missing-auth", test_missing_auth)
    await run("malformed-auth", test_malformed_auth)
    await run("unknown-provider", test_unknown_provider)
    await run("empty-input", test_empty_input_array)
    await run("missing-model-field", test_missing_model_field)
    await run("malformed-json", test_malformed_json_body)
    await run("delete-without-auth", test_delete_without_auth)
    await run("fake-key-owner-scoping", test_fake_key_owner_scoping)

    if not API_KEY:
        print("\n=== key-dependent tests SKIPPED: ZEROCACHE_DEMO_KEY not set ===")
        for name in [
            "basic-miss-then-hit",
            "within-batch-dedup",
            "whitespace-normalization",
            "large-batch-boundary",
            "concurrent-coalescing-probe",
            "delete-then-reembed",
        ]:
            report.record(name, "SKIP")
    else:
        print("\n=== key-dependent tests (real Gemini calls through Zerocache, real cost) ===")
        await run("basic-miss-then-hit", test_basic_miss_then_hit)
        await run("within-batch-dedup", test_within_batch_dedup)
        await run("whitespace-normalization", test_whitespace_normalization)
        await run("large-batch-boundary", test_large_batch_across_provider_chunk_boundary)
        await run("concurrent-coalescing-probe", test_concurrent_duplicate_requests_coalescing)
        await run("delete-then-reembed", test_delete_then_reembed_is_a_miss_again)

    print("\n=== summary ===")
    counts = {"PASS": 0, "FAIL": 0, "FINDING": 0, "SKIP": 0}
    for _, result, _ in report.results:
        counts[result] += 1
    print(f"{counts['PASS']} passed, {counts['FAIL']} failed, {counts['FINDING']} findings, {counts['SKIP']} skipped")

    if counts["FAIL"] > 0:
        print("\nFAILURES:")
        for name, result, detail in report.results:
            if result == "FAIL":
                print(f"  - {name}: {detail}")
    if counts["FINDING"] > 0:
        print("\nFINDINGS (not failures, but worth flagging):")
        for name, result, detail in report.results:
            if result == "FINDING":
                print(f"  - {name}: {detail}")

    if counts["FAIL"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
