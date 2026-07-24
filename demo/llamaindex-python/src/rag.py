"""A minimal RAG pipeline: chunk a small corpus, embed the chunks through
Zerocache (not directly against Gemini), index the vectors, then answer a
query via similarity search.

Run it twice in a row (`uv run python src/rag.py`) to see the point of the
whole exercise: the first run is all misses, the second run -- same
corpus, same model, same key -- should be all hits, and noticeably faster.
"""

import time

from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai_like import OpenAILikeEmbedding

from logging_http import make_logging_http_client
from zerocache_config import MODEL, ZEROCACHE_GEMINI_BASE_URL, require_api_key

CORPUS = [
    (
        "Zerocache overview",
        "Zerocache is a Rust-native embedding cache that sits between an application's ingestion pipeline "
        "and its embedding provider. It intercepts OpenAI-compatible /v1/embeddings requests, serves "
        "previously-computed vectors from a local content-addressed store, and forwards only cache misses "
        "upstream. Adoption requires no SDK -- a consumer just points its existing embedding client at "
        "Zerocache's base_url.",
    ),
    (
        "Cache key design",
        "The cache key is derived from blake3(owner_id, provider, model, model_version, text). Owner_id is "
        "a hash of the caller's forwarded API key, never the raw key itself. Including model_version means "
        "a model upgrade can never silently return a stale-but-plausible vector from an older model version.",
    ),
    (
        "Multi-tenant isolation",
        "Every request requires a real forwarded provider API key via the Authorization header. Two "
        "different callers embedding identical text under the same model share a cache hit for free, but "
        "two different callers never share a cache entry with each other, by design -- this avoids both "
        "unfair cost sharing and a cache-timing existence-leak risk.",
    ),
    (
        "Storage backends",
        "Zerocache ships two EmbeddingStore implementations: an embedded sled store for local development "
        "and single-instance deployments, and a Redis-backed store for multi-replica Kubernetes deployments. "
        "Selection happens at startup via the ZEROCACHE_STORAGE_BACKEND environment variable.",
    ),
    (
        "Supported providers",
        "Three EmbeddingProvider adapters exist today: OpenAI, Mistral, and Gemini, selected per-request via "
        "the URL path as POST /{provider}/v1/embeddings. Each adapter chunks its input into batches of at "
        "most 100 texts per upstream call.",
    ),
]


def ingest() -> VectorStoreIndex:
    documents = [Document(text=text, metadata={"title": title}) for title, text in CORPUS]
    splitter = SentenceSplitter(chunk_size=200, chunk_overlap=20)

    start = time.monotonic()
    index = VectorStoreIndex.from_documents(documents, transformations=[splitter], show_progress=False)
    elapsed_ms = (time.monotonic() - start) * 1000
    print(f"ingested {len(CORPUS)} source documents in {elapsed_ms:.0f}ms\n")
    return index


def query(index: VectorStoreIndex, question: str) -> None:
    print(f'\nquery: "{question}"')
    retriever = index.as_retriever(similarity_top_k=2)
    results = retriever.retrieve(question)
    for result in results:
        title = result.node.metadata.get("title", "?")
        snippet = result.node.get_content()[:80]
        print(f"  - [{title}] {snippet}...")


def main() -> None:
    api_key = require_api_key()

    Settings.embed_model = OpenAILikeEmbedding(
        model_name=MODEL,
        api_key=api_key,
        api_base=ZEROCACHE_GEMINI_BASE_URL,
        embed_batch_size=100,
        http_client=make_logging_http_client("rag"),
    )

    print("=== ingesting corpus (expect all misses on a cold cache) ===")
    index = ingest()

    query(index, "How does Zerocache decide which cache entry belongs to which caller?")
    query(index, "What storage backends does Zerocache support?")


if __name__ == "__main__":
    main()
