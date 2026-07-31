# syntax=docker/dockerfile:1

# ---- Planner: compute cargo-chef's dependency recipe ----
FROM rust:1.97.1-slim-bookworm AS chef
RUN apt-get update && apt-get install -y --no-install-recommends \
        pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*
RUN cargo install cargo-chef --locked
WORKDIR /build

FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

# ---- Builder: cook (cache) dependencies, then build the real binary ----
FROM chef AS builder
COPY --from=planner /build/recipe.json recipe.json
RUN cargo chef cook --release --recipe-path recipe.json
COPY . .
RUN cargo build --release -p zerocache-http

# ---- Runtime: minimal image with just the compiled binary ----
FROM debian:bookworm-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates libssl3 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --home-dir /nonexistent --shell /usr/sbin/nologin zerocache

COPY --from=builder /build/target/release/zerocache-http /usr/local/bin/zerocache-http

ENV ZEROCACHE_STORAGE_PATH=/data
RUN mkdir -p /data && chown zerocache:zerocache /data
VOLUME ["/data"]

USER zerocache
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -fs http://127.0.0.1:8080/health || exit 1

ENTRYPOINT ["/usr/local/bin/zerocache-http"]
