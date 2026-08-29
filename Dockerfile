# syntax=docker/dockerfile:1

# Static musl build -> the runtime image is `FROM scratch` plus one binary.
# reqwest is rustls-only (no OpenSSL) and rustls ships webpki roots, so the
# image needs no libssl, no ca-certificates, no shell -- nothing but the binary.

# ---- dashboard: build the Astro SPA fresh (committed dashboard/dist is only a
#      convenience for Node-less `cargo build`; the image never trusts it) ----
FROM node:22-slim AS dashboard
WORKDIR /dashboard
COPY dashboard/package.json dashboard/package-lock.json ./
RUN npm ci
COPY dashboard/ ./
RUN npm run build

# ---- chef: toolchain with the musl target + cargo-chef ----
FROM rust:1.97.1-slim-bookworm AS chef
RUN apt-get update && apt-get install -y --no-install-recommends \
        musl-tools musl-dev \
    && rm -rf /var/lib/apt/lists/* \
    && rustup target add x86_64-unknown-linux-musl \
    && cargo install cargo-chef --locked
WORKDIR /build
ENV CC_x86_64_unknown_linux_musl=musl-gcc

# ---- planner: distil the dependency graph ----
FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

# ---- builder: cook deps (cached), then build the static binary ----
FROM chef AS builder
COPY --from=planner /build/recipe.json recipe.json
RUN cargo chef cook --release --target x86_64-unknown-linux-musl --recipe-path recipe.json
COPY . .
# zerocache-http/src/dashboard.rs embeds dashboard/dist via include_dir!.
# Overwrite the committed copy with the freshly-built one so the image's
# dashboard always matches dashboard/src regardless of what's checked in.
COPY --from=dashboard /dashboard/dist ./dashboard/dist
RUN cargo build --release --target x86_64-unknown-linux-musl -p zerocache-http
# an empty, correctly-owned data dir to hand to the scratch stage (scratch has
# no shell to `mkdir`)
RUN install -d -o 10001 -g 10001 /out/data

# ---- runtime: nothing but the binary ----
FROM scratch AS runtime
COPY --from=builder /build/target/x86_64-unknown-linux-musl/release/zerocache-http /usr/local/bin/zerocache-http
COPY --from=builder --chown=10001:10001 /out/data /data

ENV ZEROCACHE_STORAGE_PATH=/data
VOLUME ["/data"]
USER 10001:10001
EXPOSE 8080

# The binary is its own probe -- the scratch image has no curl.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD ["/usr/local/bin/zerocache-http", "--health-check"]

ENTRYPOINT ["/usr/local/bin/zerocache-http"]
