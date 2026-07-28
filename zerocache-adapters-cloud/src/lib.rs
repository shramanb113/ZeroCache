//! Shared adapter layer for cloud embedding surfaces (Azure, Amazon Bedrock,
//! GCP Vertex AI).
//!
//! Each of those clouds is a single API in front of several independent
//! vendors with different request/response JSON, so the variation is
//! two-dimensional and the abstraction is two traits:
//!
//! - [`CloudRouter`] answers "given the caller's `model` string, which
//!   endpoint and which wire strategy?"
//! - [`TextWireStrategy`] answers "what bytes go up, what comes back, and how
//!   many inputs fit in one call?"
//!
//! Everything between them -- client construction, chunking, status mapping,
//! count checking, usage accumulation -- lives once in [`CloudProvider`],
//! which is the only thing here that implements
//! `zerocache_ports::EmbeddingProvider`.
//!
//! This crate deliberately does not absorb `zerocache-adapters-openai`,
//! `-mistral`, `-gemini`, or `-huggingface`. Those four have a fixed wire
//! shape and no vendor dimension; folding them in would be a large diff with
//! no user-visible benefit.

mod client;
mod driver;
mod strategy;

pub use client::{build_client, KIT_VERSION, MAX_RETRIES, PROVIDER_TIMEOUT};
pub use driver::CloudProvider;
pub use strategy::{CloudRouter, EmbedCall, EmbedOutcome, ResolvedModel, TextWireStrategy};
