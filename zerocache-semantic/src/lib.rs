//! Local semantic layer for the completion cache: a CPU text embedder
//! (candle + all-MiniLM-L6-v2, model bytes compiled in) and an in-memory
//! per-scope HNSW index. No disk, no network. Compiled only into a
//! `zerocache-http --features semantic` build.

mod embedder;
mod index;

pub use embedder::{SemanticError, TextEmbed};
pub use index::{ScopeKey, SearchHit, SemanticIndex};

/// all-MiniLM-L6-v2 output width.
pub const EMBEDDING_DIM: usize = 384;
