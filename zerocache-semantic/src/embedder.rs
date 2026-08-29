use std::fmt;

#[derive(Debug)]
pub struct SemanticError(pub String);

impl fmt::Display for SemanticError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "semantic error: {}", self.0)
    }
}

impl std::error::Error for SemanticError {}

/// Turns text into an L2-normalized embedding. A trait (not just the
/// concrete `TextEmbedder`) so `zerocache-http`'s tests can substitute a
/// deterministic mock and never load the real model.
pub trait TextEmbed: Send + Sync {
    fn embed(&self, text: &str) -> Result<Vec<f32>, SemanticError>;
}
