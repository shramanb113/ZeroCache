use std::fmt;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use tokenizers::Tokenizer;

const MODEL_F16: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/model/all-MiniLM-L6-v2.f16.safetensors"
));
const TOKENIZER_JSON: &[u8] =
    include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/model/tokenizer.json"));
const CONFIG_JSON: &[u8] = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/model/config.json"));

const MAX_TOKENS: usize = 256;

#[derive(Debug)]
pub struct SemanticError(pub String);

impl fmt::Display for SemanticError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "semantic error: {}", self.0)
    }
}

impl std::error::Error for SemanticError {}

/// Text to an L2-normalized embedding. A trait so `zerocache-http` tests can
/// swap in a deterministic mock instead of loading the real model.
pub trait TextEmbed: Send + Sync {
    fn embed(&self, text: &str) -> Result<Vec<f32>, SemanticError>;
}

/// all-MiniLM-L6-v2 via candle, CPU, weights compiled in. Build once with
/// `load()`; call `embed` from `spawn_blocking` (CPU-bound).
pub struct TextEmbedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl TextEmbedder {
    pub fn load() -> Result<Self, SemanticError> {
        let device = Device::Cpu;
        let config: Config = serde_json::from_slice(CONFIG_JSON)
            .map_err(|e| SemanticError(format!("config.json: {e}")))?;
        let mut tokenizer = Tokenizer::from_bytes(TOKENIZER_JSON)
            .map_err(|e| SemanticError(format!("tokenizer.json: {e}")))?;
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: MAX_TOKENS,
                ..Default::default()
            }))
            .map_err(|e| SemanticError(format!("tokenizer truncation: {e}")))?;

        let vb = VarBuilder::from_slice_safetensors(MODEL_F16, DType::F32, &device)
            .map_err(|e| SemanticError(format!("safetensors: {e}")))?;
        let model =
            BertModel::load(vb, &config).map_err(|e| SemanticError(format!("bert load: {e}")))?;

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    fn embed_inner(&self, text: &str) -> candle_core::Result<Vec<f32>> {
        let enc = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let n = enc.get_ids().len();
        let ids = Tensor::from_vec(enc.get_ids().to_vec(), (1, n), &self.device)?;
        let mask = Tensor::from_vec(enc.get_attention_mask().to_vec(), (1, n), &self.device)?;
        let token_type_ids = ids.zeros_like()?;

        let hidden = self.model.forward(&ids, &token_type_ids, Some(&mask))?; // (1, n, 384)

        let mask_f = mask.to_dtype(DType::F32)?.unsqueeze(2)?; // (1, n, 1)
        let summed = hidden.broadcast_mul(&mask_f)?.sum(1)?; // (1, 384)
        let counts = mask_f.sum(1)?.clamp(1e-9, f32::INFINITY)?; // (1, 1)
        let mean = summed.broadcast_div(&counts)?;

        let norm = mean.sqr()?.sum_keepdim(1)?.sqrt()?.clamp(1e-12, f32::INFINITY)?;
        mean.broadcast_div(&norm)?.squeeze(0)?.to_vec1::<f32>()
    }
}

impl TextEmbed for TextEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>, SemanticError> {
        self.embed_inner(text)
            .map_err(|e| SemanticError(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[test]
    fn load_succeeds_and_embed_returns_a_unit_384_vector() {
        let e = TextEmbedder::load().expect("model is embedded in the binary");
        let v = e.embed("how do I reset my password?").unwrap();
        assert_eq!(v.len(), crate::EMBEDDING_DIM);
        assert!(v.iter().all(|f| f.is_finite()));
        let norm = dot(&v, &v).sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "norm was {norm}");
    }

    #[test]
    fn identical_text_produces_a_bit_identical_vector() {
        let e = TextEmbedder::load().unwrap();
        assert_eq!(e.embed("same text").unwrap(), e.embed("same text").unwrap());
    }

    #[test]
    fn a_paraphrase_is_close_and_an_unrelated_sentence_is_far() {
        let e = TextEmbedder::load().unwrap();
        let q = e.embed("How do I reset my password?").unwrap();
        let para = e.embed("how can i reset my password").unwrap();
        let unrelated = e.embed("what is the capital of France").unwrap();
        assert!(dot(&q, &para) > 0.85, "paraphrase {}", dot(&q, &para));
        assert!(dot(&q, &unrelated) < 0.6, "unrelated {}", dot(&q, &unrelated));
    }
}
