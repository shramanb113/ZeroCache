use serde::{Deserialize, Deserializer, Serialize};

#[derive(Deserialize)]
pub struct EmbeddingsRequest {
    pub model: String,
    #[serde(deserialize_with = "deserialize_input")]
    pub input: Vec<String>,
}

// OpenAI's real /v1/embeddings contract accepts `input` as either a single
// string or an array of strings -- a bare string is exactly what a single
// embedQuery() call from real clients (confirmed against @langchain/openai
// and the underlying openai npm SDK) sends for a lone piece of text, not a
// one-element array. Without this, Zerocache silently breaks the "point
// your existing OpenAI-compatible client at Zerocache, zero SDK changes"
// promise for that call shape: found via LangChain TS battle-testing,
// where it broke embedQuery, a first-class method, not an edge case, and
// MemoryVectorStore.similaritySearch's query-embedding step under the hood.
fn deserialize_input<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum StringOrVec {
        One(String),
        Many(Vec<String>),
    }

    match StringOrVec::deserialize(deserializer)? {
        StringOrVec::One(s) => Ok(vec![s]),
        StringOrVec::Many(v) => Ok(v),
    }
}

#[derive(Serialize)]
pub struct EmbeddingsResponse {
    pub object: &'static str,
    pub data: Vec<EmbeddingObject>,
    pub model: String,
    pub usage: Usage,
}

#[derive(Serialize)]
pub struct EmbeddingObject {
    pub embedding: Vec<f32>,
    pub index: usize,
}

#[derive(Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

#[derive(Serialize)]
pub struct DeleteResponse {
    pub deleted: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_accepts_a_json_array_of_strings() {
        let req: EmbeddingsRequest = serde_json::from_str(r#"{"model":"m","input":["a","b"]}"#).unwrap();
        assert_eq!(req.input, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn input_accepts_a_bare_json_string_as_a_single_element_vec() {
        // What embedQuery() (a single-text call) actually sends -- confirmed
        // against @langchain/openai's embeddings.js during battle-testing.
        let req: EmbeddingsRequest = serde_json::from_str(r#"{"model":"m","input":"just one string"}"#).unwrap();
        assert_eq!(req.input, vec!["just one string".to_string()]);
    }

    #[test]
    fn input_rejects_a_non_string_array_element() {
        let result: Result<EmbeddingsRequest, _> = serde_json::from_str(r#"{"model":"m","input":[1,2,3]}"#);
        assert!(result.is_err(), "an array of numbers must still be rejected, not silently coerced");
    }
}
