use serde::{Deserialize, Serialize};
use zerocache_adapters_cloud::{EmbedCall, EmbedOutcome, ResolvedModel, TextWireStrategy};
use zerocache_ports::{ProviderError, ProviderUsage};

/// Bedrock's API-key auth: a plain bearer token against bedrock-runtime, no
/// SigV4. Verified 2026-07-28 against AWS's own cURL example in
/// docs.aws.amazon.com/bedrock/latest/userguide/api-keys-use.html.
fn bedrock_headers(api_key: &str) -> Vec<(&'static str, String)> {
    vec![
        ("Authorization", format!("Bearer {api_key}")),
        ("Accept", "application/json".to_string()),
    ]
}

fn invoke_url(resolved: &ResolvedModel) -> String {
    format!(
        "{}/model/{}/invoke",
        resolved.endpoint_base, resolved.model_id
    )
}

// ---------------------------------------------------------------- Titan ----

#[derive(Serialize)]
struct TitanRequest<'a> {
    #[serde(rename = "inputText")]
    input_text: &'a str,
}

#[derive(Deserialize)]
struct TitanResponse {
    embedding: Vec<f32>,
    #[serde(rename = "inputTextTokenCount")]
    input_text_token_count: u32,
}

/// Amazon Titan Embeddings (`amazon.titan-embed-text-v1`,
/// `amazon.titan-embed-text-v2:0`).
///
/// `inputText` is a scalar string, not an array, so one text per call is not a
/// conservative choice -- it is the only thing the API accepts. A 500-chunk
/// ingestion batch that misses entirely costs 500 sequential HTTP calls
/// against Titan, which is worth knowing before choosing it as an ingestion
/// model.
pub struct TitanEmbedStrategy;

impl TextWireStrategy for TitanEmbedStrategy {
    fn max_batch(&self) -> usize {
        1
    }

    fn build_call(
        &self,
        api_key: &str,
        resolved: &ResolvedModel,
        texts: &[String],
    ) -> Result<EmbedCall, ProviderError> {
        let text = texts.first().ok_or_else(|| {
            ProviderError("titan strategy called with an empty chunk".to_string())
        })?;

        // Deliberately not sending `dimensions`, `normalize`, or
        // `embeddingTypes`: all three change the returned vector, none is
        // expressible in the OpenAI-shaped wire contract Zerocache exposes,
        // and none would be visible in the cache key. Titan's own defaults
        // (1024 dims, normalized, float) are what a caller gets.
        let body = serde_json::to_vec(&TitanRequest { input_text: text })
            .map_err(|e| ProviderError(e.to_string()))?;

        Ok(EmbedCall {
            url: invoke_url(resolved),
            headers: bedrock_headers(api_key),
            body,
        })
    }

    fn parse_response(&self, _expected: usize, body: &[u8]) -> Result<EmbedOutcome, ProviderError> {
        let parsed: TitanResponse =
            serde_json::from_slice(body).map_err(|e| ProviderError(e.to_string()))?;
        Ok(EmbedOutcome {
            vectors: vec![parsed.embedding],
            usage: ProviderUsage {
                prompt_tokens: parsed.input_text_token_count,
                total_tokens: parsed.input_text_token_count,
            },
        })
    }
}

// --------------------------------------------------------------- Cohere ----

#[derive(Serialize)]
struct CohereRequest<'a> {
    texts: &'a [String],
    input_type: &'a str,
}

/// Cohere's response is polymorphic: a flat `[[f32]]` when `embedding_types`
/// was not requested (`response_type: "embeddings_floats"`), or a
/// type-keyed object when it was (`response_type: "embeddings_by_type"`).
/// This strategy never requests `embedding_types`, so the flat form is what
/// it expects -- but it parses both, so a Bedrock-side default change
/// degrades into a working parse instead of a deserialization error at
/// runtime.
#[derive(Deserialize)]
#[serde(untagged)]
enum CohereEmbeddings {
    Flat(Vec<Vec<f32>>),
    ByType { float: Vec<Vec<f32>> },
}

#[derive(Deserialize)]
struct CohereResponse {
    embeddings: CohereEmbeddings,
}

/// Cohere Embed v3 (`cohere.embed-english-v3`,
/// `cohere.embed-multilingual-v3`) and v4 (`cohere.embed-v4:0`). Both accept
/// `texts` + `input_type` and return the same envelope, so one strategy
/// covers both.
pub struct CohereEmbedStrategy;

impl TextWireStrategy for CohereEmbedStrategy {
    fn max_batch(&self) -> usize {
        // AWS documents "0 to 96 texts per call" for v3 and "Max 96 per call"
        // for v4 -- the same number, and a real API limit rather than the
        // unverified 100 the four wire-shape-fixed adapters share.
        96
    }

    fn build_call(
        &self,
        api_key: &str,
        resolved: &ResolvedModel,
        texts: &[String],
    ) -> Result<EmbedCall, ProviderError> {
        let input_type = resolved.qualifier.as_deref().ok_or_else(|| {
            ProviderError("cohere strategy reached without a resolved input_type".to_string())
        })?;

        let body = serde_json::to_vec(&CohereRequest { texts, input_type })
            .map_err(|e| ProviderError(e.to_string()))?;

        Ok(EmbedCall {
            url: invoke_url(resolved),
            headers: bedrock_headers(api_key),
            body,
        })
    }

    fn parse_response(&self, _expected: usize, body: &[u8]) -> Result<EmbedOutcome, ProviderError> {
        let parsed: CohereResponse =
            serde_json::from_slice(body).map_err(|e| ProviderError(e.to_string()))?;
        let vectors = match parsed.embeddings {
            CohereEmbeddings::Flat(v) => v,
            CohereEmbeddings::ByType { float } => float,
        };
        // Cohere on Bedrock reports no token usage at all -- report zero
        // rather than fabricate, the same posture as Gemini and HuggingFace.
        Ok(EmbedOutcome {
            vectors,
            usage: ProviderUsage::default(),
        })
    }
}

#[cfg(test)]
mod tests {
    use httpmock::prelude::*;
    use serde_json::json;
    use zerocache_ports::EmbeddingProvider;

    use crate::{new_provider, BedrockProvider};

    fn provider(base: String) -> BedrockProvider {
        new_provider("us-east-1", base)
    }

    #[tokio::test]
    async fn titan_sends_scalar_input_text_with_bearer_auth_and_reports_its_token_count() {
        let server = MockServer::start_async().await;
        let mock = server
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/model/amazon.titan-embed-text-v2:0/invoke")
                    .header("Authorization", "Bearer test-key")
                    .json_body(json!({ "inputText": "hello" }));
                then.status(200)
                    .json_body(json!({ "embedding": [1.0, 2.0], "inputTextTokenCount": 3 }));
            })
            .await;

        let (vectors, usage) = provider(server.base_url())
            .embed_batch(
                "test-key",
                "amazon.titan-embed-text-v2:0",
                &["hello".to_string()],
            )
            .await
            .unwrap();

        mock.assert_async().await;
        assert_eq!(vectors, vec![vec![1.0, 2.0]]);
        assert_eq!(
            usage.prompt_tokens, 3,
            "Titan does report token usage -- it must not be dropped"
        );
        assert_eq!(usage.total_tokens, 3);
    }

    #[tokio::test]
    async fn titan_issues_one_call_per_text_and_sums_usage_across_them() {
        // inputText is a scalar, so N texts is genuinely N calls.
        let server = MockServer::start_async().await;
        let mock = server
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/model/amazon.titan-embed-text-v2:0/invoke");
                then.status(200)
                    .json_body(json!({ "embedding": [7.0], "inputTextTokenCount": 4 }));
            })
            .await;

        let texts = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let (vectors, usage) = provider(server.base_url())
            .embed_batch("test-key", "amazon.titan-embed-text-v2:0", &texts)
            .await
            .unwrap();

        assert_eq!(mock.hits_async().await, 3);
        assert_eq!(vectors.len(), 3);
        assert_eq!(usage.prompt_tokens, 12);
    }

    #[tokio::test]
    async fn cohere_sends_a_texts_array_with_the_required_input_type() {
        let server = MockServer::start_async().await;
        let mock = server
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/model/cohere.embed-english-v3/invoke")
                    .header("Authorization", "Bearer test-key")
                    .json_body(json!({ "texts": ["a", "b"], "input_type": "search_document" }));
                then.status(200).json_body(json!({
                    "id": "abc",
                    "response_type": "embeddings_floats",
                    "embeddings": [[1.0], [2.0]],
                    "texts": ["a", "b"]
                }));
            })
            .await;

        let (vectors, usage) = provider(server.base_url())
            .embed_batch(
                "test-key",
                "cohere.embed-english-v3",
                &["a".to_string(), "b".to_string()],
            )
            .await
            .unwrap();

        mock.assert_async().await;
        assert_eq!(vectors, vec![vec![1.0], vec![2.0]]);
        assert_eq!(
            usage.prompt_tokens, 0,
            "Cohere on Bedrock reports no usage at all -- must stay zero, not fabricated"
        );
    }

    #[tokio::test]
    async fn cohere_input_type_qualifier_reaches_the_wire() {
        let server = MockServer::start_async().await;
        let mock = server
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/model/cohere.embed-english-v3/invoke")
                    .json_body(json!({ "texts": ["q"], "input_type": "search_query" }));
                then.status(200).json_body(
                    json!({ "response_type": "embeddings_floats", "embeddings": [[5.0]] }),
                );
            })
            .await;

        provider(server.base_url())
            .embed_batch(
                "test-key",
                "cohere.embed-english-v3#search_query",
                &["q".to_string()],
            )
            .await
            .unwrap();

        mock.assert_async().await;
    }

    #[tokio::test]
    async fn cohere_also_parses_the_embeddings_by_type_response_shape() {
        // This strategy never asks for embedding_types, so it should always
        // get the flat form -- but a Bedrock-side default change must degrade
        // into a working parse, not a deserialization error in production.
        let server = MockServer::start_async().await;
        server
            .mock_async(|when, then| {
                when.method(POST).path("/model/cohere.embed-v4:0/invoke");
                then.status(200).json_body(json!({
                    "response_type": "embeddings_by_type",
                    "embeddings": { "float": [[1.5], [2.5]] }
                }));
            })
            .await;

        let (vectors, _usage) = provider(server.base_url())
            .embed_batch(
                "test-key",
                "cohere.embed-v4:0",
                &["a".to_string(), "b".to_string()],
            )
            .await
            .unwrap();

        assert_eq!(vectors, vec![vec![1.5], vec![2.5]]);
    }

    #[tokio::test]
    async fn cohere_chunks_at_ninety_six_and_concatenates_in_order() {
        let server = MockServer::start_async().await;
        let first = server
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/model/cohere.embed-english-v3/invoke")
                    .matches(|req| {
                        let body: serde_json::Value =
                            serde_json::from_slice(req.body.as_deref().unwrap_or_default())
                                .unwrap();
                        body["texts"].as_array().map(|a| a.len()) == Some(96)
                    });
                then.status(200).json_body_obj(&json!({
                    "embeddings": (0..96).map(|i| json!([i as f64])).collect::<Vec<_>>()
                }));
            })
            .await;
        let second = server
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/model/cohere.embed-english-v3/invoke")
                    .matches(|req| {
                        let body: serde_json::Value =
                            serde_json::from_slice(req.body.as_deref().unwrap_or_default())
                                .unwrap();
                        body["texts"].as_array().map(|a| a.len()) == Some(4)
                    });
                then.status(200).json_body_obj(&json!({
                    "embeddings": (0..4).map(|i| json!([1000.0 + i as f64])).collect::<Vec<_>>()
                }));
            })
            .await;

        let texts: Vec<String> = (0..100).map(|i| format!("text-{i}")).collect();
        let (vectors, _usage) = provider(server.base_url())
            .embed_batch("test-key", "cohere.embed-english-v3", &texts)
            .await
            .unwrap();

        first.assert_async().await;
        second.assert_async().await;
        assert_eq!(vectors.len(), 100);
        assert_eq!(vectors[0], vec![0.0]);
        assert_eq!(vectors[95], vec![95.0]);
        assert_eq!(vectors[96], vec![1000.0]);
        assert_eq!(vectors[99], vec![1003.0]);
    }

    #[tokio::test]
    async fn an_http_error_status_surfaces_as_an_error() {
        let server = MockServer::start_async().await;
        server
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/model/cohere.embed-english-v3/invoke");
                then.status(403)
                    .json_body(json!({ "message": "not authorized" }));
            })
            .await;

        let result = provider(server.base_url())
            .embed_batch("bad-key", "cohere.embed-english-v3", &["x".to_string()])
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn a_response_count_mismatch_is_a_hard_error() {
        let server = MockServer::start_async().await;
        server
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/model/cohere.embed-english-v3/invoke");
                then.status(200).json_body(json!({ "embeddings": [[1.0]] }));
            })
            .await;

        let result = provider(server.base_url())
            .embed_batch(
                "test-key",
                "cohere.embed-english-v3",
                &["a".to_string(), "b".to_string()],
            )
            .await;

        assert!(
            result.is_err(),
            "a count mismatch must be a hard error, not a silent misalignment"
        );
    }

    #[test]
    fn cache_scope_separates_regions_vendors_and_input_types() {
        let p = provider("https://bedrock-runtime.{region}.amazonaws.com".to_string());
        let east = p.cache_scope("us-east-1/cohere.embed-english-v3").unwrap();
        let west = p.cache_scope("eu-west-1/cohere.embed-english-v3").unwrap();
        let query = p
            .cache_scope("us-east-1/cohere.embed-english-v3#search_query")
            .unwrap();
        let titan = p
            .cache_scope("us-east-1/amazon.titan-embed-text-v2:0")
            .unwrap();

        assert_ne!(east, west, "region must not be cached across");
        assert_ne!(east, query, "input_type must not be cached across");
        assert_ne!(east, titan, "vendor must not be cached across");
    }

    #[test]
    fn cache_scope_rejects_an_unsupported_model_rather_than_inventing_a_scope() {
        let p = provider("https://bedrock-runtime.{region}.amazonaws.com".to_string());
        assert!(p.cache_scope("meta.llama3-8b").is_err());
    }
}
