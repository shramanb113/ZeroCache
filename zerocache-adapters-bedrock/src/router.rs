use zerocache_adapters_cloud::{CloudRouter, ResolvedModel, TextWireStrategy};
use zerocache_ports::ProviderError;

use crate::strategy::{CohereEmbedStrategy, TitanEmbedStrategy};

pub const DEFAULT_BEDROCK_REGION: &str = "us-east-1";

/// `{region}` is substituted per request. A template with no placeholder is
/// used verbatim, which is what makes httpmock testing possible and also
/// covers a VPC/PrivateLink endpoint that is not region-templated.
pub const DEFAULT_BEDROCK_ENDPOINT_TEMPLATE: &str = "https://bedrock-runtime.{region}.amazonaws.com";

/// Cohere requires `input_type` and it changes the vector. `search_document`
/// is the corpus-indexing value, which is what an ingestion cache is
/// overwhelmingly used for; a caller embedding queries must say so explicitly
/// with a `#search_query` suffix, and gets a separate cache namespace for free
/// because the suffix is part of the canonical model form.
pub const DEFAULT_COHERE_INPUT_TYPE: &str = "search_document";

pub struct BedrockRouter {
    default_region: String,
    endpoint_template: String,
    titan: TitanEmbedStrategy,
    cohere: CohereEmbedStrategy,
}

impl BedrockRouter {
    pub fn new(default_region: impl Into<String>, endpoint_template: impl Into<String>) -> Self {
        Self {
            default_region: default_region.into(),
            endpoint_template: endpoint_template.into(),
            titan: TitanEmbedStrategy,
            cohere: CohereEmbedStrategy,
        }
    }

    /// Defaults matching the real AWS endpoint. Used by tests and by
    /// zerocache-http when no override env vars are set.
    pub fn with_defaults() -> Self {
        Self::new(DEFAULT_BEDROCK_REGION, DEFAULT_BEDROCK_ENDPOINT_TEMPLATE)
    }
}

/// One caller `model` string split into its parts.
///
/// Grammar: `[<region>/]<modelId>[#<input_type>]`
///
/// Splitting is unambiguous: Bedrock model IDs contain `.` and `:` (e.g.
/// `amazon.titan-embed-text-v2:0`) but never `/`, and `#` is not legal in a
/// model ID at all.
struct BedrockModelParts {
    region: String,
    model_id: String,
    input_type: Option<String>,
}

fn split_model(model: &str, default_region: &str) -> Result<BedrockModelParts, ProviderError> {
    let (head, input_type) = match model.split_once('#') {
        Some((h, q)) if !q.is_empty() => (h, Some(q.to_string())),
        Some((_, _)) => {
            return Err(ProviderError(format!(
                "bedrock model '{model}' has an empty '#' qualifier -- expected '#<input_type>' or no '#' at all"
            )))
        }
        None => (model, None),
    };

    let (region, model_id) = match head.split_once('/') {
        Some((r, m)) => (r.to_string(), m.to_string()),
        None => (default_region.to_string(), head.to_string()),
    };

    if region.is_empty() || model_id.is_empty() {
        return Err(ProviderError(format!(
            "bedrock model '{model}' is malformed -- expected '[<region>/]<modelId>[#<input_type>]', e.g. 'us-east-1/cohere.embed-english-v3#search_query'"
        )));
    }

    Ok(BedrockModelParts { region, model_id, input_type })
}

impl CloudRouter for BedrockRouter {
    fn resolve(&self, model: &str) -> Result<ResolvedModel, ProviderError> {
        let parts = split_model(model, &self.default_region)?;
        let endpoint_base = self.endpoint_template.replace("{region}", &parts.region);

        // Cohere's input_type is baked into the canonical form so that
        // `x#search_query` and `x#search_document` -- which produce genuinely
        // different vectors for identical text -- never share a cache entry.
        // Titan has no such parameter, so its canonical form omits it even if
        // the caller supplied one, and an unusable qualifier is rejected
        // rather than silently ignored.
        let is_cohere = parts.model_id.starts_with("cohere.embed");
        let is_titan = parts.model_id.starts_with("amazon.titan-embed");

        if !is_cohere && !is_titan {
            return Err(ProviderError(format!(
                "unsupported bedrock embedding model '{}' -- expected an id starting with 'amazon.titan-embed' or 'cohere.embed'",
                parts.model_id
            )));
        }

        if is_titan && parts.input_type.is_some() {
            return Err(ProviderError(format!(
                "bedrock model '{}' is an Amazon Titan model, which has no input_type parameter -- drop the '#' qualifier",
                parts.model_id
            )));
        }

        let qualifier = if is_cohere {
            Some(parts.input_type.clone().unwrap_or_else(|| DEFAULT_COHERE_INPUT_TYPE.to_string()))
        } else {
            None
        };

        let canonical = match &qualifier {
            Some(input_type) => format!("{}/{}#{}", parts.region, parts.model_id, input_type),
            None => format!("{}/{}", parts.region, parts.model_id),
        };

        Ok(ResolvedModel { canonical, model_id: parts.model_id, endpoint_base, qualifier })
    }

    fn strategy_for(&self, resolved: &ResolvedModel) -> Result<&dyn TextWireStrategy, ProviderError> {
        if resolved.model_id.starts_with("amazon.titan-embed") {
            Ok(&self.titan)
        } else if resolved.model_id.starts_with("cohere.embed") {
            Ok(&self.cohere)
        } else {
            Err(ProviderError(format!(
                "unsupported bedrock embedding model '{}'",
                resolved.model_id
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn router() -> BedrockRouter {
        BedrockRouter::new(DEFAULT_BEDROCK_REGION, DEFAULT_BEDROCK_ENDPOINT_TEMPLATE)
    }

    #[test]
    fn bare_model_id_uses_the_default_region() {
        let r = router().resolve("amazon.titan-embed-text-v2:0").unwrap();
        assert_eq!(r.model_id, "amazon.titan-embed-text-v2:0");
        assert_eq!(r.endpoint_base, "https://bedrock-runtime.us-east-1.amazonaws.com");
        assert_eq!(r.canonical, "us-east-1/amazon.titan-embed-text-v2:0");
    }

    #[test]
    fn explicit_region_overrides_the_default_and_reaches_a_different_host() {
        let r = router().resolve("eu-west-1/amazon.titan-embed-text-v2:0").unwrap();
        assert_eq!(r.endpoint_base, "https://bedrock-runtime.eu-west-1.amazonaws.com");
        assert_eq!(r.canonical, "eu-west-1/amazon.titan-embed-text-v2:0");
    }

    #[test]
    fn a_bare_model_id_and_its_explicitly_qualified_form_collapse_to_one_cache_entry() {
        // Same target, two spellings -- must produce identical canonical form,
        // or the cache splits in two for no reason.
        let bare = router().resolve("cohere.embed-english-v3").unwrap();
        let qualified = router().resolve("us-east-1/cohere.embed-english-v3").unwrap();
        assert_eq!(bare.canonical, qualified.canonical);
    }

    #[test]
    fn the_same_model_in_two_regions_never_collapses_to_one_cache_entry() {
        let east = router().resolve("us-east-1/cohere.embed-english-v3").unwrap();
        let west = router().resolve("eu-west-1/cohere.embed-english-v3").unwrap();
        assert_ne!(
            east.canonical, west.canonical,
            "two regional deployments are two upstreams and must not share cached vectors"
        );
    }

    #[test]
    fn cohere_defaults_to_search_document_and_records_it_in_the_canonical_form() {
        let r = router().resolve("cohere.embed-english-v3").unwrap();
        assert_eq!(r.qualifier.as_deref(), Some("search_document"));
        assert_eq!(r.canonical, "us-east-1/cohere.embed-english-v3#search_document");
    }

    #[test]
    fn cohere_input_type_changes_the_canonical_form_so_query_and_document_vectors_never_collide() {
        let doc = router().resolve("cohere.embed-english-v3#search_document").unwrap();
        let query = router().resolve("cohere.embed-english-v3#search_query").unwrap();
        assert_ne!(
            doc.canonical, query.canonical,
            "input_type changes the vector, so it has to change the cache identity"
        );
        assert_eq!(query.qualifier.as_deref(), Some("search_query"));
    }

    #[test]
    fn titan_has_no_qualifier_and_rejects_one_rather_than_ignoring_it() {
        let r = router().resolve("amazon.titan-embed-text-v2:0").unwrap();
        assert_eq!(r.qualifier, None);

        let err = router().resolve("amazon.titan-embed-text-v2:0#search_query").unwrap_err();
        assert!(
            err.0.contains("no input_type parameter"),
            "silently ignoring a qualifier would let a caller believe it took effect: {}",
            err.0
        );
    }

    #[test]
    fn an_unknown_vendor_prefix_is_rejected_with_a_message_naming_the_supported_ones() {
        let err = router().resolve("meta.llama3-8b").unwrap_err();
        assert!(err.0.contains("amazon.titan-embed"), "{}", err.0);
        assert!(err.0.contains("cohere.embed"), "{}", err.0);
    }

    #[test]
    fn a_malformed_model_string_is_rejected() {
        assert!(router().resolve("").is_err());
        assert!(router().resolve("/cohere.embed-english-v3").is_err());
        assert!(router().resolve("us-east-1/").is_err());
        assert!(router().resolve("cohere.embed-english-v3#").is_err());
    }

    #[test]
    fn strategy_selection_follows_the_vendor_prefix_including_batch_size() {
        let r = router();
        let titan = r.resolve("amazon.titan-embed-text-v2:0").unwrap();
        assert_eq!(r.strategy_for(&titan).unwrap().max_batch(), 1);

        let cohere_v3 = r.resolve("cohere.embed-english-v3").unwrap();
        assert_eq!(r.strategy_for(&cohere_v3).unwrap().max_batch(), 96);

        let cohere_v4 = r.resolve("cohere.embed-v4:0").unwrap();
        assert_eq!(r.strategy_for(&cohere_v4).unwrap().max_batch(), 96);
    }

    #[test]
    fn an_endpoint_template_without_a_region_placeholder_is_used_verbatim() {
        // What makes httpmock testing and a PrivateLink endpoint both work.
        let r = BedrockRouter::new("us-east-1", "http://127.0.0.1:9999");
        assert_eq!(
            r.resolve("cohere.embed-english-v3").unwrap().endpoint_base,
            "http://127.0.0.1:9999"
        );
    }
}
