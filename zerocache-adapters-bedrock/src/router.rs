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
