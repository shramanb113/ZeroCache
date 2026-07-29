use zerocache_adapters_cloud::{CloudRouter, ResolvedModel, TextWireStrategy};
use zerocache_ports::ProviderError;

use crate::strategy::{AzureFoundryStrategy, AzureOpenAiV1Strategy};

/// Only the Foundry Models surface takes an api-version; the GA
/// `/openai/v1` path does not. Verified 2026-07-28.
pub const DEFAULT_AZURE_FOUNDRY_API_VERSION: &str = "2024-05-01-preview";

/// Model-string prefix that routes to the Foundry Models surface instead of
/// Azure OpenAI. Azure deployment names never contain `:`, so this is
/// unambiguous.
pub const FOUNDRY_MODEL_PREFIX: &str = "foundry:";

/// How the caller's forwarded credential is presented upstream.
///
/// Both are supported by both Azure surfaces. `Bearer` (an Entra ID access
/// token) is the default because Microsoft's own docs recommend it -- it
/// avoids a long-lived credential -- and because it matches what every other
/// Zerocache adapter already receives on the `Authorization` header, so a
/// caller changes nothing but the base URL.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AzureAuthMode {
    Bearer,
    ApiKey,
}

impl AzureAuthMode {
    pub fn header(&self, credential: &str) -> (&'static str, String) {
        match self {
            AzureAuthMode::Bearer => ("Authorization", format!("Bearer {credential}")),
            AzureAuthMode::ApiKey => ("api-key", credential.to_string()),
        }
    }

    /// Stable identifier folded into the canonical model form. Two auth modes
    /// against the same resource reach the same weights, so this is NOT part
    /// of the cache identity -- it exists only so the strategies can report
    /// what they did in tests.
    pub fn as_str(&self) -> &'static str {
        match self {
            AzureAuthMode::Bearer => "bearer",
            AzureAuthMode::ApiKey => "api-key",
        }
    }
}

pub struct AzureRouter {
    openai_base_url: String,
    /// `None` means no Foundry endpoint was configured, so a `foundry:` model
    /// string is a resolution error naming the env var that fixes it.
    foundry_base_url: Option<String>,
    foundry_api_version: String,
    openai: AzureOpenAiV1Strategy,
    foundry: AzureFoundryStrategy,
}

impl AzureRouter {
    pub fn new(
        openai_base_url: impl Into<String>,
        foundry_base_url: Option<String>,
        foundry_api_version: impl Into<String>,
        auth_mode: AzureAuthMode,
    ) -> Self {
        Self {
            openai_base_url: openai_base_url.into(),
            foundry_base_url,
            foundry_api_version: foundry_api_version.into(),
            openai: AzureOpenAiV1Strategy::new(auth_mode),
            foundry: AzureFoundryStrategy::new(auth_mode),
        }
    }
}

struct AzureModelParts {
    foundry: bool,
    deployment: String,
    input_type: Option<String>,
}

fn split_model(model: &str) -> Result<AzureModelParts, ProviderError> {
    let (head, input_type) = match model.split_once('#') {
        Some((h, q)) if !q.is_empty() => (h, Some(q.to_string())),
        Some((_, _)) => {
            return Err(ProviderError(format!(
                "azure model '{model}' has an empty '#' qualifier -- expected '#<input_type>' or no '#' at all"
            )))
        }
        None => (model, None),
    };

    let (foundry, deployment) = match head.strip_prefix(FOUNDRY_MODEL_PREFIX) {
        Some(rest) => (true, rest.to_string()),
        None => (false, head.to_string()),
    };

    if deployment.is_empty() {
        return Err(ProviderError(format!(
            "azure model '{model}' is malformed -- expected '<deployment>' or 'foundry:<model>', optionally suffixed with '#<input_type>'"
        )));
    }

    Ok(AzureModelParts { foundry, deployment, input_type })
}

impl CloudRouter for AzureRouter {
    fn resolve(&self, model: &str) -> Result<ResolvedModel, ProviderError> {
        let parts = split_model(model)?;

        if parts.foundry {
            let base = self.foundry_base_url.as_ref().ok_or_else(|| {
                ProviderError(format!(
                    "azure model '{model}' targets the Foundry Models surface but ZEROCACHE_AZURE_FOUNDRY_BASE_URL is unset"
                ))
            })?;

            // The api-version is part of the endpoint identity: two versions
            // of the same surface can return different vectors, and an
            // operator bumping it must get a cold cache, not stale hits.
            let endpoint_base = format!("{base}/models/embeddings?api-version={}", self.foundry_api_version);

            let canonical = match &parts.input_type {
                Some(input_type) => format!(
                    "foundry:{}@{}#{}",
                    parts.deployment, self.foundry_api_version, input_type
                ),
                None => format!("foundry:{}@{}", parts.deployment, self.foundry_api_version),
            };

            return Ok(ResolvedModel {
                canonical,
                model_id: parts.deployment,
                endpoint_base,
                qualifier: parts.input_type,
            });
        }

        if parts.input_type.is_some() {
            return Err(ProviderError(format!(
                "azure model '{model}' targets the Azure OpenAI surface, whose models do not accept an input_type -- drop the '#' qualifier, or prefix the model with '{FOUNDRY_MODEL_PREFIX}' to use the Foundry Models surface"
            )));
        }

        Ok(ResolvedModel {
            canonical: format!("openai:{}", parts.deployment),
            model_id: parts.deployment,
            endpoint_base: format!("{}/openai/v1/embeddings", self.openai_base_url),
            qualifier: None,
        })
    }

    fn strategy_for(&self, resolved: &ResolvedModel) -> Result<&dyn TextWireStrategy, ProviderError> {
        if resolved.canonical.starts_with("foundry:") {
            Ok(&self.foundry)
        } else {
            Ok(&self.openai)
        }
    }
}
