use zerocache_adapters_cloud::{CloudRouter, ResolvedModel, TextWireStrategy};
use zerocache_ports::ProviderError;

use crate::strategy::VertexPredictStrategy;

pub const DEFAULT_VERTEX_LOCATION: &str = "us-central1";

/// `{location}` is substituted per request. A template with no placeholder is
/// used verbatim, which is what makes httpmock testing possible and also
/// covers a Private Service Connect endpoint that is not region-templated.
pub const DEFAULT_VERTEX_ENDPOINT_TEMPLATE: &str = "https://{location}-aiplatform.googleapis.com";

/// Google documents 250 instances per request for the text-embedding models.
const STANDARD_MAX_BATCH: usize = 250;

/// gemini-embedding-001 accepts exactly one input text per request -- it is
/// also excluded from the batch-prediction API entirely. Verified 2026-07-28.
const GEMINI_MAX_BATCH: usize = 1;

pub struct VertexRouter {
    /// `None` means the caller's `model` string must carry `location/project/`
    /// itself. There is no sane default project, unlike location.
    default_project: Option<String>,
    default_location: String,
    endpoint_template: String,
    standard: VertexPredictStrategy,
    gemini: VertexPredictStrategy,
}

impl VertexRouter {
    pub fn new(
        default_project: Option<String>,
        default_location: impl Into<String>,
        endpoint_template: impl Into<String>,
    ) -> Self {
        Self {
            default_project,
            default_location: default_location.into(),
            endpoint_template: endpoint_template.into(),
            standard: VertexPredictStrategy::new(STANDARD_MAX_BATCH),
            gemini: VertexPredictStrategy::new(GEMINI_MAX_BATCH),
        }
    }
}

/// One caller `model` string split into its parts.
///
/// Grammar: `[<location>/<project>/]<modelId>[#<task_type>]`
///
/// Location and project travel together -- a project without a location is
/// ambiguous, since the location appears in both the hostname and the resource
/// path. Splitting is unambiguous: Vertex model IDs never contain `/` or `#`.
struct VertexModelParts {
    location: String,
    project: String,
    model_id: String,
    task_type: Option<String>,
}

fn split_model(
    model: &str,
    default_project: Option<&str>,
    default_location: &str,
) -> Result<VertexModelParts, ProviderError> {
    let (head, task_type) = match model.split_once('#') {
        Some((h, q)) if !q.is_empty() => (h, Some(q.to_string())),
        Some((_, _)) => {
            return Err(ProviderError(format!(
                "vertexai model '{model}' has an empty '#' qualifier -- expected '#<task_type>' or no '#' at all"
            )))
        }
        None => (model, None),
    };

    let segments: Vec<&str> = head.split('/').collect();
    let (location, project, model_id) = match segments.as_slice() {
        [model_id] => {
            let project = default_project.ok_or_else(|| {
                ProviderError(format!(
                    "vertexai model '{model}' has no project and ZEROCACHE_VERTEX_PROJECT is unset -- either set that env var or send the model as '<location>/<project>/{model_id}'"
                ))
            })?;
            (default_location.to_string(), project.to_string(), (*model_id).to_string())
        }
        [location, project, model_id] => {
            ((*location).to_string(), (*project).to_string(), (*model_id).to_string())
        }
        _ => {
            return Err(ProviderError(format!(
                "vertexai model '{model}' is malformed -- expected '<modelId>' or '<location>/<project>/<modelId>', optionally suffixed with '#<task_type>'"
            )))
        }
    };

    if location.is_empty() || project.is_empty() || model_id.is_empty() {
        return Err(ProviderError(format!(
            "vertexai model '{model}' has an empty location, project, or model id"
        )));
    }

    Ok(VertexModelParts { location, project, model_id, task_type })
}

impl CloudRouter for VertexRouter {
    fn resolve(&self, model: &str) -> Result<ResolvedModel, ProviderError> {
        let parts = split_model(model, self.default_project.as_deref(), &self.default_location)?;

        let host = self.endpoint_template.replace("{location}", &parts.location);
        // The full resource path is folded into endpoint_base rather than
        // rebuilt in the strategy: project and location are part of *which
        // upstream*, so putting them here is what gets them into cache_scope.
        let endpoint_base = format!(
            "{host}/v1/projects/{}/locations/{}/publishers/google/models",
            parts.project, parts.location
        );

        let canonical = match &parts.task_type {
            Some(task_type) => format!(
                "{}/{}/{}#{}",
                parts.location, parts.project, parts.model_id, task_type
            ),
            None => format!("{}/{}/{}", parts.location, parts.project, parts.model_id),
        };

        Ok(ResolvedModel {
            canonical,
            model_id: parts.model_id,
            endpoint_base,
            qualifier: parts.task_type,
        })
    }

    fn strategy_for(&self, resolved: &ResolvedModel) -> Result<&dyn TextWireStrategy, ProviderError> {
        // Same wire shape either way -- only Google's per-model input limit
        // differs, so these are two instances of one strategy type, not two
        // strategy types.
        if resolved.model_id.starts_with("gemini-embedding") {
            Ok(&self.gemini)
        } else {
            Ok(&self.standard)
        }
    }
}
