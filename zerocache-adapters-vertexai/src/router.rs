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
/// `location`, `project`, and `model_id` are further character-validated
/// since all three flow verbatim into the outbound request URL -- an
/// unvalidated value would be an SSRF vector, letting a caller-supplied
/// `model` string redirect the request to an arbitrary host/path.
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

    // All three fields are interpolated verbatim into the outbound request
    // URL (endpoint host and resource path for `location`/`project`, the
    // final path segment for `model_id`) -- reject anything containing
    // URL-structural characters rather than let a caller-supplied `model`
    // string smuggle a different host/path/query through, which is a real
    // SSRF vector, not a hypothetical one. Applied here, after both match
    // arms above have resolved their values, so neither the bare-model-id
    // form (with a malicious default) nor the explicit-location/project form
    // can bypass it.
    if !location.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-') {
        return Err(ProviderError(format!(
            "vertexai model '{model}' has an invalid location -- expected a GCP location like 'us-central1'"
        )));
    }

    if !project.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-') {
        return Err(ProviderError(format!(
            "vertexai model '{model}' has an invalid project -- expected a GCP project id like 'my-project'"
        )));
    }

    if model_id.contains('/') || model_id.contains('?') || model_id.contains('#') {
        return Err(ProviderError(format!(
            "vertexai model '{model}' has an invalid model id -- '/', '?', and '#' are not allowed"
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

#[cfg(test)]
mod tests {
    use super::*;

    fn router_with_project() -> VertexRouter {
        VertexRouter::new(
            Some("my-project".to_string()),
            DEFAULT_VERTEX_LOCATION,
            DEFAULT_VERTEX_ENDPOINT_TEMPLATE,
        )
    }

    fn router_without_project() -> VertexRouter {
        VertexRouter::new(None, DEFAULT_VERTEX_LOCATION, DEFAULT_VERTEX_ENDPOINT_TEMPLATE)
    }

    #[test]
    fn bare_model_id_uses_the_default_project_and_location() {
        let r = router_with_project().resolve("text-embedding-005").unwrap();
        assert_eq!(r.model_id, "text-embedding-005");
        assert_eq!(
            r.endpoint_base,
            "https://us-central1-aiplatform.googleapis.com/v1/projects/my-project/locations/us-central1/publishers/google/models"
        );
        assert_eq!(r.canonical, "us-central1/my-project/text-embedding-005");
    }

    #[test]
    fn a_fully_qualified_model_overrides_both_defaults() {
        let r = router_with_project()
            .resolve("europe-west4/other-project/text-embedding-005")
            .unwrap();
        assert_eq!(
            r.endpoint_base,
            "https://europe-west4-aiplatform.googleapis.com/v1/projects/other-project/locations/europe-west4/publishers/google/models"
        );
        assert_eq!(r.canonical, "europe-west4/other-project/text-embedding-005");
    }

    #[test]
    fn a_bare_model_id_and_its_explicitly_qualified_form_collapse_to_one_cache_entry() {
        let bare = router_with_project().resolve("text-embedding-005").unwrap();
        let qualified = router_with_project()
            .resolve("us-central1/my-project/text-embedding-005")
            .unwrap();
        assert_eq!(bare.canonical, qualified.canonical);
    }

    #[test]
    fn two_projects_never_collapse_to_one_cache_entry() {
        let a = router_with_project().resolve("us-central1/proj-a/text-embedding-005").unwrap();
        let b = router_with_project().resolve("us-central1/proj-b/text-embedding-005").unwrap();
        assert_ne!(
            a.canonical, b.canonical,
            "two projects are two billing and deployment boundaries and must not share cached vectors"
        );
    }

    #[test]
    fn a_missing_project_with_no_default_is_an_error_naming_both_fixes() {
        let err = router_without_project().resolve("text-embedding-005").unwrap_err();
        assert!(err.0.contains("ZEROCACHE_VERTEX_PROJECT"), "{}", err.0);
        assert!(err.0.contains("<location>/<project>/"), "{}", err.0);
    }

    #[test]
    fn task_type_changes_the_canonical_form_so_query_and_document_vectors_never_collide() {
        let doc = router_with_project()
            .resolve("text-embedding-005#RETRIEVAL_DOCUMENT")
            .unwrap();
        let query = router_with_project()
            .resolve("text-embedding-005#RETRIEVAL_QUERY")
            .unwrap();
        assert_ne!(doc.canonical, query.canonical);
        assert_eq!(query.qualifier.as_deref(), Some("RETRIEVAL_QUERY"));
    }

    #[test]
    fn an_omitted_task_type_stays_omitted_rather_than_being_defaulted_here() {
        let r = router_with_project().resolve("text-embedding-005").unwrap();
        assert_eq!(r.qualifier, None, "Google's own default must apply, not one invented by this adapter");
    }

    #[test]
    fn gemini_embedding_selects_the_single_instance_strategy() {
        let r = router_with_project();
        let gemini = r.resolve("gemini-embedding-001").unwrap();
        assert_eq!(
            r.strategy_for(&gemini).unwrap().max_batch(),
            1,
            "gemini-embedding-001 accepts exactly one input per request"
        );

        let standard = r.resolve("text-embedding-005").unwrap();
        assert_eq!(r.strategy_for(&standard).unwrap().max_batch(), 250);

        let multilingual = r.resolve("text-multilingual-embedding-002").unwrap();
        assert_eq!(r.strategy_for(&multilingual).unwrap().max_batch(), 250);
    }

    #[test]
    fn a_malformed_model_string_is_rejected() {
        let r = router_with_project();
        assert!(r.resolve("").is_err());
        assert!(r.resolve("my-project/text-embedding-005").is_err(), "two segments is ambiguous");
        assert!(r.resolve("a/b/c/d").is_err());
        assert!(r.resolve("us-central1//text-embedding-005").is_err());
        assert!(r.resolve("text-embedding-005#").is_err());
    }

    #[test]
    fn an_endpoint_template_without_a_location_placeholder_is_used_verbatim() {
        let r = VertexRouter::new(Some("p".to_string()), "us-central1", "http://127.0.0.1:9999");
        assert_eq!(
            r.resolve("text-embedding-005").unwrap().endpoint_base,
            "http://127.0.0.1:9999/v1/projects/p/locations/us-central1/publishers/google/models"
        );
    }
}
