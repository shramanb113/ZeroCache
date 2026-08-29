//! Serves the Astro-built savings dashboard, embedded into the binary at
//! compile time from `dashboard/dist` (rebuild with `npm run build` in that
//! directory after changing anything under `dashboard/src`). Reachable at
//! `GET /dashboard` on the same origin as `/metrics`, which it polls.

use axum::{
    extract::Path,
    http::{header, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
};
use include_dir::{include_dir, Dir};

static DIST: Dir<'static> = include_dir!("$CARGO_MANIFEST_DIR/../dashboard/dist");

fn content_type(path: &str) -> &'static str {
    match path.rsplit('.').next() {
        Some("html") => "text/html; charset=utf-8",
        Some("js") | Some("mjs") => "text/javascript; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("json") => "application/json",
        Some("svg") => "image/svg+xml",
        Some("woff2") => "font/woff2",
        Some("woff") => "font/woff",
        Some("png") => "image/png",
        Some("ico") => "image/x-icon",
        _ => "application/octet-stream",
    }
}

fn serve(rel: &str) -> Response {
    let rel = rel.trim_start_matches('/');
    let rel = if rel.is_empty() { "index.html" } else { rel };
    match DIST.get_file(rel) {
        Some(file) => (
            [(
                header::CONTENT_TYPE,
                HeaderValue::from_static(content_type(rel)),
            )],
            file.contents(),
        )
            .into_response(),
        None => (StatusCode::NOT_FOUND, "not found").into_response(),
    }
}

pub async fn index() -> Response {
    serve("index.html")
}

pub async fn asset(Path(path): Path<String>) -> Response {
    serve(&path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_js() -> String {
        fn walk(dir: &Dir, out: &mut String) {
            for f in dir.files() {
                if f.path().extension().and_then(|e| e.to_str()) == Some("js") {
                    out.push_str(f.contents_utf8().unwrap_or(""));
                }
            }
            for d in dir.dirs() {
                walk(d, out);
            }
        }
        let mut s = String::new();
        walk(&DIST, &mut s);
        s
    }

    #[test]
    fn dist_is_embedded_with_an_index() {
        assert!(
            DIST.get_file("index.html").is_some(),
            "dashboard/dist/index.html missing -- run `npm run build` in dashboard/"
        );
    }

    #[test]
    fn bundle_references_every_metric_name_it_parses() {
        // The dashboard reads these families out of /metrics text. If a counter
        // is renamed in app.rs this test fails instead of the dashboard
        // silently showing zeros.
        let js = all_js();
        for name in [
            "zerocache_completion_cache_hits_total",
            "zerocache_completion_cache_misses_total",
            "zerocache_completion_semantic_hits_total",
            "zerocache_completion_prompt_tokens_saved_total",
            "zerocache_completion_completion_tokens_saved_total",
            "zerocache_cache_hits_total",
            "zerocache_cache_misses_total",
            "zerocache_provider_prompt_tokens_total",
        ] {
            assert!(
                js.contains(name),
                "dashboard bundle no longer references {name}"
            );
        }
    }

    #[test]
    fn assets_are_served_with_a_sensible_content_type() {
        assert_eq!(content_type("index.html"), "text/html; charset=utf-8");
        assert_eq!(
            content_type("_astro/x.js"),
            "text/javascript; charset=utf-8"
        );
        assert_eq!(content_type("weird"), "application/octet-stream");
    }
}
