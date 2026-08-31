//! OpenAI-wire SSE framing and delta assembly for the streaming completion
//! cache. Pure: no I/O, no async. `zerocache-http/src/completion.rs` drives
//! it with bytes from a `StreamingChatCompletionProvider`.

use std::time::Duration;

use serde_json::{json, Value};
use zerocache_ports::CompletionUsage;

/// Inter-frame gap when replaying a cached stream, so a hit still reads as a
/// live stream in a UI. Not env-configurable.
pub const SSE_REPLAY_FRAME_DELAY: Duration = Duration::from_millis(3);

#[derive(Debug)]
pub enum SseEvent {
    Data(Value),
    Done,
    /// A `data:` payload that was not `[DONE]` and did not parse as JSON.
    Malformed(String),
}

/// One complete SSE frame: its exact bytes including the terminating blank
/// line, and the event it parsed to. `event` is `None` for a frame carrying
/// no `data:` field (a `:` comment / keep-alive) -- still forwarded verbatim.
#[derive(Debug)]
pub struct ParsedFrame {
    pub raw: Vec<u8>,
    pub event: Option<SseEvent>,
}

/// Push parser: fed arbitrary byte chunks, emits complete SSE frames. Only
/// `data:` fields are interpreted (OpenAI streams carry nothing else);
/// comment lines (`:`) and other fields yield `event: None`. Frame-granular
/// on purpose: the streaming tee forwards `raw` one frame at a time so it can
/// withhold exactly the injected usage frame without disturbing a boundary,
/// even when a read splits a frame or carries two.
pub struct SseFrameParser {
    buf: Vec<u8>,
}

impl SseFrameParser {
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }

    pub fn feed(&mut self, chunk: &[u8]) -> Vec<ParsedFrame> {
        self.buf.extend_from_slice(chunk);
        let mut out = Vec::new();
        // Frames are separated by a blank line ("\n\n"). Tolerate "\r\n\r\n".
        while let Some(pos) = find_frame_boundary(&self.buf) {
            let raw = self.buf[..pos.0 + pos.1].to_vec();
            let frame = self.buf[..pos.0].to_vec();
            self.buf.drain(..pos.0 + pos.1);
            out.push(ParsedFrame {
                event: parse_frame(&frame),
                raw,
            });
        }
        out
    }

    /// Flush a trailing frame with no terminating blank line (some servers
    /// drop it on a clean close). Returns its raw bytes too, so a byte-exact
    /// tee still forwards what the upstream sent.
    pub fn finish(&mut self) -> Option<ParsedFrame> {
        if self.buf.iter().any(|b| !b.is_ascii_whitespace()) {
            let frame = std::mem::take(&mut self.buf);
            let event = parse_frame(&frame);
            return Some(ParsedFrame { raw: frame, event });
        }
        self.buf.clear();
        None
    }
}

/// Returns `(boundary_start, boundary_len)` of the first `\n\n` / `\r\n\r\n`.
fn find_frame_boundary(buf: &[u8]) -> Option<(usize, usize)> {
    for i in 0..buf.len().saturating_sub(1) {
        if &buf[i..i + 2] == b"\n\n" {
            return Some((i, 2));
        }
        if i + 4 <= buf.len() && &buf[i..i + 4] == b"\r\n\r\n" {
            return Some((i, 4));
        }
    }
    None
}

fn parse_frame(frame: &[u8]) -> Option<SseEvent> {
    let text = String::from_utf8_lossy(frame);
    let mut data = String::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("data:") {
            if !data.is_empty() {
                data.push('\n');
            }
            data.push_str(rest.strip_prefix(' ').unwrap_or(rest));
        }
    }
    if data.is_empty() {
        return None;
    }
    if data.trim() == "[DONE]" {
        return Some(SseEvent::Done);
    }
    match serde_json::from_str::<Value>(&data) {
        Ok(v) => Some(SseEvent::Data(v)),
        Err(_) => Some(SseEvent::Malformed(data)),
    }
}

#[derive(Default)]
struct ChoiceAcc {
    role: Option<String>,
    content: String,
    finish_reason: Option<Value>,
    /// index -> (id, name, arguments)
    tool_calls: Vec<ToolCallAcc>,
}

#[derive(Default)]
struct ToolCallAcc {
    id: Option<String>,
    ty: Option<String>,
    name: Option<String>,
    arguments: String,
}

pub struct DeltaAssembler {
    id: Option<Value>,
    model: Option<Value>,
    created: Option<Value>,
    system_fingerprint: Option<Value>,
    choices: Vec<ChoiceAcc>,
    usage: CompletionUsage,
    saw_done: bool,
    saw_error: bool,
    saw_malformed: bool,
}

impl DeltaAssembler {
    pub fn new() -> Self {
        Self {
            id: None,
            model: None,
            created: None,
            system_fingerprint: None,
            choices: Vec::new(),
            usage: CompletionUsage::default(),
            saw_done: false,
            saw_error: false,
            saw_malformed: false,
        }
    }

    pub fn ingest(&mut self, event: &SseEvent) {
        let v = match event {
            SseEvent::Done => {
                self.saw_done = true;
                return;
            }
            SseEvent::Malformed(_) => {
                self.saw_malformed = true;
                return;
            }
            SseEvent::Data(v) => v,
        };

        if v.get("error").is_some() {
            self.saw_error = true;
            return;
        }

        take_top(&mut self.id, v.get("id"));
        take_top(&mut self.model, v.get("model"));
        take_top(&mut self.created, v.get("created"));
        take_top(&mut self.system_fingerprint, v.get("system_fingerprint"));

        if let Some(u) = v.get("usage").and_then(|u| u.as_object()) {
            let g = |k: &str| u.get(k).and_then(|n| n.as_u64()).unwrap_or(0) as u32;
            let (p, c, t) = (
                g("prompt_tokens"),
                g("completion_tokens"),
                g("total_tokens"),
            );
            if p + c + t > 0 {
                self.usage = CompletionUsage {
                    prompt_tokens: p,
                    completion_tokens: c,
                    total_tokens: t,
                };
            }
        }

        let Some(choices) = v.get("choices").and_then(|c| c.as_array()) else {
            return;
        };
        for ch in choices {
            let idx = ch.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
            if self.choices.len() <= idx {
                self.choices.resize_with(idx + 1, ChoiceAcc::default);
            }
            let acc = &mut self.choices[idx];
            if let Some(fr) = ch.get("finish_reason") {
                if !fr.is_null() {
                    acc.finish_reason = Some(fr.clone());
                }
            }
            let Some(delta) = ch.get("delta") else {
                continue;
            };
            if let Some(r) = delta.get("role").and_then(|r| r.as_str()) {
                acc.role.get_or_insert_with(|| r.to_string());
            }
            if let Some(c) = delta.get("content").and_then(|c| c.as_str()) {
                acc.content.push_str(c);
            }
            if let Some(tcs) = delta.get("tool_calls").and_then(|t| t.as_array()) {
                for tc in tcs {
                    let ti = tc.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
                    if acc.tool_calls.len() <= ti {
                        acc.tool_calls.resize_with(ti + 1, ToolCallAcc::default);
                    }
                    let t = &mut acc.tool_calls[ti];
                    if let Some(id) = tc.get("id").and_then(|x| x.as_str()) {
                        t.id.get_or_insert_with(|| id.to_string());
                    }
                    if let Some(ty) = tc.get("type").and_then(|x| x.as_str()) {
                        t.ty.get_or_insert_with(|| ty.to_string());
                    }
                    if let Some(f) = tc.get("function") {
                        if let Some(n) = f.get("name").and_then(|x| x.as_str()) {
                            t.name.get_or_insert_with(|| n.to_string());
                        }
                        if let Some(a) = f.get("arguments").and_then(|x| x.as_str()) {
                            t.arguments.push_str(a);
                        }
                    }
                }
            }
        }
    }

    pub fn finish(self) -> Assembled {
        let has_payload = self
            .choices
            .iter()
            .any(|c| !c.content.is_empty() || c.tool_calls.iter().any(|t| t.name.is_some()));
        let all_finished =
            !self.choices.is_empty() && self.choices.iter().all(|c| c.finish_reason.is_some());

        let completeness = if self.saw_error {
            Completeness::Incomplete("stream carried an error frame")
        } else if self.saw_malformed {
            Completeness::Incomplete("stream carried a malformed frame")
        } else if !self.saw_done && !all_finished {
            Completeness::Incomplete("stream ended without [DONE] or a finish_reason")
        } else if !has_payload {
            Completeness::Incomplete("assembled completion has no content or tool calls")
        } else {
            Completeness::Complete
        };

        let choices: Vec<Value> = self
            .choices
            .into_iter()
            .enumerate()
            .map(|(i, c)| {
                let mut message = serde_json::Map::new();
                message.insert(
                    "role".into(),
                    json!(c.role.unwrap_or_else(|| "assistant".into())),
                );
                message.insert("content".into(), json!(c.content));
                if !c.tool_calls.is_empty() {
                    let tcs: Vec<Value> = c
                        .tool_calls
                        .into_iter()
                        .enumerate()
                        .map(|(ti, t)| {
                            json!({
                                "index": ti,
                                "id": t.id.unwrap_or_default(),
                                "type": t.ty.unwrap_or_else(|| "function".into()),
                                "function": { "name": t.name.unwrap_or_default(), "arguments": t.arguments }
                            })
                        })
                        .collect();
                    message.insert("tool_calls".into(), json!(tcs));
                }
                json!({
                    "index": i,
                    "message": Value::Object(message),
                    "finish_reason": c.finish_reason.unwrap_or(Value::Null),
                })
            })
            .collect();

        let mut body = serde_json::Map::new();
        if let Some(v) = self.id {
            body.insert("id".into(), v);
        }
        body.insert("object".into(), json!("chat.completion"));
        if let Some(v) = self.created {
            body.insert("created".into(), v);
        }
        if let Some(v) = self.model {
            body.insert("model".into(), v);
        }
        if let Some(v) = self.system_fingerprint {
            body.insert("system_fingerprint".into(), v);
        }
        body.insert("choices".into(), json!(choices));
        body.insert(
            "usage".into(),
            json!({
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens,
            }),
        );

        Assembled {
            body: Value::Object(body),
            usage: self.usage,
            completeness,
        }
    }
}

fn take_top(slot: &mut Option<Value>, v: Option<&Value>) {
    if slot.is_none() {
        if let Some(v) = v {
            if !v.is_null() {
                *slot = Some(v.clone());
            }
        }
    }
}

pub struct Assembled {
    pub body: Value,
    pub usage: CompletionUsage,
    pub completeness: Completeness,
}

#[derive(Debug)]
pub enum Completeness {
    Complete,
    Incomplete(&'static str),
}

/// Synthesize SSE frames from an assembled body, for replaying an entry that
/// was stored by a `stream:false` miss (no `raw_sse`). Content faithful;
/// frame boundaries and ids are not claimed to match a real stream.
pub fn rechunk(body: &Value) -> Vec<Vec<u8>> {
    let mut frames = Vec::new();
    let empty = Vec::new();
    let choices = body
        .get("choices")
        .and_then(|c| c.as_array())
        .unwrap_or(&empty);
    let id = body.get("id").cloned().unwrap_or(json!("zerocache-replay"));
    let model = body.get("model").cloned().unwrap_or(json!(""));

    for (i, ch) in choices.iter().enumerate() {
        let msg = ch.get("message").cloned().unwrap_or(json!({}));
        let role = msg.get("role").cloned().unwrap_or(json!("assistant"));
        let mut delta = serde_json::Map::new();
        delta.insert("role".into(), role);
        if let Some(content) = msg.get("content") {
            delta.insert("content".into(), content.clone());
        }
        if let Some(tcs) = msg.get("tool_calls") {
            delta.insert("tool_calls".into(), tcs.clone());
        }
        frames.push(sse_line(&json!({
            "id": id, "object": "chat.completion.chunk", "model": model,
            "choices": [{ "index": i, "delta": Value::Object(delta), "finish_reason": Value::Null }]
        })));
        frames.push(sse_line(&json!({
            "id": id, "object": "chat.completion.chunk", "model": model,
            "choices": [{ "index": i, "delta": {}, "finish_reason": ch.get("finish_reason").cloned().unwrap_or(json!("stop")) }]
        })));
    }
    if let Some(usage) = body.get("usage") {
        frames.push(sse_line(&json!({
            "id": id, "object": "chat.completion.chunk", "model": model,
            "choices": [], "usage": usage
        })));
    }
    frames.push(b"data: [DONE]\n\n".to_vec());
    frames
}

fn sse_line(v: &Value) -> Vec<u8> {
    format!("data: {}\n\n", v).into_bytes()
}

/// Split stored raw SSE bytes into individual frames, each ending in `\n\n`,
/// for paced replay. A trailing partial (no blank line) is emitted as-is.
pub fn split_frames(raw: &[u8]) -> Vec<Vec<u8>> {
    let mut frames = Vec::new();
    let mut start = 0;
    let mut i = 0;
    while i + 1 < raw.len() {
        if &raw[i..i + 2] == b"\n\n" {
            frames.push(raw[start..i + 2].to_vec());
            start = i + 2;
            i += 2;
        } else {
            i += 1;
        }
    }
    if start < raw.len() {
        frames.push(raw[start..].to_vec());
    }
    frames
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn events(raw: &str) -> Vec<SseEvent> {
        let mut p = SseFrameParser::new();
        let mut out: Vec<SseEvent> = p
            .feed(raw.as_bytes())
            .into_iter()
            .filter_map(|f| f.event)
            .collect();
        out.extend(p.finish().and_then(|f| f.event));
        out
    }

    #[test]
    fn parser_reassembles_a_frame_split_across_two_feeds() {
        let mut p = SseFrameParser::new();
        let a = p.feed(b"data: {\"choices\":[{\"del");
        assert!(a.is_empty());
        let b = p.feed(b"ta\":{\"content\":\"hi\"}}]}\n\n");
        assert_eq!(b.len(), 1);
        assert_eq!(
            b[0].raw,
            b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n"
        );
        match &b[0].event {
            Some(SseEvent::Data(v)) => {
                assert_eq!(v["choices"][0]["delta"]["content"], "hi")
            }
            other => panic!("expected Data, got {other:?}"),
        }
    }

    #[test]
    fn parser_recognizes_the_done_sentinel() {
        let evs = events("data: {\"choices\":[]}\n\ndata: [DONE]\n\n");
        assert!(matches!(evs.last(), Some(SseEvent::Done)));
    }

    #[test]
    fn assembler_merges_content_deltas_across_choices() {
        let raw = "data: {\"id\":\"c1\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"he\"}}]}\n\n\
                   data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"llo\"},\"finish_reason\":\"stop\"}]}\n\n\
                   data: [DONE]\n\n";
        let mut a = DeltaAssembler::new();
        for ev in events(raw) {
            a.ingest(&ev);
        }
        let out = a.finish();
        assert!(matches!(out.completeness, Completeness::Complete));
        assert_eq!(out.body["choices"][0]["message"]["content"], "hello");
        assert_eq!(out.body["choices"][0]["message"]["role"], "assistant");
        assert_eq!(out.body["choices"][0]["finish_reason"], "stop");
        assert_eq!(out.body["id"], "c1");
    }

    #[test]
    fn assembler_merges_tool_call_deltas_by_index() {
        let raw = "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"t1\",\"function\":{\"name\":\"f\",\"arguments\":\"{\\\"a\\\":\"}}]}}]}\n\n\
                   data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"1}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n\
                   data: [DONE]\n\n";
        let mut a = DeltaAssembler::new();
        for ev in events(raw) {
            a.ingest(&ev);
        }
        let out = a.finish();
        assert!(matches!(out.completeness, Completeness::Complete));
        let tc = &out.body["choices"][0]["message"]["tool_calls"][0];
        assert_eq!(tc["id"], "t1");
        assert_eq!(tc["function"]["name"], "f");
        assert_eq!(tc["function"]["arguments"], "{\"a\":1}");
    }

    #[test]
    fn assembler_captures_a_trailing_usage_only_chunk() {
        let raw = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"x\"},\"finish_reason\":\"stop\"}]}\n\n\
                   data: {\"choices\":[],\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":3,\"total_tokens\":14}}\n\n\
                   data: [DONE]\n\n";
        let mut a = DeltaAssembler::new();
        for ev in events(raw) {
            a.ingest(&ev);
        }
        let out = a.finish();
        assert_eq!(out.usage.prompt_tokens, 11);
        assert_eq!(out.usage.completion_tokens, 3);
        assert_eq!(out.usage.total_tokens, 14);
        assert_eq!(out.body["usage"]["total_tokens"], 14);
    }

    #[test]
    fn no_terminator_is_incomplete() {
        let raw = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"}}]}\n\n";
        let mut a = DeltaAssembler::new();
        for ev in events(raw) {
            a.ingest(&ev);
        }
        assert!(matches!(
            a.finish().completeness,
            Completeness::Incomplete(_)
        ));
    }

    #[test]
    fn all_choices_finished_is_complete_even_without_done() {
        let raw = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"x\"},\"finish_reason\":\"stop\"}]}\n\n";
        let mut a = DeltaAssembler::new();
        for ev in events(raw) {
            a.ingest(&ev);
        }
        assert!(matches!(a.finish().completeness, Completeness::Complete));
    }

    #[test]
    fn an_error_frame_makes_it_incomplete() {
        let raw = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"x\"},\"finish_reason\":\"stop\"}]}\n\n\
                   data: {\"error\":{\"message\":\"boom\"}}\n\n\
                   data: [DONE]\n\n";
        let mut a = DeltaAssembler::new();
        for ev in events(raw) {
            a.ingest(&ev);
        }
        assert!(matches!(
            a.finish().completeness,
            Completeness::Incomplete(_)
        ));
    }

    #[test]
    fn a_malformed_data_frame_makes_it_incomplete() {
        let raw = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"x\"},\"finish_reason\":\"stop\"}]}\n\n\
                   data: {oops not json}\n\n\
                   data: [DONE]\n\n";
        let evs = events(raw);
        assert!(evs.iter().any(|e| matches!(e, SseEvent::Malformed(_))));
        let mut a = DeltaAssembler::new();
        for ev in &evs {
            a.ingest(ev);
        }
        assert!(matches!(
            a.finish().completeness,
            Completeness::Incomplete(_)
        ));
    }

    #[test]
    fn parser_tolerates_crlf_frame_delimiters() {
        let evs = events(
            "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\r\n\r\ndata: [DONE]\r\n\r\n",
        );
        match evs.first() {
            Some(SseEvent::Data(v)) => {
                assert_eq!(v["choices"][0]["delta"]["content"], "hi")
            }
            other => panic!("expected Data, got {other:?}"),
        }
        assert!(matches!(evs.last(), Some(SseEvent::Done)));
    }

    #[test]
    fn finish_flushes_an_unterminated_trailing_frame() {
        let mut p = SseFrameParser::new();
        assert!(p
            .feed(b"data: {\"choices\":[{\"delta\":{\"content\":\"tail\"}}]}")
            .is_empty());
        let flushed = p.finish().expect("a non-whitespace tail flushes");
        assert_eq!(
            flushed.raw,
            b"data: {\"choices\":[{\"delta\":{\"content\":\"tail\"}}]}"
        );
        match &flushed.event {
            Some(SseEvent::Data(v)) => {
                assert_eq!(v["choices"][0]["delta"]["content"], "tail")
            }
            other => panic!("expected Data, got {other:?}"),
        }
    }

    #[test]
    fn empty_content_and_no_tool_calls_is_incomplete() {
        let raw = "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":\"stop\"}]}\n\n\
                   data: [DONE]\n\n";
        let mut a = DeltaAssembler::new();
        for ev in events(raw) {
            a.ingest(&ev);
        }
        assert!(matches!(
            a.finish().completeness,
            Completeness::Incomplete(_)
        ));
    }

    #[test]
    fn tool_calls_with_empty_content_is_complete() {
        let raw = "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"tool_calls\":[{\"index\":0,\"id\":\"t1\",\"function\":{\"name\":\"f\",\"arguments\":\"{}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n\
                   data: [DONE]\n\n";
        let mut a = DeltaAssembler::new();
        for ev in events(raw) {
            a.ingest(&ev);
        }
        assert!(matches!(a.finish().completeness, Completeness::Complete));
    }

    #[test]
    fn rechunk_output_reassembles_to_the_same_message() {
        let body = json!({
            "id": "c9", "model": "m", "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello world"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
        });
        let frames = rechunk(&body);
        let joined: Vec<u8> = frames.concat();
        let mut a = DeltaAssembler::new();
        let mut p = SseFrameParser::new();
        for f in p.feed(&joined) {
            if let Some(ev) = &f.event {
                a.ingest(ev);
            }
        }
        let out = a.finish();
        assert!(matches!(out.completeness, Completeness::Complete));
        assert_eq!(out.body["choices"][0]["message"]["content"], "hello world");
    }

    #[test]
    fn feed_splits_a_chunk_carrying_usage_and_done_into_two_frames() {
        // The leak scenario: HTTP/2 delivers the trailing usage frame and
        // `[DONE]` in one read. Frame-granular parsing must still hand them
        // back separately, each with its own bytes, so the tee can withhold
        // exactly the usage frame.
        let mut p = SseFrameParser::new();
        let frames = p.feed(
            b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":2,\"total_tokens\":3}}\n\ndata: [DONE]\n\n",
        );
        assert_eq!(frames.len(), 2);
        assert!(frames[0].raw.ends_with(b"}}\n\n"));
        assert!(matches!(frames[0].event, Some(SseEvent::Data(_))));
        assert_eq!(frames[1].raw, b"data: [DONE]\n\n");
        assert!(matches!(frames[1].event, Some(SseEvent::Done)));
    }

    #[test]
    fn feed_holds_a_partial_trailing_frame_until_its_boundary_arrives() {
        // The corruption scenario: a read ends with a whole frame plus the
        // first bytes of the next. Only the complete frame is emitted; the
        // partial bytes stay buffered and are never forwarded mid-frame.
        let mut p = SseFrameParser::new();
        let first = p.feed(b"data: {\"choices\":[],\"usage\":{\"total_tokens\":3}}\n\ndata: [DO");
        assert_eq!(first.len(), 1);
        assert!(matches!(first[0].event, Some(SseEvent::Data(_))));
        let second = p.feed(b"NE]\n\n");
        assert_eq!(second.len(), 1);
        assert_eq!(second[0].raw, b"data: [DONE]\n\n");
        assert!(matches!(second[0].event, Some(SseEvent::Done)));
    }

    #[test]
    fn split_frames_preserves_each_data_line_with_its_terminator() {
        let raw = b"data: {\"a\":1}\n\ndata: [DONE]\n\n";
        let frames = split_frames(raw);
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0], b"data: {\"a\":1}\n\n");
        assert_eq!(frames[1], b"data: [DONE]\n\n");
    }
}
