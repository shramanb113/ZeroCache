//! `impl CompletionVectorStore for RedisStore` -- the multi-replica semantic
//! index change-feed. One global Redis Stream, `zerocache:semantic:events`:
//! `insert` -> `XADD op=put`, `delete` -> `XADD op=del`, `load_all` /
//! `changes_since` -> `XRANGE` folded by `exact_key` (last op wins).

use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use base64::Engine;
use redis::streams::{StreamId, StreamRangeReply, StreamReadReply};
use zerocache_core::CacheKey;
use zerocache_ports::{CompletionVectorStore, StoreError, VectorChanges, VectorRecord};

use crate::{apply_socket_timeouts, RedisStore, STORE_TIMEOUT};

const SEMANTIC_STREAM_KEY: &str = "zerocache:semantic:events";
const XRANGE_PAGE: usize = 1000;
// `XREAD BLOCK` page size -- also the drain rate when `changes_blocking` is
// catching up a backlog (each call returns one page, the poll loop calls
// again immediately while entries remain, then blocks once caught up).
const XREAD_PAGE: usize = 1000;
// Socket read-timeout headroom over the requested block, so the OS-level
// timeout never fires before Redis returns the `XREAD BLOCK`.
const BLOCK_SOCKET_SLACK: Duration = Duration::from_secs(2);

/// One key's latest state within a folded range: an upsert record, or a
/// tombstone carrying the scope hash the delete must land on.
enum Slot {
    Up(VectorRecord),
    Del([u8; 32]),
}

/// Drain a folded `exact_key -> Slot` map into the `VectorChanges` shape.
fn drain_slots(slots: HashMap<CacheKey, Slot>) -> (Vec<VectorRecord>, Vec<(CacheKey, [u8; 32])>) {
    let mut upserts = Vec::new();
    let mut deletes = Vec::new();
    for (key, slot) in slots {
        match slot {
            Slot::Up(r) => upserts.push(r),
            Slot::Del(s) => deletes.push((key, s)),
        }
    }
    (upserts, deletes)
}

fn to_hex(b: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for x in b {
        s.push_str(&format!("{x:02x}"));
    }
    s
}

fn from_hex(s: &str) -> Option<[u8; 32]> {
    if s.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(s.get(i * 2..i * 2 + 2)?, 16).ok()?;
    }
    Some(out)
}

fn encode_vector_b64(v: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for f in v {
        bytes.extend_from_slice(&f.to_le_bytes());
    }
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

fn decode_vector_b64(s: &str) -> Option<Vec<f32>> {
    let bytes = base64::engine::general_purpose::STANDARD.decode(s).ok()?;
    if bytes.is_empty() || bytes.len() % 4 != 0 {
        return None;
    }
    Some(
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect(),
    )
}

fn field(id: &StreamId, key: &str) -> Option<String> {
    match id.map.get(key)? {
        redis::Value::BulkString(b) => String::from_utf8(b.clone()).ok(),
        redis::Value::SimpleString(s) => Some(s.clone()),
        _ => None,
    }
}

enum Parsed {
    Put(VectorRecord),
    Del { key: CacheKey, scope: [u8; 32] },
}

fn parse_entry(id: &StreamId) -> Option<Parsed> {
    match field(id, "op")?.as_str() {
        "put" => {
            let exact_key = CacheKey::from_bytes(from_hex(&field(id, "k")?)?);
            let scope_hash = from_hex(&field(id, "s")?)?;
            let coarse_key_hash = from_hex(&field(id, "c")?)?;
            let index_version = field(id, "iv")?.parse::<u8>().ok()?;
            let vector = decode_vector_b64(&field(id, "v")?)?;
            Some(Parsed::Put(VectorRecord {
                exact_key,
                scope_hash,
                coarse_key_hash,
                index_version,
                vector,
            }))
        }
        "del" => Some(Parsed::Del {
            key: CacheKey::from_bytes(from_hex(&field(id, "k")?)?),
            scope: from_hex(&field(id, "s")?)?,
        }),
        _ => None,
    }
}

impl RedisStore {
    fn xrange_fold(&self, cursor: Option<String>) -> Result<VectorChanges, StoreError> {
        let mut conn = self.pool.get().map_err(|e| StoreError(e.to_string()))?;
        apply_socket_timeouts(&conn)?;

        let mut start = match &cursor {
            None => "-".to_string(),
            Some(id) => format!("({id}"),
        };
        let mut last_seen: Option<String> = None;
        let mut slots: HashMap<CacheKey, Slot> = HashMap::new();
        let mut malformed = 0usize;

        loop {
            let reply: StreamRangeReply = redis::cmd("XRANGE")
                .arg(SEMANTIC_STREAM_KEY)
                .arg(&start)
                .arg("+")
                .arg("COUNT")
                .arg(XRANGE_PAGE)
                .query(&mut *conn)
                .map_err(|e| StoreError(e.to_string()))?;
            let n = reply.ids.len();
            for entry in reply.ids {
                last_seen = Some(entry.id.clone());
                match parse_entry(&entry) {
                    Some(Parsed::Put(r)) => {
                        slots.insert(r.exact_key, Slot::Up(r));
                    }
                    Some(Parsed::Del { key, scope }) => {
                        slots.insert(key, Slot::Del(scope));
                    }
                    None => malformed += 1,
                }
            }
            if n < XRANGE_PAGE {
                break;
            }
            start = format!("({}", last_seen.as_ref().unwrap());
        }

        if malformed > 0 {
            tracing::warn!(
                "semantic index stream: skipped {malformed} malformed/unknown entr{} in one changes_since call",
                if malformed == 1 { "y" } else { "ies" }
            );
        }

        let (upserts, deletes) = drain_slots(slots);
        // Redis must always hand back a Some cursor so zerocache-http spawns
        // the poll task even on a cold/empty stream. "0-0" == "from the start".
        let out_cursor = last_seen.or(cursor).or_else(|| Some("0-0".to_string()));
        Ok(VectorChanges {
            upserts,
            deletes,
            cursor: out_cursor,
        })
    }

    /// One `XREAD BLOCK` page after `cursor`, folded last-op-wins. Returns
    /// immediately while a backlog remains (blocks only once caught up), so the
    /// poll loop needs no interval ticker -- this call is the pacing.
    fn xread_blocking(
        &self,
        cursor: String,
        timeout: Duration,
    ) -> Result<VectorChanges, StoreError> {
        // A dedicated connection, not the r2d2 pool: this read parks for up to
        // `timeout` (bounded to <= 60s by zerocache-http's clamp, so never the
        // infinite `BLOCK 0`), and its socket timeout must outlast the block.
        let mut conn = self
            .client
            .get_connection()
            .map_err(|e| StoreError(e.to_string()))?;
        conn.set_read_timeout(Some(timeout + BLOCK_SOCKET_SLACK))
            .map_err(|e| StoreError(e.to_string()))?;
        conn.set_write_timeout(Some(STORE_TIMEOUT))
            .map_err(|e| StoreError(e.to_string()))?;

        let reply: Option<StreamReadReply> = redis::cmd("XREAD")
            .arg("BLOCK")
            .arg(timeout.as_millis() as u64)
            .arg("COUNT")
            .arg(XREAD_PAGE)
            .arg("STREAMS")
            .arg(SEMANTIC_STREAM_KEY)
            .arg(&cursor)
            .query(&mut conn)
            .map_err(|e| StoreError(e.to_string()))?;

        let mut slots: HashMap<CacheKey, Slot> = HashMap::new();
        let mut last_seen: Option<String> = None;
        let mut malformed = 0usize;
        for entry in reply.into_iter().flat_map(|r| r.keys).flat_map(|k| k.ids) {
            last_seen = Some(entry.id.clone());
            match parse_entry(&entry) {
                Some(Parsed::Put(r)) => {
                    slots.insert(r.exact_key, Slot::Up(r));
                }
                Some(Parsed::Del { key, scope }) => {
                    slots.insert(key, Slot::Del(scope));
                }
                None => malformed += 1,
            }
        }
        if malformed > 0 {
            tracing::warn!(
                "semantic index stream: skipped {malformed} malformed/unknown entr{} in one changes_blocking call",
                if malformed == 1 { "y" } else { "ies" }
            );
        }

        let (upserts, deletes) = drain_slots(slots);
        Ok(VectorChanges {
            upserts,
            deletes,
            cursor: Some(last_seen.unwrap_or(cursor)),
        })
    }
}

impl CompletionVectorStore for RedisStore {
    fn insert(&self, record: VectorRecord) -> Result<(), StoreError> {
        let mut conn = self.pool.get().map_err(|e| StoreError(e.to_string()))?;
        apply_socket_timeouts(&conn)?;
        redis::cmd("XADD")
            .arg(SEMANTIC_STREAM_KEY)
            .arg("MAXLEN")
            .arg("~")
            .arg(self.semantic_index_maxlen)
            .arg("*")
            .arg("op")
            .arg("put")
            .arg("k")
            .arg(to_hex(record.exact_key.as_bytes()))
            .arg("s")
            .arg(to_hex(&record.scope_hash))
            .arg("c")
            .arg(to_hex(&record.coarse_key_hash))
            .arg("iv")
            .arg(record.index_version.to_string())
            .arg("v")
            .arg(encode_vector_b64(&record.vector))
            .query::<String>(&mut *conn)
            .map_err(|e| StoreError(e.to_string()))?;

        // Opportunistic time-expiry: default stream IDs are ms timestamps, so
        // MINID ~ <now - ttl> gives the feed the same expiry the completion
        // blob store already has. Best-effort -- never fail the insert.
        if let Some(ttl) = self.ttl {
            let now_ms = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis();
            let min_id = now_ms.saturating_sub(ttl.as_millis());
            let _ = redis::cmd("XTRIM")
                .arg(SEMANTIC_STREAM_KEY)
                .arg("MINID")
                .arg("~")
                .arg(min_id.to_string())
                .query::<i64>(&mut *conn);
        }
        Ok(())
    }

    fn delete(&self, exact_key: &CacheKey, scope_hash: &[u8; 32]) -> Result<(), StoreError> {
        let mut conn = self.pool.get().map_err(|e| StoreError(e.to_string()))?;
        apply_socket_timeouts(&conn)?;
        redis::cmd("XADD")
            .arg(SEMANTIC_STREAM_KEY)
            .arg("MAXLEN")
            .arg("~")
            .arg(self.semantic_index_maxlen)
            .arg("*")
            .arg("op")
            .arg("del")
            .arg("k")
            .arg(to_hex(exact_key.as_bytes()))
            .arg("s")
            .arg(to_hex(scope_hash))
            .query::<String>(&mut *conn)
            .map_err(|e| StoreError(e.to_string()))?;
        Ok(())
    }

    fn load_all(&self) -> Result<Vec<VectorRecord>, StoreError> {
        Ok(self.xrange_fold(None)?.upserts)
    }

    fn changes_since(&self, cursor: Option<String>) -> Result<VectorChanges, StoreError> {
        self.xrange_fold(cursor)
    }

    fn changes_blocking(
        &self,
        cursor: String,
        timeout: Duration,
    ) -> Result<VectorChanges, StoreError> {
        self.xread_blocking(cursor, timeout)
    }
}

#[cfg(test)]
mod live_redis_tests {
    use testcontainers_modules::{
        redis::{Redis, REDIS_PORT},
        testcontainers::{runners::SyncRunner, Container, ImageExt},
    };

    use super::*;

    // testcontainers-modules pins the `redis` image to 5.0, which predates
    // exclusive XRANGE ranges (`(<id>`, Redis 6.2+) that changes_since relies
    // on. Pin a modern image for this module only.
    const REDIS_IMAGE_TAG: &str = "7-alpine";

    fn start_redis() -> (Container<Redis>, String) {
        let container = Redis::default()
            .with_tag(REDIS_IMAGE_TAG)
            .start()
            .expect("failed to start Redis testcontainer -- is Docker running?");
        let host = container.get_host().expect("host");
        let port = container.get_host_port_ipv4(REDIS_PORT).expect("port");
        (container, format!("redis://{host}:{port}"))
    }

    fn rec(n: u8, iv: u8) -> VectorRecord {
        VectorRecord {
            exact_key: CacheKey::from_bytes([n; 32]),
            scope_hash: [n.wrapping_add(1); 32],
            coarse_key_hash: [n.wrapping_add(2); 32],
            index_version: iv,
            vector: (0..384).map(|i| (i as f32) * 0.01 + n as f32).collect(),
        }
    }

    #[test]
    #[ignore]
    fn insert_then_changes_since_none_round_trips_every_field() {
        let (_c, url) = start_redis();
        let store = RedisStore::connect(&url, None).unwrap();
        let r = rec(1, 1);
        store.insert(r.clone()).unwrap();

        let out = store.changes_since(None).unwrap();
        assert_eq!(out.upserts.len(), 1);
        let got = &out.upserts[0];
        assert_eq!(got.exact_key, r.exact_key);
        assert_eq!(got.scope_hash, r.scope_hash);
        assert_eq!(got.coarse_key_hash, r.coarse_key_hash);
        assert_eq!(got.index_version, 1);
        assert_eq!(got.vector, r.vector);
        assert!(out.cursor.is_some());
    }

    #[test]
    #[ignore]
    fn changes_since_cursor_yields_only_newer_entries() {
        let (_c, url) = start_redis();
        let store = RedisStore::connect(&url, None).unwrap();
        store.insert(rec(1, 1)).unwrap();
        let cursor = store.changes_since(None).unwrap().cursor;
        store.insert(rec(2, 1)).unwrap();

        let out = store.changes_since(cursor).unwrap();
        assert_eq!(out.upserts.len(), 1);
        assert_eq!(out.upserts[0].exact_key, CacheKey::from_bytes([2u8; 32]));
    }

    #[test]
    #[ignore]
    fn delete_surfaces_in_deletes_not_upserts_and_wins_the_fold() {
        let (_c, url) = start_redis();
        let store = RedisStore::connect(&url, None).unwrap();
        let r = rec(3, 1);
        store.insert(r.clone()).unwrap();
        store.delete(&r.exact_key, &r.scope_hash).unwrap();

        let out = store.changes_since(None).unwrap();
        assert!(
            out.upserts.is_empty(),
            "a put-then-del must not appear as an upsert"
        );
        assert_eq!(out.deletes, vec![(r.exact_key, r.scope_hash)]);
    }

    #[test]
    #[ignore]
    fn empty_range_returns_the_passed_in_cursor_unchanged() {
        let (_c, url) = start_redis();
        let store = RedisStore::connect(&url, None).unwrap();
        store.insert(rec(1, 1)).unwrap();
        let cursor = store.changes_since(None).unwrap().cursor.unwrap();

        let out = store.changes_since(Some(cursor.clone())).unwrap();
        assert!(out.upserts.is_empty() && out.deletes.is_empty());
        assert_eq!(out.cursor, Some(cursor));
    }

    #[test]
    #[ignore]
    fn maxlen_caps_the_stream() {
        let (_c, url) = start_redis();
        let store = RedisStore::connect(&url, None)
            .unwrap()
            .with_semantic_index_maxlen(100);
        for i in 0..400u16 {
            store.insert(rec(i as u8, 1)).unwrap();
        }
        let mut conn = store.pool.get().unwrap();
        let len: usize = redis::cmd("XLEN")
            .arg(SEMANTIC_STREAM_KEY)
            .query(&mut *conn)
            .unwrap();
        assert!(len < 400, "MAXLEN ~ should have trimmed; got {len}");
    }

    #[test]
    #[ignore]
    fn changes_blocking_wakes_for_an_entry_added_after_the_call_starts() {
        let (_c, url) = start_redis();
        let store = RedisStore::connect(&url, None).unwrap();
        store.insert(rec(1, 1)).unwrap();
        let cursor = store.changes_since(None).unwrap().cursor.unwrap();

        let writer_url = url.clone();
        let writer = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(300));
            RedisStore::connect(&writer_url, None)
                .unwrap()
                .insert(rec(2, 1))
                .unwrap();
        });

        let started = std::time::Instant::now();
        let out = store
            .changes_blocking(cursor, std::time::Duration::from_secs(10))
            .unwrap();
        let waited = started.elapsed();
        writer.join().unwrap();

        assert_eq!(out.upserts.len(), 1);
        assert_eq!(out.upserts[0].exact_key, CacheKey::from_bytes([2u8; 32]));
        assert!(
            waited >= std::time::Duration::from_millis(250),
            "must have blocked until the writer inserted, waited {waited:?}"
        );
        assert!(
            waited < std::time::Duration::from_secs(9),
            "must have returned on the entry, not the timeout, waited {waited:?}"
        );
    }

    #[test]
    #[ignore]
    fn changes_blocking_times_out_empty_with_the_cursor_unchanged() {
        let (_c, url) = start_redis();
        let store = RedisStore::connect(&url, None).unwrap();
        store.insert(rec(1, 1)).unwrap();
        let cursor = store.changes_since(None).unwrap().cursor.unwrap();

        let started = std::time::Instant::now();
        let out = store
            .changes_blocking(cursor.clone(), std::time::Duration::from_millis(400))
            .unwrap();
        assert!(started.elapsed() >= std::time::Duration::from_millis(350));
        assert!(out.upserts.is_empty() && out.deletes.is_empty());
        assert_eq!(out.cursor, Some(cursor));
    }

    #[test]
    #[ignore]
    fn changes_blocking_from_zero_drains_the_backlog_without_blocking() {
        let (_c, url) = start_redis();
        let store = RedisStore::connect(&url, None).unwrap();
        for i in 1..=3u8 {
            store.insert(rec(i, 1)).unwrap();
        }
        let started = std::time::Instant::now();
        let out = store
            .changes_blocking("0-0".to_string(), std::time::Duration::from_secs(10))
            .unwrap();
        assert!(
            started.elapsed() < std::time::Duration::from_secs(2),
            "existing entries must return immediately, not after the block"
        );
        assert_eq!(out.upserts.len(), 3);
        assert!(out.cursor.is_some() && out.cursor.as_deref() != Some("0-0"));
    }

    #[test]
    #[ignore]
    fn a_future_index_version_round_trips_verbatim() {
        // The adapter never filters on index_version -- zerocache-http does,
        // at replay. This mirrors SledStore::load_all.
        let (_c, url) = start_redis();
        let store = RedisStore::connect(&url, None).unwrap();
        store.insert(rec(9, 2)).unwrap();
        let out = store.changes_since(None).unwrap();
        assert_eq!(out.upserts.len(), 1);
        assert_eq!(out.upserts[0].index_version, 2);
    }
}
