use crate::CacheKey;

pub struct Reconciled {
    pub hits: Vec<(usize, Vec<f32>)>,
    pub misses: Vec<(usize, CacheKey)>,
}

/// Splits a batch of keys into hits and misses via `lookup`, keeping each
/// entry's original index so the caller can reassemble the response in order.
/// Generic over the lookup's error type `E` so core stays ignorant of what
/// kind of I/O backs it — a lookup failure aborts reconciliation rather than
/// being silently treated as a miss.
pub fn reconcile<E>(
    keys: &[CacheKey],
    lookup: impl Fn(&CacheKey) -> Result<Option<Vec<f32>>, E>,
) -> Result<Reconciled, E> {
    let mut hits = Vec::new();
    let mut misses = Vec::new();

    for (index, key) in keys.iter().enumerate() {
        match lookup(key)? {
            Some(vector) => hits.push((index, vector)),
            None => misses.push((index, *key)),
        }
    }

    Ok(Reconciled { hits, misses })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_hits_and_misses_preserving_index() {
        let hit_key = CacheKey::derive("m", "v1", "cached");
        let miss_key = CacheKey::derive("m", "v1", "not cached");
        let keys = [miss_key, hit_key, miss_key];

        let result = reconcile(&keys, |k| {
            Ok::<_, std::convert::Infallible>(if *k == hit_key { Some(vec![1.0, 2.0]) } else { None })
        })
        .unwrap();

        assert_eq!(result.hits, vec![(1, vec![1.0, 2.0])]);
        assert_eq!(
            result.misses.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
            vec![0, 2]
        );
    }

    #[test]
    fn all_miss_when_store_is_empty() {
        let keys = [CacheKey::derive("m", "v1", "a"), CacheKey::derive("m", "v1", "b")];
        let result = reconcile(&keys, |_| Ok::<_, std::convert::Infallible>(None)).unwrap();
        assert_eq!(result.hits.len(), 0);
        assert_eq!(result.misses.len(), 2);
    }

    #[test]
    fn aborts_and_propagates_on_lookup_error() {
        let keys = [CacheKey::derive("m", "v1", "a")];
        let result = reconcile(&keys, |_| Err("store unavailable"));
        assert_eq!(result.err(), Some("store unavailable"));
    }
}
