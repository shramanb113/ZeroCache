use crate::CacheKey;

pub struct Reconciled {
    pub hits: Vec<(usize, Vec<f32>)>,
    pub misses: Vec<(usize, CacheKey)>,
}

/// Splits a batch of keys into hits and misses via `lookup`, keeping each
/// entry's original index so the caller can reassemble the response in order.
pub fn reconcile(keys: &[CacheKey], lookup: impl Fn(&CacheKey) -> Option<Vec<f32>>) -> Reconciled {
    let mut hits = Vec::new();
    let mut misses = Vec::new();

    for (index, key) in keys.iter().enumerate() {
        match lookup(key) {
            Some(vector) => hits.push((index, vector)),
            None => misses.push((index, *key)),
        }
    }

    Reconciled { hits, misses }
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
            if *k == hit_key { Some(vec![1.0, 2.0]) } else { None }
        });

        assert_eq!(result.hits, vec![(1, vec![1.0, 2.0])]);
        assert_eq!(
            result.misses.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
            vec![0, 2]
        );
    }

    #[test]
    fn all_miss_when_store_is_empty() {
        let keys = [CacheKey::derive("m", "v1", "a"), CacheKey::derive("m", "v1", "b")];
        let result = reconcile(&keys, |_| None);
        assert_eq!(result.hits.len(), 0);
        assert_eq!(result.misses.len(), 2);
    }
}
