use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::sync::RwLock;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CacheEntry {
    pub key: Vec<f32>,
    pub value: Vec<f32>,
    pub version: u64,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheDelta {
    pub shard_id: usize,
    pub entries: HashMap<String, CacheEntry>,
    pub base_version: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CacheSyncMessage {
    FullSync {
        shard_id: usize,
        entries: HashMap<String, CacheEntry>,
    },
    DeltaSync(CacheDelta),
    Invalidate {
        shard_id: usize,
        keys: Vec<String>,
    },
    VersionRequest {
        shard_id: usize,
    },
    VersionResponse {
        shard_id: usize,
        version: u64,
    },
}

pub struct KVCache {
    entries: RwLock<HashMap<String, CacheEntry>>,
    current_version: std::sync::atomic::AtomicU64,
    shard_id: usize,
}

#[allow(dead_code)]
impl KVCache {
    pub fn new(shard_id: usize) -> Self {
        Self {
            entries: RwLock::new(HashMap::new()),
            current_version: std::sync::atomic::AtomicU64::new(0),
            shard_id,
        }
    }

    pub async fn get(&self, key: &str) -> Option<CacheEntry> {
        let entries = self.entries.read().await;
        entries.get(key).cloned()
    }

    pub async fn put(&self, key: String, value: Vec<f32>) -> Result<u64> {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let version = self
            .current_version
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            + 1;

        let entry = CacheEntry {
            key: vec![], // For simplicity, we'll use the key string hash
            value,
            version,
            timestamp,
        };

        let mut entries = self.entries.write().await;
        entries.insert(key, entry);

        Ok(version)
    }

    #[allow(dead_code)]
    pub async fn get_all_entries(&self) -> HashMap<String, CacheEntry> {
        let entries = self.entries.read().await;
        entries.clone()
    }

    pub async fn get_delta_since(&self, base_version: u64) -> HashMap<String, CacheEntry> {
        let entries = self.entries.read().await;
        entries
            .iter()
            .filter(|(_, entry)| entry.version > base_version)
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    pub async fn apply_delta(&self, delta: &CacheDelta) -> Result<()> {
        if delta.shard_id != self.shard_id {
            return Ok(()); // Not for this shard
        }

        let mut entries = self.entries.write().await;
        for (key, entry) in &delta.entries {
            // Only apply if the entry is newer
            if let Some(existing) = entries.get(key) {
                if entry.version > existing.version {
                    entries.insert(key.clone(), entry.clone());
                }
            } else {
                entries.insert(key.clone(), entry.clone());
            }
        }

        // Update our version if the delta has a higher version
        let max_version = delta
            .entries
            .values()
            .map(|e| e.version)
            .max()
            .unwrap_or(delta.base_version);

        let current = self
            .current_version
            .load(std::sync::atomic::Ordering::SeqCst);
        if max_version > current {
            self.current_version
                .store(max_version, std::sync::atomic::Ordering::SeqCst);
        }

        Ok(())
    }

    pub async fn invalidate_keys(&self, keys: &[String]) -> Result<()> {
        let mut entries = self.entries.write().await;
        for key in keys {
            entries.remove(key);
        }
        Ok(())
    }

    pub fn get_current_version(&self) -> u64 {
        self.current_version
            .load(std::sync::atomic::Ordering::SeqCst)
    }

    pub fn get_shard_id(&self) -> usize {
        self.shard_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_kv_cache_basic_operations() {
        let cache = KVCache::new(0);

        // Test put and get
        let version = cache
            .put("key1".to_string(), vec![1.0, 2.0, 3.0])
            .await
            .unwrap();
        assert_eq!(version, 1);

        let entry = cache.get("key1").await.unwrap();
        assert_eq!(entry.value, vec![1.0, 2.0, 3.0]);
        assert_eq!(entry.version, 1);

        // Test non-existent key
        assert!(cache.get("nonexistent").await.is_none());
    }

    #[tokio::test]
    async fn test_cache_versioning() {
        let cache = KVCache::new(0);

        // Put multiple entries
        cache.put("key1".to_string(), vec![1.0]).await.unwrap();
        cache.put("key2".to_string(), vec![2.0]).await.unwrap();

        assert_eq!(cache.get_current_version(), 2);

        // Get delta since version 0
        let delta = cache.get_delta_since(0).await;
        assert_eq!(delta.len(), 2);

        // Get delta since version 1
        let delta = cache.get_delta_since(1).await;
        assert_eq!(delta.len(), 1);
        assert!(delta.contains_key("key2"));
    }

    #[tokio::test]
    async fn test_delta_application() {
        let cache1 = KVCache::new(0);
        let cache2 = KVCache::new(0);

        // Cache1 has some data
        cache1.put("key1".to_string(), vec![1.0]).await.unwrap();
        cache1.put("key2".to_string(), vec![2.0]).await.unwrap();

        // Create delta from cache1
        let entries = cache1.get_delta_since(0).await;
        let delta = CacheDelta {
            shard_id: 0,
            entries,
            base_version: 0,
        };

        // Apply delta to cache2
        cache2.apply_delta(&delta).await.unwrap();

        // Cache2 should now have the same data
        let entry1 = cache2.get("key1").await.unwrap();
        let entry2 = cache2.get("key2").await.unwrap();

        assert_eq!(entry1.value, vec![1.0]);
        assert_eq!(entry2.value, vec![2.0]);
        assert_eq!(cache2.get_current_version(), 2);
    }

    #[tokio::test]
    async fn test_conflict_resolution() {
        let cache = KVCache::new(0);

        // Put initial value
        cache.put("key1".to_string(), vec![1.0]).await.unwrap();

        // Create delta with newer version
        let mut entries = HashMap::new();
        entries.insert(
            "key1".to_string(),
            CacheEntry {
                key: vec![],
                value: vec![2.0], // Updated value
                version: 5,       // Higher version
                timestamp: 100,
            },
        );

        let delta = CacheDelta {
            shard_id: 0,
            entries,
            base_version: 1,
        };

        // Apply delta
        cache.apply_delta(&delta).await.unwrap();

        // Should have the updated value
        let entry = cache.get("key1").await.unwrap();
        assert_eq!(entry.value, vec![2.0]);
        assert_eq!(entry.version, 5);
    }
}
