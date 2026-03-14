//! Simple memory pooling for tensor allocations.
//!
//! This module provides a lightweight pool for reusing `Vec<f32>` buffers to
//! reduce allocation churn during repeated inference workloads. It also
//! includes a basic garbage collection mechanism to prevent unbounded growth.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Debug)]
struct PooledVec {
    vec: Vec<f32>,
    last_used: Instant,
}

#[derive(Debug)]
struct PoolState {
    map: HashMap<usize, VecDeque<PooledVec>>,
    puts_since_cleanup: usize,
}

/// A global pool that holds reusable vectors keyed by their capacity.
#[derive(Clone)]
pub struct TensorPool {
    inner: Arc<Mutex<PoolState>>,
    /// Maximum number of vectors to keep per capacity.
    max_per_capacity: usize,
    /// Duration after which unused vectors are eligible for cleanup.
    max_age: Duration,
    /// How often to run cleanup (number of `put_vec` calls).
    cleanup_every: usize,
}

impl TensorPool {
    pub fn new(max_per_capacity: usize, max_age: Duration) -> Self {
        Self::with_cleanup_interval(max_per_capacity, max_age, 64)
    }

    pub fn with_cleanup_interval(
        max_per_capacity: usize,
        max_age: Duration,
        cleanup_every: usize,
    ) -> Self {
        Self {
            inner: Arc::new(Mutex::new(PoolState {
                map: HashMap::new(),
                puts_since_cleanup: 0,
            })),
            max_per_capacity,
            max_age,
            cleanup_every,
        }
    }

    /// Get a vector with at least `capacity` elements.
    ///
    /// The returned vector will have length 0 but capacity >= requested.
    pub fn get_vec(&self, capacity: usize) -> Vec<f32> {
        let mut guard = self.inner.lock().unwrap();
        if let Some(queue) = guard.map.get_mut(&capacity) {
            while let Some(mut entry) = queue.pop_front() {
                if entry.last_used.elapsed() <= self.max_age {
                    entry.vec.clear();
                    return entry.vec;
                }
            }
        }
        Vec::with_capacity(capacity)
    }

    /// Return a vector to the pool for future reuse.
    pub fn put_vec(&self, mut vec: Vec<f32>) {
        let cap = vec.capacity();
        vec.clear();

        let mut guard = self.inner.lock().unwrap();
        let queue = guard.map.entry(cap).or_default();
        if queue.len() < self.max_per_capacity {
            queue.push_back(PooledVec {
                vec,
                last_used: Instant::now(),
            });
        }
        // If the pool is already full, drop the vector.

        // Periodically cleanup old entries to prevent unbounded growth.
        guard.puts_since_cleanup += 1;
        if guard.puts_since_cleanup >= self.cleanup_every {
            guard.puts_since_cleanup = 0;
            let now = Instant::now();
            for queue in guard.map.values_mut() {
                while let Some(front) = queue.front() {
                    if now.duration_since(front.last_used) > self.max_age {
                        queue.pop_front();
                    } else {
                        break;
                    }
                }
            }
        }
    }

    /// Removes old entries from the pool.
    pub fn cleanup(&self) {
        let mut guard = self.inner.lock().unwrap();
        let now = Instant::now();

        for queue in guard.map.values_mut() {
            while let Some(front) = queue.front() {
                if now.duration_since(front.last_used) > self.max_age {
                    queue.pop_front();
                } else {
                    break;
                }
            }
        }
    }
}

impl Default for TensorPool {
    fn default() -> Self {
        Self::new(16, Duration::from_secs(60))
    }
}

lazy_static::lazy_static! {
    /// Global tensor pool used by inference code.
    pub static ref GLOBAL_TENSOR_POOL: TensorPool = TensorPool::default();
}
