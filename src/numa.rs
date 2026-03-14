//! Basic NUMA-aware helper utilities.
//!
//! This module provides a small, cross-platform helper for optionally pinning
//! the current thread to a given CPU core. This can help reduce cross-NUMA
//! traffic on multi-socket systems.

use std::env;
use std::sync::atomic::{AtomicUsize, Ordering};

static THREAD_CORE_IDX: AtomicUsize = AtomicUsize::new(0);

#[cfg(unix)]
fn set_affinity_unix(core: usize) -> bool {
    use libc::{CPU_SET, CPU_ZERO, cpu_set_t, pthread_self, sched_setaffinity};

    unsafe {
        let mut set: cpu_set_t = std::mem::zeroed();
        CPU_ZERO(&mut set);
        CPU_SET(core, &mut set);
        let tid = pthread_self();
        sched_setaffinity(tid as libc::pid_t, std::mem::size_of::<cpu_set_t>(), &set) == 0
    }
}

#[cfg(windows)]
fn set_affinity_windows(_core: usize) -> bool {
    // Windows affinity pinning is not implemented in this crate.
    // This is intentionally left as a no-op to avoid adding heavyweight
    // native dependencies.
    false
}

#[cfg(not(any(unix, windows)))]
fn set_affinity_noop(_core: usize) -> bool {
    // Unsupported platform.
    false
}

/// Attempt to pin the current thread to a specific core index.
///
/// Returns `true` if the pinning succeeded, and `false` otherwise.
pub fn pin_current_thread(core: usize) -> bool {
    #[cfg(unix)]
    {
        set_affinity_unix(core)
    }

    #[cfg(windows)]
    {
        set_affinity_windows(core)
    }

    #[cfg(not(any(unix, windows)))]
    {
        set_affinity_noop(core)
    }
}

/// Parses a comma-separated list of cores from the environment.
///
/// Supports:
/// - `SWARM_CPU_CORE=3` (single core)
/// - `SWARM_CPU_CORES=0,1,2,3` (round-robin assignment across workers)
pub fn parse_cores_from_env() -> Option<Vec<usize>> {
    if let Ok(val) = env::var("SWARM_CPU_CORES") {
        let cores: Vec<usize> = val
            .split(',')
            .filter_map(|s| s.trim().parse::<usize>().ok())
            .collect();
        if !cores.is_empty() {
            return Some(cores);
        }
    }

    if let Some(core_id) = env::var("SWARM_CPU_CORE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        return Some(vec![core_id]);
    }

    None
}

/// If the `SWARM_CPU_CORE(S)` environment variable is set, pin the current
/// thread to one of the provided cores.
///
/// When multiple cores are configured, threads will be assigned in a round-
/// robin manner based on internal thread start order.
pub fn pin_thread_from_env() {
    if let Some(cores) = parse_cores_from_env() {
        let idx = THREAD_CORE_IDX.fetch_add(1, Ordering::Relaxed);
        let core = cores[idx % cores.len()];
        if pin_current_thread(core) {
            tracing::info!("pinned current thread to core {}", core);
        } else {
            tracing::warn!("failed to pin thread to core {}", core);
        }
    }
}
