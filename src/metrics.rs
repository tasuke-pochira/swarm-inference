use lazy_static::lazy_static;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

lazy_static! {
    pub static ref MESSAGES_RECEIVED: AtomicUsize = AtomicUsize::new(0);
    pub static ref MESSAGES_SENT: AtomicUsize = AtomicUsize::new(0);
    pub static ref PROCESSING_TIME_MS: AtomicU64 = AtomicU64::new(0);
    pub static ref ERRORS_COUNT: AtomicUsize = AtomicUsize::new(0);
    pub static ref CONNECTIONS_REUSED: AtomicUsize = AtomicUsize::new(0);
    pub static ref CONNECTIONS_RECONNECTED: AtomicUsize = AtomicUsize::new(0);
    pub static ref LATENCY_MS: AtomicU64 = AtomicU64::new(0);
    pub static ref CPU_LOAD: AtomicU64 = AtomicU64::new(0);
    pub static ref TENSOR_OP_TIME_MS: AtomicU64 = AtomicU64::new(0);
    pub static ref GPU_OP_TIME_MS: AtomicU64 = AtomicU64::new(0);
    pub static ref ALERTS_TRIGGERED: AtomicU64 = AtomicU64::new(0);
}

pub fn get_metrics() -> String {
    format!(
        "messages_received: {}\nmessages_sent: {}\nprocessing_time_ms: {}\nerrors: {}\nconnections_reused: {}\nconnections_reconnected: {}\nlatency_ms: {}\ncpu_load: {}\ntensor_op_time_ms: {}\ngpu_op_time_ms: {}\nalerts_triggered: {}",
        MESSAGES_RECEIVED.load(Ordering::Relaxed),
        MESSAGES_SENT.load(Ordering::Relaxed),
        PROCESSING_TIME_MS.load(Ordering::Relaxed),
        ERRORS_COUNT.load(Ordering::Relaxed),
        CONNECTIONS_REUSED.load(Ordering::Relaxed),
        CONNECTIONS_RECONNECTED.load(Ordering::Relaxed),
        LATENCY_MS.load(Ordering::Relaxed),
        CPU_LOAD.load(Ordering::Relaxed),
        TENSOR_OP_TIME_MS.load(Ordering::Relaxed),
        GPU_OP_TIME_MS.load(Ordering::Relaxed),
        ALERTS_TRIGGERED.load(Ordering::Relaxed),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_metrics() {
        // Reset metrics for test
        MESSAGES_RECEIVED.store(5, Ordering::Relaxed);
        MESSAGES_SENT.store(3, Ordering::Relaxed);
        ERRORS_COUNT.store(1, Ordering::Relaxed);
        LATENCY_MS.store(42, Ordering::Relaxed);
        CPU_LOAD.store(75, Ordering::Relaxed);
        TENSOR_OP_TIME_MS.store(12, Ordering::Relaxed);
        GPU_OP_TIME_MS.store(34, Ordering::Relaxed);
        ALERTS_TRIGGERED.store(2, Ordering::Relaxed);

        let metrics = get_metrics();
        assert!(metrics.contains("messages_received: 5"));
        assert!(metrics.contains("latency_ms: 42"));
        assert!(metrics.contains("cpu_load: 75"));
        assert!(metrics.contains("tensor_op_time_ms: 12"));
        assert!(metrics.contains("gpu_op_time_ms: 34"));
        assert!(metrics.contains("alerts_triggered: 2"));
    }
}
