use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

#[tokio::test]
async fn test_alerting_handler_invoked() {
    // Reset counter
    swarm_inference::metrics::ALERTS_TRIGGERED.store(0, Ordering::Relaxed);

    let called = Arc::new(Mutex::new(false));
    let called_clone = called.clone();

    swarm_inference::alerting::set_alert_handler(Arc::new(move |msg: &str| {
        let mut lock = called_clone.lock().unwrap();
        *lock = true;
        assert!(
            msg.contains("critical"),
            "Alert message should contain context"
        );
    }));

    swarm_inference::alerting::alert("critical issue: node down");

    assert!(*called.lock().unwrap());
    assert_eq!(
        swarm_inference::metrics::ALERTS_TRIGGERED.load(Ordering::Relaxed),
        1
    );
}
