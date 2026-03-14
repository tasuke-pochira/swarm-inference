use lazy_static::lazy_static;
use std::sync::{Arc, RwLock};

pub type AlertCallback = Arc<dyn Fn(&str) + Send + Sync + 'static>;

lazy_static! {
    static ref ALERT_HANDLER: RwLock<Option<AlertCallback>> = RwLock::new(None);
}

/// Trigger an alert for a critical issue.
///
/// By default this logs an error via tracing and increments the `ALERTS_TRIGGERED` metric.
/// Users can override the handler via `set_alert_handler`.
pub fn alert(message: &str) {
    // Record a metric for observability
    crate::metrics::ALERTS_TRIGGERED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    // Call custom handler if registered
    if let Some(handler) = ALERT_HANDLER.read().unwrap().as_ref() {
        handler(message);
        return;
    }

    tracing::error!(%message, "critical alert");
}

/// Replace the current alert handler.
///
/// This can be used to forward critical alerts to external systems (webhooks, pager, etc.).
pub fn set_alert_handler(handler: AlertCallback) {
    let mut slot = ALERT_HANDLER.write().unwrap();
    *slot = Some(handler);
}
