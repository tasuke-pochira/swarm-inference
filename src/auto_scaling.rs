//! Auto-scaling controller for the Swarm Inference Protocol.
//!
//! This module provides intelligent auto-scaling capabilities that go beyond
//! basic Kubernetes HPA by considering inference workload characteristics,
//! queue depths, latency patterns, and predictive scaling.

use crate::metrics::get_metrics;
use k8s_openapi::api::apps::v1::StatefulSet;
use kube::{
    Client,
    api::{Api, Patch, PatchParams},
};
use serde::{Deserialize, Serialize};
use serde_json::{self, json};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Metrics structure for parsing system metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metrics {
    pub messages_received: usize,
    pub messages_sent: usize,
    pub processing_time_ms: u64,
    pub errors: usize,
    pub connections_reused: usize,
    pub connections_reconnected: usize,
    pub latency_ms: u64,
    pub cpu_load: u64,
    pub tensor_op_time_ms: u64,
    pub gpu_op_time_ms: u64,
    pub alerts_triggered: u64,
}

/// Auto-scaling controller for managing cluster resources
pub struct AutoScaler {
    config: ScalingConfig,
    _k8s_config: crate::config::KubernetesConfig,
    metrics_history: Arc<RwLock<VecDeque<ScalingMetrics>>>,
    last_scale_time: Arc<RwLock<Instant>>,
    current_scale: Arc<RwLock<ScaleState>>,
    _k8s_client: Option<Client>,
}

/// Configuration for auto-scaling behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingConfig {
    /// Enable auto-scaling
    pub enabled: bool,
    /// Minimum number of nodes
    pub min_nodes: usize,
    /// Maximum number of nodes
    pub max_nodes: usize,
    /// Target average queue depth
    pub target_queue_depth: f64,
    /// Maximum allowed latency (seconds)
    pub max_latency_seconds: f64,
    /// CPU utilization threshold for scaling up
    pub cpu_scale_up_threshold: f64,
    /// CPU utilization threshold for scaling down
    pub cpu_scale_down_threshold: f64,
    /// Memory utilization threshold
    pub memory_threshold: f64,
    /// Cooldown period between scaling operations (seconds)
    pub scale_cooldown_seconds: u64,
    /// Number of metrics samples to keep for analysis
    pub metrics_history_size: usize,
    /// Scaling sensitivity (0.0-1.0, higher = more aggressive)
    pub scaling_sensitivity: f64,
    /// Enable predictive scaling based on trends
    pub predictive_scaling: bool,
    /// Lookback window for predictive scaling (seconds)
    pub predictive_window_seconds: u64,
    /// Evaluation interval in seconds
    pub interval_seconds: u64,
}

/// Metrics snapshot for scaling decisions
#[derive(Debug, Clone)]
pub struct ScalingMetrics {
    #[allow(dead_code)]
    pub timestamp: Instant,
    pub queue_depth: f64,
    pub avg_latency_ms: f64,
    pub cpu_utilization: f64,
    pub memory_utilization: f64,
    #[allow(dead_code)]
    pub active_nodes: usize,
    #[allow(dead_code)]
    pub pending_requests: usize,
}

/// Current scaling state
#[derive(Debug, Clone)]
pub struct ScaleState {
    pub current_nodes: usize,
    pub target_nodes: usize,
    pub last_scale_reason: String,
    pub last_scale_time: Instant,
}

/// Scaling decision and reasoning
#[derive(Debug, Clone)]
pub struct ScaleDecision {
    pub action: ScaleAction,
    pub target_nodes: usize,
    pub reason: String,
    pub confidence: f64, // 0.0-1.0
}

/// Type of scaling action
#[derive(Debug, Clone, PartialEq)]
pub enum ScaleAction {
    ScaleUp,
    ScaleDown,
    Hold,
}

impl Default for ScalingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            min_nodes: 3,
            max_nodes: 10,
            target_queue_depth: 5.0,
            max_latency_seconds: 2.0,
            cpu_scale_up_threshold: 70.0,
            cpu_scale_down_threshold: 30.0,
            memory_threshold: 80.0,
            scale_cooldown_seconds: 300, // 5 minutes
            metrics_history_size: 60,    // 1 minute of data
            scaling_sensitivity: 0.7,
            predictive_scaling: true,
            predictive_window_seconds: 600, // 10 minutes
            interval_seconds: 30,           // 30 seconds
        }
    }
}

impl AutoScaler {
    /// Create a new auto-scaler with the given configuration
    pub async fn new(
        config: ScalingConfig,
        k8s_config: crate::config::KubernetesConfig,
    ) -> anyhow::Result<Self> {
        let k8s_client = if config.enabled {
            match Client::try_default().await {
                Ok(client) => Some(client),
                Err(e) => {
                    tracing::warn!(
                        "Could not initialize Kubernetes client (continuing in non-k8s mode): {}",
                        e
                    );
                    None
                }
            }
        } else {
            None
        };

        Ok(Self {
            config: config.clone(),
            _k8s_config: k8s_config,
            metrics_history: Arc::new(RwLock::new(VecDeque::with_capacity(
                config.metrics_history_size,
            ))),
            last_scale_time: Arc::new(RwLock::new(Instant::now())),
            current_scale: Arc::new(RwLock::new(ScaleState {
                current_nodes: config.min_nodes,
                target_nodes: config.min_nodes,
                last_scale_reason: "Initial state".to_string(),
                last_scale_time: Instant::now(),
            })),
            _k8s_client: k8s_client,
        })
    }

    /// Reset cooldown period for testing purposes
    #[cfg(test)]
    pub async fn reset_cooldown(&self) {
        *self.last_scale_time.write().await =
            Instant::now() - Duration::from_secs(self.config.scale_cooldown_seconds + 1);
    }

    /// Record current metrics for scaling analysis
    pub async fn record_metrics(&self, metrics: ScalingMetrics) {
        let mut history = self.metrics_history.write().await;
        history.push_back(metrics);
        if history.len() > self.config.metrics_history_size {
            history.pop_front();
        }
    }

    /// Make a scaling decision based on current state and metrics
    pub async fn evaluate_scaling(&self) -> ScaleDecision {
        if !self.config.enabled {
            return ScaleDecision {
                action: ScaleAction::Hold,
                target_nodes: self.get_current_scale().await.current_nodes,
                reason: "Auto-scaling disabled".to_string(),
                confidence: 1.0,
            };
        }

        // Check cooldown period
        let last_scale = *self.last_scale_time.read().await;
        if last_scale.elapsed() < Duration::from_secs(self.config.scale_cooldown_seconds) {
            return ScaleDecision {
                action: ScaleAction::Hold,
                target_nodes: self.get_current_scale().await.current_nodes,
                reason: "In cooldown period".to_string(),
                confidence: 1.0,
            };
        }

        let history = self.metrics_history.read().await;
        if history.is_empty() {
            return ScaleDecision {
                action: ScaleAction::Hold,
                target_nodes: self.get_current_scale().await.current_nodes,
                reason: "Insufficient metrics data".to_string(),
                confidence: 0.5,
            };
        }

        let current_metrics = history.back().unwrap();
        let current_scale = self.get_current_scale().await;

        // Analyze current conditions
        let scale_up_reasons = self
            .analyze_scale_up_conditions(&history, current_metrics)
            .await;
        let scale_down_reasons = self
            .analyze_scale_down_conditions(&history, current_metrics)
            .await;

        // Make decision based on analysis
        if !scale_up_reasons.is_empty() && current_scale.current_nodes < self.config.max_nodes {
            let confidence = self.calculate_confidence(&scale_up_reasons);
            ScaleDecision {
                action: ScaleAction::ScaleUp,
                target_nodes: (current_scale.current_nodes + 1).min(self.config.max_nodes),
                reason: format!("Scale up: {}", scale_up_reasons.join(", ")),
                confidence,
            }
        } else if !scale_down_reasons.is_empty()
            && current_scale.current_nodes > self.config.min_nodes
        {
            let confidence = self.calculate_confidence(&scale_down_reasons);
            ScaleDecision {
                action: ScaleAction::ScaleDown,
                target_nodes: (current_scale.current_nodes - 1).max(self.config.min_nodes),
                reason: format!("Scale down: {}", scale_down_reasons.join(", ")),
                confidence,
            }
        } else {
            ScaleDecision {
                action: ScaleAction::Hold,
                target_nodes: current_scale.current_nodes,
                reason: "Conditions stable".to_string(),
                confidence: 0.8,
            }
        }
    }

    /// Execute a scaling decision
    pub async fn execute_scaling(&self, decision: &ScaleDecision) -> anyhow::Result<()> {
        if decision.action == ScaleAction::Hold {
            return Ok(());
        }

        // Update internal state first
        let mut current_scale = self.current_scale.write().await;
        current_scale.target_nodes = decision.target_nodes;
        current_scale.last_scale_reason = decision.reason.clone();
        current_scale.last_scale_time = Instant::now();

        *self.last_scale_time.write().await = Instant::now();

        // If running in Kubernetes, scale the StatefulSet
        if let Err(e) = self
            .scale_kubernetes_statefulset(decision.target_nodes)
            .await
        {
            tracing::warn!("Failed to scale Kubernetes StatefulSet: {}", e);
            // Continue with internal state update even if K8s scaling fails
        }

        tracing::info!(
            "Scaling decision executed: {} -> {} nodes (reason: {}, confidence: {:.2})",
            current_scale.current_nodes,
            decision.target_nodes,
            decision.reason,
            decision.confidence
        );

        current_scale.current_nodes = decision.target_nodes;

        Ok(())
    }

    /// Scale the Kubernetes StatefulSet
    async fn scale_kubernetes_statefulset(&self, target_replicas: usize) -> anyhow::Result<()> {
        // Only attempt Kubernetes scaling if we're running in a cluster
        if !self.is_running_in_kubernetes().await {
            return Ok(()); // Not an error, just not in K8s
        }

        let client = Client::try_default()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to create Kubernetes client: {}", e))?;

        let statefulsets: Api<StatefulSet> = Api::namespaced(client, "swarm-inference");

        let patch = json!({
            "spec": {
                "replicas": target_replicas
            }
        });

        let patch_params = PatchParams::apply("swarm-autoscaler").force();
        let _ = statefulsets
            .patch("swarm-nodes", &patch_params, &Patch::Merge(&patch))
            .await
            .map_err(|e| anyhow::anyhow!("Failed to patch StatefulSet: {}", e))?;

        tracing::info!(
            "Successfully scaled StatefulSet to {} replicas",
            target_replicas
        );
        Ok(())
    }

    /// Check if we're running in a Kubernetes cluster
    async fn is_running_in_kubernetes(&self) -> bool {
        std::env::var("KUBERNETES_SERVICE_HOST").is_ok()
    }

    /// Get current scaling state
    pub async fn get_current_scale(&self) -> ScaleState {
        self.current_scale.read().await.clone()
    }

    /// Get scaling metrics from the system
    pub async fn collect_metrics(&self) -> anyhow::Result<ScalingMetrics> {
        // This would integrate with the actual metrics system
        // For now, return mock data
        let metrics_str = get_metrics();
        let _metrics: Metrics = serde_json::from_str(&metrics_str)
            .map_err(|e| anyhow::anyhow!("Failed to parse metrics: {}", e))?;

        Ok(ScalingMetrics {
            timestamp: Instant::now(),
            queue_depth: 2.5,         // Mock queue depth
            avg_latency_ms: 150.0,    // Mock latency
            cpu_utilization: 65.0,    // Mock CPU usage
            memory_utilization: 70.0, // Mock memory usage
            active_nodes: self.get_current_scale().await.current_nodes,
            pending_requests: 3, // Mock pending requests
        })
    }

    /// Analyze conditions that suggest scaling up
    async fn analyze_scale_up_conditions(
        &self,
        history: &VecDeque<ScalingMetrics>,
        current: &ScalingMetrics,
    ) -> Vec<String> {
        let mut reasons = Vec::new();

        // High queue depth
        if current.queue_depth > self.config.target_queue_depth * 1.5 {
            reasons.push(format!(
                "Queue depth {:.1} > target {:.1}",
                current.queue_depth, self.config.target_queue_depth
            ));
        }

        // High latency
        if current.avg_latency_ms > self.config.max_latency_seconds * 1000.0 {
            reasons.push(format!(
                "Latency {:.0}ms > max {:.0}ms",
                current.avg_latency_ms,
                self.config.max_latency_seconds * 1000.0
            ));
        }

        // High CPU utilization
        if current.cpu_utilization > self.config.cpu_scale_up_threshold {
            reasons.push(format!(
                "CPU utilization {:.1}% > threshold {:.1}%",
                current.cpu_utilization, self.config.cpu_scale_up_threshold
            ));
        }

        // High memory utilization
        if current.memory_utilization > self.config.memory_threshold {
            reasons.push(format!(
                "Memory utilization {:.1}% > threshold {:.1}%",
                current.memory_utilization, self.config.memory_threshold
            ));
        }

        // Predictive scaling based on trends
        if self.config.predictive_scaling && self.detect_upward_trend(history) {
            reasons.push("Predictive scaling: upward load trend detected".to_string());
        }

        reasons
    }

    /// Analyze conditions that suggest scaling down
    async fn analyze_scale_down_conditions(
        &self,
        history: &VecDeque<ScalingMetrics>,
        current: &ScalingMetrics,
    ) -> Vec<String> {
        let mut reasons = Vec::new();

        // Low resource utilization
        if current.cpu_utilization < self.config.cpu_scale_down_threshold
            && current.memory_utilization < self.config.memory_threshold * 0.7
        {
            reasons.push(format!(
                "Low utilization: CPU {:.1}%, Memory {:.1}%",
                current.cpu_utilization, current.memory_utilization
            ));
        }

        // Low queue depth consistently
        let avg_queue_depth =
            history.iter().map(|m| m.queue_depth).sum::<f64>() / history.len() as f64;
        if avg_queue_depth < self.config.target_queue_depth * 0.3 {
            reasons.push(format!(
                "Consistently low queue depth: avg {:.1}",
                avg_queue_depth
            ));
        }

        reasons
    }

    /// Detect upward trend in load metrics
    fn detect_upward_trend(&self, history: &VecDeque<ScalingMetrics>) -> bool {
        if history.len() < 10 {
            return false;
        }

        let recent = history.iter().rev().take(5).collect::<Vec<_>>();
        let older = history.iter().rev().skip(5).take(5).collect::<Vec<_>>();

        let recent_avg_queue =
            recent.iter().map(|m| m.queue_depth).sum::<f64>() / recent.len() as f64;
        let older_avg_queue = older.iter().map(|m| m.queue_depth).sum::<f64>() / older.len() as f64;

        let recent_avg_cpu =
            recent.iter().map(|m| m.cpu_utilization).sum::<f64>() / recent.len() as f64;
        let older_avg_cpu =
            older.iter().map(|m| m.cpu_utilization).sum::<f64>() / older.len() as f64;

        // Trend detection: recent averages significantly higher than older
        (recent_avg_queue > older_avg_queue * 1.3) || (recent_avg_cpu > older_avg_cpu * 1.2)
    }

    /// Calculate confidence score for scaling decision
    fn calculate_confidence(&self, reasons: &[String]) -> f64 {
        let base_confidence = reasons.len() as f64 * 0.2; // 0.2 per reason
        (base_confidence * self.config.scaling_sensitivity).min(1.0)
    }
}

/// Auto-scaling service that runs periodic scaling evaluations
#[derive(Clone)]
pub struct ScalingService {
    scaler: Arc<AutoScaler>,
    evaluation_interval: Duration,
}

impl ScalingService {
    /// Create a new scaling service
    pub fn new(scaler: Arc<AutoScaler>, evaluation_interval: Duration) -> Self {
        Self {
            scaler,
            evaluation_interval,
        }
    }

    /// Start the scaling service
    pub async fn run(self) -> anyhow::Result<()> {
        tracing::info!(
            "Starting auto-scaling service with {}s evaluation interval",
            self.evaluation_interval.as_secs()
        );

        let mut interval = tokio::time::interval(self.evaluation_interval);

        loop {
            interval.tick().await;

            // Collect current metrics
            match self.scaler.collect_metrics().await {
                Ok(metrics) => {
                    // Record metrics for analysis
                    self.scaler.record_metrics(metrics).await;

                    // Evaluate scaling decision
                    let decision = self.scaler.evaluate_scaling().await;

                    // Log the decision
                    match decision.action {
                        ScaleAction::ScaleUp => {
                            tracing::info!(
                                "Scale up decision: {} -> {} nodes (confidence: {:.2})",
                                self.scaler.get_current_scale().await.current_nodes,
                                decision.target_nodes,
                                decision.confidence
                            );
                        }
                        ScaleAction::ScaleDown => {
                            tracing::info!(
                                "Scale down decision: {} -> {} nodes (confidence: {:.2})",
                                self.scaler.get_current_scale().await.current_nodes,
                                decision.target_nodes,
                                decision.confidence
                            );
                        }
                        ScaleAction::Hold => {
                            tracing::debug!("Hold decision: {}", decision.reason);
                        }
                    }

                    // Execute the scaling decision
                    if let Err(e) = self.scaler.execute_scaling(&decision).await {
                        tracing::error!("Failed to execute scaling decision: {}", e);
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to collect scaling metrics: {}", e);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::sleep;

    #[tokio::test]
    async fn test_autoscaler_creation() {
        let config = ScalingConfig::default();
        let k8s_config = crate::config::KubernetesConfig::default();
        let scaler = AutoScaler::new(config.clone(), k8s_config).await.unwrap();

        let scale_state = scaler.get_current_scale().await;
        assert_eq!(scale_state.current_nodes, config.min_nodes);
        assert_eq!(scale_state.target_nodes, config.min_nodes);
    }

    #[tokio::test]
    async fn test_scaling_decision_disabled() {
        let config = ScalingConfig {
            enabled: false,
            ..Default::default()
        };
        let k8s_config = crate::config::KubernetesConfig::default();
        let scaler = AutoScaler::new(config, k8s_config).await.unwrap();

        let decision = scaler.evaluate_scaling().await;
        assert_eq!(decision.action, ScaleAction::Hold);
        assert_eq!(decision.reason, "Auto-scaling disabled");
    }

    #[tokio::test]
    async fn test_scale_up_on_high_load() {
        let config = ScalingConfig {
            enabled: true,
            min_nodes: 3,
            max_nodes: 10,
            target_queue_depth: 5.0,
            ..Default::default()
        };
        let k8s_config = crate::config::KubernetesConfig::default();
        let scaler = AutoScaler::new(config, k8s_config).await.unwrap();

        // Record high load metrics
        let metrics = ScalingMetrics {
            timestamp: Instant::now(),
            queue_depth: 15.0, // Much higher than target
            avg_latency_ms: 500.0,
            cpu_utilization: 85.0,
            memory_utilization: 75.0,
            active_nodes: 3,
            pending_requests: 10,
        };

        scaler.record_metrics(metrics).await;

        // Reset cooldown to allow immediate scaling
        scaler.reset_cooldown().await;

        let decision = scaler.evaluate_scaling().await;
        println!(
            "Decision: {:?}, Reason: {}",
            decision.action, decision.reason
        );
        assert_eq!(decision.action, ScaleAction::ScaleUp);
        assert_eq!(decision.target_nodes, 4);
        assert!(decision.reason.contains("Queue depth"));
    }

    #[tokio::test]
    async fn test_cooldown_period() {
        let config = ScalingConfig {
            enabled: true,
            scale_cooldown_seconds: 1, // Short cooldown for testing
            ..Default::default()
        };
        let k8s_config = crate::config::KubernetesConfig::default();
        let scaler = AutoScaler::new(config, k8s_config).await.unwrap();

        // First scaling should work
        let decision1 = scaler.evaluate_scaling().await;
        assert_eq!(decision1.action, ScaleAction::Hold); // No metrics yet

        // Execute a scaling action
        scaler
            .execute_scaling(&ScaleDecision {
                action: ScaleAction::ScaleUp,
                target_nodes: 4,
                reason: "Test".to_string(),
                confidence: 1.0,
            })
            .await
            .unwrap();

        // Second scaling should be blocked by cooldown
        let decision2 = scaler.evaluate_scaling().await;
        assert_eq!(decision2.action, ScaleAction::Hold);
        assert_eq!(decision2.reason, "In cooldown period");

        // Wait for cooldown to expire
        sleep(Duration::from_secs(2)).await;

        // Third scaling should work again
        let decision3 = scaler.evaluate_scaling().await;
        assert_ne!(decision3.reason, "In cooldown period");
    }
}
