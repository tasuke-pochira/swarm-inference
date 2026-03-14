//! Configuration management for the Swarm Inference Protocol.
//!
//! This module provides layered configuration loading from:
//! - Default values
//! - Configuration files (TOML, YAML, JSON)
//! - Environment variables
//! - Command line overrides

use crate::auto_scaling::ScalingConfig;
use config::{Config as ConfigLoader, ConfigError, File};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Main configuration structure for the swarm inference system
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Config {
    /// Network configuration
    pub network: NetworkConfig,
    /// Performance and resource settings
    pub performance: PerformanceConfig,
    /// GPU and compute settings
    pub compute: ComputeConfig,
    /// Monitoring and observability
    pub monitoring: MonitoringConfig,
    /// Security settings
    pub security: SecurityConfig,
    /// Kubernetes-specific settings
    pub kubernetes: KubernetesConfig,
    /// Auto-scaling configuration
    pub auto_scaling: ScalingConfig,
}

/// Network-related configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Default listen address for nodes
    pub listen_addr: String,
    /// Coordinator address
    pub coordinator_addr: String,
    /// Connection timeout in seconds
    pub connection_timeout_secs: u64,
    /// Maximum message size in bytes
    pub max_message_size: usize,
    /// QUIC certificate validity days
    pub cert_validity_days: u64,
}

/// Performance and resource configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Number of worker threads
    pub worker_threads: Option<usize>,
    /// Maximum memory usage in MB
    pub max_memory_mb: Option<u64>,
    /// KV-cache synchronization interval in ms
    pub kv_cache_sync_interval_ms: u64,
    /// Erasure coding redundancy factor
    pub erasure_redundancy: usize,
    /// Consensus timeout in seconds
    pub consensus_timeout_secs: u64,
}

/// GPU and compute configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeConfig {
    /// Preferred GPU device (by index or name)
    pub preferred_gpu: Option<String>,
    /// Enable CUDA acceleration if available
    pub enable_cuda: bool,
    /// Maximum concurrent GPU operations
    pub max_concurrent_gpu_ops: usize,
    /// GPU memory pool size in MB
    pub gpu_memory_pool_mb: usize,
}

/// Monitoring and observability configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Enable metrics collection
    pub enable_metrics: bool,
    /// Metrics collection interval in seconds
    pub metrics_interval_secs: u64,
    /// Dashboard listen address
    pub dashboard_addr: String,
    /// Enable tracing
    pub enable_tracing: bool,
    /// Tracing level (error, warn, info, debug, trace)
    pub tracing_level: String,
    /// Alert thresholds
    pub alert_thresholds: AlertThresholds,
}

/// Alert threshold configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    /// Maximum latency threshold in ms
    pub max_latency_ms: u64,
    /// Error rate threshold (percentage)
    pub error_rate_threshold: f64,
    /// Memory usage threshold (percentage)
    pub memory_usage_threshold: f64,
}

/// Security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Enable TLS encryption
    pub enable_tls: bool,
    /// Certificate file path
    pub cert_file: Option<String>,
    /// Private key file path
    pub key_file: Option<String>,
    /// Authentication token (for API access)
    pub auth_token: Option<String>,
}

/// Kubernetes-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KubernetesConfig {
    /// Enable Kubernetes integration
    pub enabled: bool,
    /// Namespace for Kubernetes resources
    pub namespace: String,
    /// Service account name
    pub service_account: String,
    /// Node selector for pod scheduling
    pub node_selector: HashMap<String, String>,
    /// Tolerations for pod scheduling
    pub tolerations: Vec<Toleration>,
    /// Affinity rules for pod scheduling
    pub affinity: Option<Affinity>,
}

/// Kubernetes toleration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Toleration {
    /// Taint key
    pub key: String,
    /// Taint operator
    pub operator: String,
    /// Taint value
    pub value: Option<String>,
    /// Toleration effect
    pub effect: Option<String>,
}

/// Kubernetes affinity configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Affinity {
    /// Node affinity rules
    pub node_affinity: Option<NodeAffinity>,
    /// Pod affinity rules
    pub pod_affinity: Option<PodAffinity>,
    /// Pod anti-affinity rules
    pub pod_anti_affinity: Option<PodAntiAffinity>,
}

/// Node affinity configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeAffinity {
    /// Required node selector terms
    pub required_during_scheduling_ignored_during_execution: Vec<NodeSelectorTerm>,
    /// Preferred node selector terms
    pub preferred_during_scheduling_ignored_during_execution: Vec<PreferredSchedulingTerm>,
}

/// Pod affinity configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PodAffinity {
    /// Required pod selector terms
    pub required_during_scheduling_ignored_during_execution: Vec<PodAffinityTerm>,
    /// Preferred pod selector terms
    pub preferred_during_scheduling_ignored_during_execution: Vec<WeightedPodAffinityTerm>,
}

/// Pod anti-affinity configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PodAntiAffinity {
    /// Required pod selector terms
    pub required_during_scheduling_ignored_during_execution: Vec<PodAffinityTerm>,
    /// Preferred pod selector terms
    pub preferred_during_scheduling_ignored_during_execution: Vec<WeightedPodAffinityTerm>,
}

/// Node selector term
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeSelectorTerm {
    /// Match expressions
    pub match_expressions: Vec<NodeSelectorRequirement>,
}

/// Node selector requirement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeSelectorRequirement {
    /// Key
    pub key: String,
    /// Operator
    pub operator: String,
    /// Values
    pub values: Vec<String>,
}

/// Preferred scheduling term
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreferredSchedulingTerm {
    /// Weight
    pub weight: i32,
    /// Preference
    pub preference: NodeSelectorTerm,
}

/// Pod affinity term
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PodAffinityTerm {
    /// Label selector
    pub label_selector: Option<LabelSelector>,
    /// Namespaces
    pub namespaces: Vec<String>,
    /// Topology key
    pub topology_key: String,
}

/// Weighted pod affinity term
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightedPodAffinityTerm {
    /// Weight
    pub weight: i32,
    /// Pod affinity term
    pub pod_affinity_term: PodAffinityTerm,
}

/// Label selector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelSelector {
    /// Match labels
    pub match_labels: HashMap<String, String>,
    /// Match expressions
    pub match_expressions: Vec<LabelSelectorRequirement>,
}

/// Label selector requirement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelSelectorRequirement {
    /// Key
    pub key: String,
    /// Operator
    pub operator: String,
    /// Values
    pub values: Vec<String>,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            listen_addr: "127.0.0.1:8080".to_string(),
            coordinator_addr: "127.0.0.1:8080".to_string(),
            connection_timeout_secs: 30,
            max_message_size: 10 * 1024 * 1024, // 10MB
            cert_validity_days: 365,
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            worker_threads: None, // Use tokio default
            max_memory_mb: None,  // No limit
            kv_cache_sync_interval_ms: 100,
            erasure_redundancy: 2,
            consensus_timeout_secs: 10,
        }
    }
}

impl Default for ComputeConfig {
    fn default() -> Self {
        Self {
            preferred_gpu: None,
            enable_cuda: false,
            max_concurrent_gpu_ops: 4,
            gpu_memory_pool_mb: 1024, // 1GB
        }
    }
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enable_metrics: true,
            metrics_interval_secs: 30,
            dashboard_addr: "127.0.0.1:9090".to_string(),
            enable_tracing: true,
            tracing_level: "info".to_string(),
            alert_thresholds: AlertThresholds::default(),
        }
    }
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            max_latency_ms: 1000,
            error_rate_threshold: 5.0,
            memory_usage_threshold: 90.0,
        }
    }
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            enable_tls: true,
            cert_file: None,
            key_file: None,
            auth_token: None,
        }
    }
}

impl Default for KubernetesConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            namespace: "swarm-inference".to_string(),
            service_account: "swarm-inference-sa".to_string(),
            node_selector: HashMap::new(),
            tolerations: Vec::new(),
            affinity: None,
        }
    }
}

impl Config {
    /// Load configuration from multiple sources with priority:
    /// 1. Defaults
    /// 2. Configuration file (if provided)
    /// 3. Environment variables (SWARM_*)
    /// 4. Command line overrides
    pub fn load(config_file: Option<&Path>) -> Result<Self, ConfigError> {
        let mut loader = ConfigLoader::builder().add_source(config::File::from_str(
            &serde_yaml::to_string(&Self::default()).unwrap(),
            config::FileFormat::Yaml,
        ));

        // Add configuration file if provided
        if let Some(file_path) = config_file.filter(|p| p.exists()) {
            let format = match file_path.extension().and_then(|s| s.to_str()) {
                Some("toml") => config::FileFormat::Toml,
                Some("yaml") | Some("yml") => config::FileFormat::Yaml,
                Some("json") => config::FileFormat::Json,
                _ => {
                    return Err(ConfigError::Message(
                        "Unsupported config file format. Use .toml, .yaml, or .json".to_string(),
                    ));
                }
            };
            loader = loader.add_source(File::from(file_path).format(format));
        }

        // Add environment variables with SWARM_ prefix
        loader = loader.add_source(
            config::Environment::with_prefix("SWARM")
                .separator("_")
                .try_parsing(true),
        );

        let config = loader.build()?;
        config.try_deserialize()
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), String> {
        // Validate network addresses
        if self.network.listen_addr.is_empty() {
            return Err("Network listen_addr cannot be empty".to_string());
        }
        if self.network.coordinator_addr.is_empty() {
            return Err("Network coordinator_addr cannot be empty".to_string());
        }

        // Validate performance settings
        if self.performance.erasure_redundancy == 0 {
            return Err("Performance erasure_redundancy must be greater than 0".to_string());
        }

        // Validate compute settings
        if self.compute.max_concurrent_gpu_ops == 0 {
            return Err("Compute max_concurrent_gpu_ops must be greater than 0".to_string());
        }

        // Validate monitoring settings
        if self.monitoring.metrics_interval_secs == 0 {
            return Err("Monitoring metrics_interval_secs must be greater than 0".to_string());
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.network.listen_addr, "127.0.0.1:8080");
        assert!(config.monitoring.enable_metrics);
    }

    #[test]
    fn test_config_validation() {
        let mut config = Config::default();
        assert!(config.validate().is_ok());

        // Test invalid config
        config.network.listen_addr = "".to_string();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_load_from_file() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("config.yaml");

        let yaml_content = r#"
network:
  listen_addr: "0.0.0.0:9090"
monitoring:
  enable_metrics: false
"#;

        fs::write(&file_path, yaml_content).unwrap();

        let config = Config::load(Some(&file_path)).unwrap();
        assert_eq!(config.network.listen_addr, "0.0.0.0:9090");
        assert!(!config.monitoring.enable_metrics);
    }
}
