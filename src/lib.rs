// Swarm Inference Protocol
// Developed by Tasuke Pochira, independent developer
// Licensed under Apache 2.0

//! Swarm Inference Protocol - A heterogeneous distributed inference system.
//!
//! This crate provides a production-ready swarm inference protocol with:
//! - Distributed model sharding across heterogeneous hardware
//! - Fault tolerance through erasure coding and consensus
//! - GPU acceleration (wgpu + optional CUDA)
//! - Real-time performance monitoring and metrics
//! - Asynchronous KV-cache synchronization
//! - Comprehensive audit logging for security events

pub mod alerting;
pub mod audit;
pub mod auto_scaling;
pub mod benchmark;
pub mod checkpoint;
pub mod config;
pub mod coordinator;
pub mod dashboard;
pub mod erasure;
pub mod gpu;
pub mod kv_cache;
pub mod memory_pool;
pub mod metrics;
pub mod model;
pub mod network;
pub mod node;

// Re-export key types for easier testing
pub use audit::{AuditLogger, AuditResult, get_audit_logger, init_audit_logger};
pub use coordinator::Coordinator;
pub use gpu::{GpuCompute, create_gpu_backend};
pub use memory_pool::{GLOBAL_TENSOR_POOL, TensorPool};
pub use metrics::get_metrics;
pub use model::{ModelArchitecture, ShardManager};
pub use network::{ConsensusManager, NodeRegistry};
pub use node::Node;
