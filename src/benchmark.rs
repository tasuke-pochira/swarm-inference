//! Performance benchmarking suite for the swarm inference protocol.
//!
//! This module provides comprehensive benchmarks for measuring the performance
//! of various components including model inference, networking, KV-cache operations,
//! and multi-node coordination.

use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::{
    ModelArchitecture, ShardManager,
    erasure::ErasureCoder,
    kv_cache::KVCache,
    network::{ConsensusManager, NodeRegistry},
};

/// Benchmark results structure
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub name: String,
    pub duration: Duration,
    pub operations_per_second: f64,
    pub throughput_mbps: Option<f64>,
    pub latency_ms: f64,
}

/// Run all performance benchmarks
pub async fn run_all_benchmarks() -> Vec<BenchmarkResult> {
    let mut results = Vec::new();

    println!("🚀 Starting Swarm Inference Performance Benchmarks");
    println!("==================================================");

    // Model inference benchmarks
    results.extend(run_model_benchmarks().await);
    results.extend(run_kv_cache_benchmarks().await);
    results.extend(run_erasure_coding_benchmarks().await);
    results.extend(run_consensus_benchmarks().await);

    println!("\n📊 Benchmark Summary");
    println!("===================");
    for result in &results {
        println!(
            "{:<30} {:>8.2} ops/sec, {:>6.2}ms latency",
            result.name, result.operations_per_second, result.latency_ms
        );
        if let Some(throughput) = result.throughput_mbps {
            println!("{:<30} {:>8.2} MB/s throughput", "", throughput);
        }
    }

    results
}

/// Benchmark model inference performance
async fn run_model_benchmarks() -> Vec<BenchmarkResult> {
    println!("\n🧠 Running Model Inference Benchmarks");

    let mut results = Vec::new();
    let input_sizes = vec![64, 128, 256, 512, 1024];

    for &input_size in &input_sizes {
        // Linear model benchmark
        let linear_result = benchmark_linear_model(input_size).await;
        results.push(linear_result);

        // Transformer model benchmark (if implemented)
        // let transformer_result = benchmark_transformer_model(input_size).await;
        // results.push(transformer_result);
    }

    results
}

/// Benchmark linear model inference
async fn benchmark_linear_model(input_size: usize) -> BenchmarkResult {
    let manager = ShardManager::new(ModelArchitecture::Linear, input_size, input_size / 2, 2);
    let input = vec![1.0f32; input_size];
    let kv_cache = Some(Arc::new(KVCache::new(0)));

    let num_iterations = 1000;
    let start = Instant::now();

    for _ in 0..num_iterations {
        let _ = manager
            .process_parallel(input.clone(), kv_cache.clone())
            .await;
    }

    let duration = start.elapsed();
    let operations_per_second = num_iterations as f64 / duration.as_secs_f64();
    let latency_ms = duration.as_nanos() as f64 / num_iterations as f64 / 1_000_000.0;

    BenchmarkResult {
        name: format!("Linear Model ({}→{})", input_size, input_size / 2),
        duration,
        operations_per_second,
        throughput_mbps: None,
        latency_ms,
    }
}

/// Benchmark KV-cache operations
async fn run_kv_cache_benchmarks() -> Vec<BenchmarkResult> {
    println!("\n💾 Running KV-Cache Benchmarks");

    let mut results = Vec::new();
    let cache = Arc::new(KVCache::new(0));
    let num_operations = 10000;

    // Benchmark cache writes
    let write_result = benchmark_cache_writes(cache.clone(), num_operations).await;
    results.push(write_result);

    // Benchmark cache reads
    let read_result = benchmark_cache_reads(cache.clone(), num_operations).await;
    results.push(read_result);

    // Benchmark cache sync
    let sync_result = benchmark_cache_sync(cache.clone(), num_operations).await;
    results.push(sync_result);

    results
}

/// Benchmark KV-cache write operations
async fn benchmark_cache_writes(cache: Arc<KVCache>, num_operations: usize) -> BenchmarkResult {
    let start = Instant::now();

    for i in 0..num_operations {
        let key = format!("key_{}", i);
        let value = vec![i as f32; 100];
        let _ = cache.put(key, value).await;
    }

    let duration = start.elapsed();
    let operations_per_second = num_operations as f64 / duration.as_secs_f64();
    let latency_ms = duration.as_nanos() as f64 / num_operations as f64 / 1_000_000.0;

    BenchmarkResult {
        name: "KV-Cache Writes".to_string(),
        duration,
        operations_per_second,
        throughput_mbps: None,
        latency_ms,
    }
}

/// Benchmark KV-cache read operations
async fn benchmark_cache_reads(cache: Arc<KVCache>, num_operations: usize) -> BenchmarkResult {
    // Pre-populate cache
    for i in 0..num_operations {
        let key = format!("key_{}", i);
        let value = vec![i as f32; 100];
        let _ = cache.put(key, value).await;
    }

    let start = Instant::now();

    for i in 0..num_operations {
        let key = format!("key_{}", i);
        let _ = cache.get(&key).await;
    }

    let duration = start.elapsed();
    let operations_per_second = num_operations as f64 / duration.as_secs_f64();
    let latency_ms = duration.as_nanos() as f64 / num_operations as f64 / 1_000_000.0;

    BenchmarkResult {
        name: "KV-Cache Reads".to_string(),
        duration,
        operations_per_second,
        throughput_mbps: None,
        latency_ms,
    }
}

/// Benchmark KV-cache synchronization
async fn benchmark_cache_sync(cache: Arc<KVCache>, num_operations: usize) -> BenchmarkResult {
    let start = Instant::now();

    for i in 0..num_operations {
        let _ = cache.get_delta_since(i as u64).await;
    }

    let duration = start.elapsed();
    let operations_per_second = num_operations as f64 / duration.as_secs_f64();
    let latency_ms = duration.as_nanos() as f64 / num_operations as f64 / 1_000_000.0;

    BenchmarkResult {
        name: "KV-Cache Sync".to_string(),
        duration,
        operations_per_second,
        throughput_mbps: None,
        latency_ms,
    }
}

/// Benchmark erasure coding operations
async fn run_erasure_coding_benchmarks() -> Vec<BenchmarkResult> {
    println!("\n🔧 Running Erasure Coding Benchmarks");

    let mut results = Vec::new();
    let data_sizes = vec![1024, 4096, 16384, 65536]; // 1KB, 4KB, 16KB, 64KB

    for &data_size in &data_sizes {
        let coder = ErasureCoder::new(4, 2).unwrap();
        let data = vec![42u8; data_size];

        // Benchmark encoding
        let encode_result = benchmark_erasure_encoding(&coder, &data).await;
        results.push(encode_result);

        // Benchmark decoding
        let decode_result = benchmark_erasure_decoding(&coder, &data).await;
        results.push(decode_result);
    }

    results
}

/// Benchmark erasure encoding
async fn benchmark_erasure_encoding(coder: &ErasureCoder, data: &[u8]) -> BenchmarkResult {
    let num_iterations = 100;
    let start = Instant::now();

    for _ in 0..num_iterations {
        let _ = coder.encode(data);
    }

    let duration = start.elapsed();
    let operations_per_second = num_iterations as f64 / duration.as_secs_f64();
    let latency_ms = duration.as_nanos() as f64 / num_iterations as f64 / 1_000_000.0;
    let throughput_mbps =
        (data.len() * num_iterations) as f64 / duration.as_secs_f64() / (1024.0 * 1024.0);

    BenchmarkResult {
        name: format!("Erasure Encode ({}KB)", data.len() / 1024),
        duration,
        operations_per_second,
        throughput_mbps: Some(throughput_mbps),
        latency_ms,
    }
}

/// Benchmark erasure decoding
async fn benchmark_erasure_decoding(coder: &ErasureCoder, data: &[u8]) -> BenchmarkResult {
    let encoded = coder.encode(data).unwrap();
    let num_iterations = 100;
    let start = Instant::now();

    for _ in 0..num_iterations {
        let _ = coder.reconstruct(&encoded);
    }

    let duration = start.elapsed();
    let operations_per_second = num_iterations as f64 / duration.as_secs_f64();
    let latency_ms = duration.as_nanos() as f64 / num_iterations as f64 / 1_000_000.0;
    let throughput_mbps =
        (data.len() * num_iterations) as f64 / duration.as_secs_f64() / (1024.0 * 1024.0);

    BenchmarkResult {
        name: format!("Erasure Decode ({}KB)", data.len() / 1024),
        duration,
        operations_per_second,
        throughput_mbps: Some(throughput_mbps),
        latency_ms,
    }
}

/// Benchmark consensus operations
async fn run_consensus_benchmarks() -> Vec<BenchmarkResult> {
    println!("\n⚖️  Running Consensus Benchmarks");

    let mut results = Vec::new();
    let registry = Arc::new(tokio::sync::Mutex::new(NodeRegistry::new()));

    // Register nodes
    {
        let mut reg = registry.lock().await;
        for i in 0..5 {
            reg.register_node(format!("127.0.0.1:808{}", i), vec!["gpu".to_string()]);
        }
    }

    let consensus = Arc::new(tokio::sync::Mutex::new(ConsensusManager::new(
        "127.0.0.1:8080".to_string(),
        registry.clone(),
    )));

    // Benchmark proposal creation
    let proposal_result = benchmark_consensus_proposals(consensus.clone()).await;
    results.push(proposal_result);

    // Benchmark voting
    let voting_result = benchmark_consensus_voting(consensus.clone()).await;
    results.push(voting_result);

    results
}

/// Benchmark consensus proposal creation
async fn benchmark_consensus_proposals(
    consensus: Arc<tokio::sync::Mutex<ConsensusManager>>,
) -> BenchmarkResult {
    let num_proposals = 1000;
    let start = Instant::now();

    for i in 0..num_proposals {
        let mut cons = consensus.lock().await;
        let assignments = vec![crate::network::ShardAssignment {
            shard_id: i,
            node_addr: format!("127.0.0.1:808{}", i % 5),
        }];
        let _ = cons.propose_shard_reassignment(assignments);
    }

    let duration = start.elapsed();
    let operations_per_second = num_proposals as f64 / duration.as_secs_f64();
    let latency_ms = duration.as_nanos() as f64 / num_proposals as f64 / 1_000_000.0;

    BenchmarkResult {
        name: "Consensus Proposals".to_string(),
        duration,
        operations_per_second,
        throughput_mbps: None,
        latency_ms,
    }
}

/// Benchmark consensus voting
async fn benchmark_consensus_voting(
    consensus: Arc<tokio::sync::Mutex<ConsensusManager>>,
) -> BenchmarkResult {
    let num_proposals = 100;
    let mut proposal_ids = Vec::new();

    // Create proposals
    {
        let mut cons = consensus.lock().await;
        for i in 0..num_proposals {
            let assignments = vec![crate::network::ShardAssignment {
                shard_id: i,
                node_addr: format!("127.0.0.1:808{}", i % 5),
            }];
            let proposal_id = cons.propose_shard_reassignment(assignments);
            proposal_ids.push(proposal_id);
        }
    }

    let start = Instant::now();

    for &proposal_id in &proposal_ids {
        let mut cons = consensus.lock().await;
        // Simulate votes from multiple nodes
        for voter_id in 1..4 {
            let _ = cons.vote_on_proposal(proposal_id, format!("127.0.0.1:808{}", voter_id), true);
        }
        let _ = cons.check_consensus(proposal_id).await;
    }

    let duration = start.elapsed();
    let operations_per_second = num_proposals as f64 / duration.as_secs_f64();
    let latency_ms = duration.as_nanos() as f64 / num_proposals as f64 / 1_000_000.0;

    BenchmarkResult {
        name: "Consensus Voting".to_string(),
        duration,
        operations_per_second,
        throughput_mbps: None,
        latency_ms,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_benchmark_execution() {
        let results = run_all_benchmarks().await;
        assert!(!results.is_empty());

        // Verify all benchmarks completed
        for result in results {
            assert!(result.duration > Duration::ZERO);
            assert!(result.operations_per_second > 0.0);
            assert!(result.latency_ms > 0.0);
        }
    }
}
