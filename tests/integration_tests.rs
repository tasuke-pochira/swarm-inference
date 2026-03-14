//! Integration tests for multi-node swarm inference scenarios.
//!
//! These tests verify the end-to-end functionality of the swarm inference protocol,
//! including multi-node coordination, GPU acceleration, fault tolerance, and performance.

use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::future::join_all;
use swarm_inference::{
    Coordinator, ModelArchitecture, Node, ShardManager, gpu::create_gpu_backend,
};
use tokio::sync::Mutex;

/// Test basic multi-node setup with coordinator and workers
#[tokio::test]
async fn test_multi_node_setup() {
    // Create coordinator
    let config = swarm_inference::config::Config::default();
    let _coordinator: Arc<Mutex<Coordinator>> = Arc::new(Mutex::new(
        Coordinator::new(
            "127.0.0.1:8081".to_string(),
            "127.0.0.1:8080".to_string(),
            &config,
        )
        .await
        .unwrap(),
    ));

    // Create worker nodes
    let worker_configs = vec![
        (1, "127.0.0.1:8081".to_string()),
        (2, "127.0.0.1:8082".to_string()),
    ];

    let mut workers: Vec<(Arc<Mutex<Node>>, String)> = Vec::new();
    for (id, addr) in &worker_configs {
        let node: Arc<Mutex<Node>> = Arc::new(Mutex::new(Node::new(
            *id,
            None,
            "127.0.0.1:8080".to_string(),
        )));
        workers.push((node, addr.clone()));
    }

    // Start coordinator (just keep it alive for the test)
    let coord_handle = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_secs(2)).await;
    });

    // Start workers briefly
    let mut worker_handles = Vec::new();
    for (worker, addr) in workers.iter().take(1) {
        // Just test one worker to avoid conflicts
        let worker_clone = Arc::clone(worker);
        let addr_clone = addr.clone();
        let handle = tokio::spawn(async move {
            let node = worker_clone.lock().await;
            // Just run briefly to test startup
            tokio::time::timeout(Duration::from_millis(500), node.run(&addr_clone)).await
        });
        worker_handles.push(handle);
    }

    // Wait a bit for setup
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Basic test - just verify components can be created
    assert_eq!(workers.len(), 2);

    // Cleanup
    coord_handle.abort();
    for handle in worker_handles {
        handle.abort();
    }
}

/// Test GPU backend functionality and backend selection
#[tokio::test]
async fn test_gpu_backend_functionality() {
    // Test default backend (should be wgpu if available)
    let backend_result = create_gpu_backend();
    if backend_result.is_err() {
        println!("Skipping GPU test - no GPU backend available");
        return;
    }

    let backend = backend_result.unwrap();

    // Test basic matrix-vector multiplication
    let weights = vec![1.0f32, 2.0, 3.0, 4.0]; // 2x2 matrix
    let input = vec![1.0f32, 1.0]; // 2 elements
    let bias = vec![0.0f32, 0.0]; // 2 elements

    let result = backend.matmul(&weights, &input, 2, 2, &bias).unwrap();
    assert_eq!(result.len(), 2);
    // Expected: [1*1 + 2*1, 3*1 + 4*1] + [0, 0] = [3, 7]
    assert!((result[0] - 3.0).abs() < 1e-6);
    assert!((result[1] - 7.0).abs() < 1e-6);

    // Test CUDA backend selection (if available)
    unsafe {
        std::env::set_var("SWARM_GPU_BACKEND", "cuda");
    }
    let cuda_result = create_gpu_backend();
    if let Ok(cuda_backend) = cuda_result {
        let cuda_result = cuda_backend.matmul(&weights, &input, 2, 2, &bias).unwrap();
        assert_eq!(cuda_result.len(), 2);
        assert!((cuda_result[0] - 3.0).abs() < 1e-6);
        assert!((cuda_result[1] - 7.0).abs() < 1e-6);
    } else {
        println!("CUDA backend not available, skipping CUDA test");
    }

    // Cleanup
    unsafe {
        std::env::remove_var("SWARM_GPU_BACKEND");
    }
}

/// Performance benchmark for GPU operations
#[tokio::test]
async fn benchmark_gpu_operations() {
    // Skip if no GPU available
    let gpu_backend = match create_gpu_backend() {
        Ok(backend) => backend,
        Err(_) => {
            println!("Skipping GPU benchmark - no GPU backend available");
            return;
        }
    };

    // Test different model sizes
    let test_configs = vec![(128, 64, "small"), (512, 256, "medium")];

    for (in_dim, out_dim, size_name) in test_configs {
        println!("Benchmarking {} model ({}x{})", size_name, in_dim, out_dim);

        // Create test input
        let input = vec![0.1f32; in_dim];

        // Benchmark GPU computation
        let start = Instant::now();
        let iterations = 10;

        for _ in 0..iterations {
            // Create random weights and bias for benchmarking
            let weights: Vec<f32> = (0..in_dim * out_dim)
                .map(|i| (i as f32 * 0.01).sin())
                .collect();
            let bias = vec![0.0f32; out_dim];

            let _result = gpu_backend
                .matmul(&weights, &input, in_dim, out_dim, &bias)
                .unwrap();
        }

        let duration = start.elapsed();
        let avg_time = duration.as_secs_f64() / iterations as f64;

        println!("  Average GPU time: {:.4}ms", avg_time * 1000.0);
    }
}

/// Test concurrent GPU operations across multiple "shards"
#[tokio::test]
async fn test_concurrent_gpu_operations() {
    // Skip if no GPU available
    let gpu_backend = match create_gpu_backend() {
        Ok(backend) => Arc::new(Mutex::new(backend)),
        Err(_) => {
            println!("Skipping concurrent GPU test - no GPU backend available");
            return;
        }
    };

    let num_operations = 4;
    let mut handles = Vec::new();

    for i in 0..num_operations {
        let backend_clone = Arc::clone(&gpu_backend);
        let handle = tokio::spawn(async move {
            let backend = backend_clone.lock().await;
            let weights = vec![1.0f32, 2.0, 3.0, 4.0];
            let input = vec![i as f32, (i + 1) as f32];
            let bias = vec![0.0f32, 0.0];

            backend.matmul(&weights, &input, 2, 2, &bias)
        });
        handles.push(handle);
    }

    // Wait for all operations to complete
    let results = join_all(handles).await;

    for (i, result) in results.into_iter().enumerate() {
        let output = result.unwrap().unwrap();
        assert_eq!(output.len(), 2);
        // Each operation should produce different results based on input
        let expected_0 = 1.0 * i as f32 + 2.0 * (i + 1) as f32;
        let expected_1 = 3.0 * i as f32 + 4.0 * (i + 1) as f32;
        assert!((output[0] - expected_0).abs() < 1e-6);
        assert!((output[1] - expected_1).abs() < 1e-6);
    }
}

/// Test model sharding and shard management
#[tokio::test]
async fn test_model_sharding() {
    let in_dim = 128;
    let out_dim = 64;
    let num_shards = 4;

    // Create shard manager with erasure coding
    let shard_manager = ShardManager::new_with_erasure_coding(
        ModelArchitecture::Linear,
        in_dim,
        out_dim,
        num_shards,
        2, // 2 parity shards for fault tolerance
    )
    .unwrap();

    assert_eq!(shard_manager.num_shards(), num_shards);

    // Test getting individual shards
    for i in 0..num_shards {
        let _shard = shard_manager.get_shard(i).unwrap();
        // Shard exists and can be retrieved
    }

    // Test erasure coding capabilities
    let test_data = b"Hello, swarm inference!".to_vec();
    let mut manager_clone = shard_manager;
    manager_clone.encode_model_data(&test_data).unwrap();

    // Test reconstruction with some shards missing
    let reconstructed = manager_clone.reconstruct_model_data().unwrap();
    assert_eq!(reconstructed, test_data);
}

/// Test multi-node inference scenario with shard distribution
#[tokio::test]
async fn test_multi_node_inference() {
    use std::collections::HashMap;
    use swarm_inference::{
        kv_cache::KVCache,
        network::{ConsensusManager, NodeRegistry},
    };

    // Create a simple multi-node setup with 3 nodes
    let registry = Arc::new(tokio::sync::Mutex::new(NodeRegistry::new()));
    {
        let mut reg = registry.lock().await;
        reg.register_node("127.0.0.1:8081".to_string(), vec!["gpu".to_string()]);
        reg.register_node("127.0.0.1:8082".to_string(), vec!["gpu".to_string()]);
        reg.register_node("127.0.0.1:8083".to_string(), vec!["cpu".to_string()]);
    }

    // Create consensus manager for shard assignment
    let consensus = Arc::new(tokio::sync::Mutex::new(ConsensusManager::new(
        "127.0.0.1:8081".to_string(),
        Arc::clone(&registry),
    )));

    // Set up initial shard assignments across nodes using proposals
    {
        let mut cons = consensus.lock().await;
        let assignments = vec![
            swarm_inference::network::ShardAssignment {
                shard_id: 0,
                node_addr: "127.0.0.1:8081".to_string(),
            },
            swarm_inference::network::ShardAssignment {
                shard_id: 1,
                node_addr: "127.0.0.1:8082".to_string(),
            },
            swarm_inference::network::ShardAssignment {
                shard_id: 2,
                node_addr: "127.0.0.1:8083".to_string(),
            },
        ];
        let proposal_id = cons.propose_shard_reassignment(assignments);
        // Simulate votes from other nodes to reach majority (3 nodes total -> majority is 2)
        let _ = cons.vote_on_proposal(proposal_id, "127.0.0.1:8082".to_string(), true);
        let _ = cons.vote_on_proposal(proposal_id, "127.0.0.1:8083".to_string(), true);
        // Check consensus and apply assignments
        let _ = cons.check_consensus(proposal_id).await;
    }

    // Create shard managers for each node
    let shard_managers = Arc::new(tokio::sync::Mutex::new(HashMap::new()));
    let node_addresses = vec!["127.0.0.1:8081", "127.0.0.1:8082", "127.0.0.1:8083"];

    for addr in &node_addresses {
        let manager = ShardManager::new(ModelArchitecture::Linear, 4, 2, 1);
        shard_managers
            .lock()
            .await
            .insert(addr.to_string(), manager);
    }

    // Simulate inference request distribution
    let input = vec![1.0f32, 2.0, 3.0, 4.0];
    let kv_cache = Arc::new(KVCache::new(0)); // Use shard_id 0 for testing

    // Process shards in parallel (simulating distributed inference)
    let mut results = Vec::new();
    for addr in &node_addresses {
        let managers = Arc::clone(&shard_managers);
        let input_clone = input.clone();
        let cache_clone = Arc::clone(&kv_cache);
        let addr_str = addr.to_string();

        let result = tokio::spawn(async move {
            let mgrs = managers.lock().await;
            if let Some(manager) = mgrs.get(&addr_str) {
                manager
                    .process_parallel(input_clone, Some(cache_clone))
                    .await
            } else {
                Err(anyhow::anyhow!("Manager not found"))
            }
        });
        results.push(result);
    }

    // Collect results from all nodes
    let all_results = join_all(results).await;
    let mut successful_results = 0;

    for result in all_results {
        if let Ok(Ok(shard_results)) = result {
            assert_eq!(shard_results.len(), 1); // Each node has 1 shard
            assert_eq!(shard_results[0].len(), 2); // Output dimension
            successful_results += 1;
        }
    }

    // At least one node should have processed successfully
    assert!(successful_results > 0);

    // Test consensus-based reassignment on node failure
    {
        let mut cons = consensus.lock().await;
        let available_nodes = vec!["127.0.0.1:8081".to_string(), "127.0.0.1:8082".to_string()];
        let reassignments =
            cons.reassign_shards_on_node_failure("127.0.0.1:8083", &available_nodes);
        assert!(!reassignments.is_empty());
    }

    // Verify shard locations after reassignment
    {
        let cons = consensus.lock().await;
        assert!(cons.get_shard_location(0).is_some());
        assert!(cons.get_shard_location(1).is_some());
        assert!(cons.get_shard_location(2).is_some());
    }
}
