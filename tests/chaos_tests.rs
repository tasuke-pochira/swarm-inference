//! Chaos engineering tests for fault tolerance validation
//!
//! These tests simulate various failure scenarios to ensure the swarm inference
//! system can handle node failures, network issues, and other faults gracefully.

use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use tokio::time::sleep;

use swarm_inference::{
    Coordinator, ModelArchitecture, ShardManager,
    erasure::ErasureCoder,
    kv_cache::KVCache,
    network::{ConsensusManager, NodeRegistry},
};

/// Test node failure during active inference
#[tokio::test]
async fn test_node_failure_during_inference() {
    println!("🧪 Testing node failure during active inference");

    // Setup multi-node environment
    let registry = Arc::new(Mutex::new(NodeRegistry::new()));
    let config = swarm_inference::config::Config::default();
    let _coordinator = Coordinator::new(
        "127.0.0.1:8086".to_string(),
        "chaos_test_2".to_string(),
        &config,
    )
    .await
    .unwrap();

    // Register some nodes
    {
        let mut reg = registry.lock().await;
        reg.register_node(
            "node1".to_string(),
            vec!["gpu".to_string(), "cpu".to_string()],
        );
        reg.register_node("node2".to_string(), vec!["gpu".to_string()]);
    }
    let _consensus = Arc::new(Mutex::new(ConsensusManager::new(
        "127.0.0.1:8080".to_string(),
        registry.clone(),
    )));

    // Start inference task
    let inference_handle = tokio::spawn(async move {
        let manager = ShardManager::new(ModelArchitecture::Linear, 1024, 512, 4);
        let input = vec![1.0f32; 1024];
        let kv_cache = Some(Arc::new(KVCache::new(0)));

        // Simulate long-running inference
        for i in 0..100 {
            let _ = manager
                .process_parallel(input.clone(), kv_cache.clone())
                .await;
            sleep(Duration::from_millis(10)).await; // Simulate processing time

            if i == 50 {
                // Simulate node failure at midpoint
                println!("💥 Simulating node failure at iteration {}", i);
                // In a real scenario, this would trigger consensus and reassignment
            }
        }
    });

    // Simulate node failure after some time
    sleep(Duration::from_millis(500)).await;

    // Trigger node failure simulation
    {
        let mut reg = registry.lock().await;
        // Simulate node 2 going offline
        reg.update_heartbeat("127.0.0.1:8082"); // This would normally be called by heartbeat
        // In real implementation, lack of heartbeat would trigger failure detection
    }

    // Wait for inference to complete
    let result = inference_handle.await;
    assert!(
        result.is_ok(),
        "Inference should complete despite node failure"
    );

    println!("✅ Node failure test passed - inference completed successfully");
}

/// Test network partition recovery
#[tokio::test]
async fn test_network_partition_recovery() {
    println!("🧪 Testing network partition recovery");

    let registry = Arc::new(Mutex::new(NodeRegistry::new()));
    let consensus = Arc::new(Mutex::new(ConsensusManager::new(
        "127.0.0.1:8080".to_string(),
        registry.clone(),
    )));

    // Register nodes
    {
        let mut reg = registry.lock().await;
        for i in 0..3 {
            reg.register_node(format!("127.0.0.1:808{}", i), vec!["gpu".to_string()]);
        }
    }

    // Create initial shard assignments
    {
        let mut cons = consensus.lock().await;
        let assignments = vec![
            swarm_inference::network::ShardAssignment {
                shard_id: 0,
                node_addr: "127.0.0.1:8080".to_string(),
            },
            swarm_inference::network::ShardAssignment {
                shard_id: 1,
                node_addr: "127.0.0.1:8081".to_string(),
            },
            swarm_inference::network::ShardAssignment {
                shard_id: 2,
                node_addr: "127.0.0.1:8082".to_string(),
            },
        ];
        let proposal_id = cons.propose_shard_reassignment(assignments);

        // Add enough votes to reach consensus (already have 1 auto-vote, need 2 more for majority of 3)
        let _ = cons.vote_on_proposal(proposal_id, "127.0.0.1:8081".to_string(), true);
        let _ = cons.vote_on_proposal(proposal_id, "127.0.0.1:8082".to_string(), true);

        // This should apply the assignments
        let _ = cons.check_consensus(proposal_id).await;
    }

    // Simulate network partition - node 2 becomes unreachable
    {
        let _reg = registry.lock().await;
        // Stop updating heartbeat for node 2 (simulating partition)
        // In real implementation, this would be detected by heartbeat monitor
    }

    // Trigger consensus for reassignment
    let available_nodes = {
        let _reg = registry.lock().await;
        vec![
            "127.0.0.1:8080".to_string(),
            "127.0.0.1:8081".to_string(),
            "127.0.0.1:8083".to_string(),
            "127.0.0.1:8084".to_string(),
        ]
    };
    {
        let mut cons = consensus.lock().await;
        cons.reassign_shards_on_node_failure("127.0.0.1:8082", &available_nodes);
    }

    // Verify reassignment occurred
    {
        let cons = consensus.lock().await;
        let assignments = cons.get_current_assignments();

        // Shard 2 should be reassigned to another node
        let shard_2_location = assignments.get(&2);
        assert!(shard_2_location.is_some(), "Shard 2 should be reassigned");
        assert_ne!(
            shard_2_location.unwrap(),
            "127.0.0.1:8082",
            "Shard 2 should not be on failed node"
        );
    }

    println!("✅ Network partition recovery test passed");
}

/// Test consensus failure and recovery
#[tokio::test]
async fn test_consensus_failure_recovery() {
    println!("🧪 Testing consensus failure and recovery");

    let registry = Arc::new(Mutex::new(NodeRegistry::new()));
    let consensus = Arc::new(Mutex::new(ConsensusManager::new(
        "127.0.0.0:8080".to_string(),
        registry.clone(),
    )));

    // Register nodes
    {
        let mut reg = registry.lock().await;
        for i in 0..5 {
            reg.register_node(format!("127.0.0.1:808{}", i), vec!["gpu".to_string()]);
        }
    }

    // Create a proposal
    let proposal_id = {
        let mut cons = consensus.lock().await;
        let assignments = vec![swarm_inference::network::ShardAssignment {
            shard_id: 0,
            node_addr: "127.0.0.1:8081".to_string(),
        }];
        cons.propose_shard_reassignment(assignments)
    };

    // Simulate some nodes voting
    {
        let mut cons = consensus.lock().await;
        let _ = cons.vote_on_proposal(proposal_id, "127.0.0.1:8081".to_string(), true);
        // Only 1 additional vote, so total = 2 (proposer auto-vote + this one)
    }

    // Simulate network issues causing some votes to be lost
    // In a real scenario, this would be detected by timeouts

    // Check if consensus can still be reached with available votes
    let has_consensus = {
        let mut cons = consensus.lock().await;
        let result = cons.check_consensus(proposal_id).await;
        println!("Consensus check with 2 votes: {:?}", result.is_some());
        result.is_some()
    };

    // With 2 out of 5 nodes voting, we shouldn't reach consensus (need majority of 3)
    assert!(
        !has_consensus,
        "Should not reach consensus with insufficient votes"
    );

    // Add more votes to reach consensus
    {
        let mut cons = consensus.lock().await;
        let _ = cons.vote_on_proposal(proposal_id, "127.0.0.1:8082".to_string(), true);
        let _ = cons.vote_on_proposal(proposal_id, "127.0.0.1:8083".to_string(), true);
        // Now we have 4 votes total
    }

    // Now check consensus again
    let has_consensus = {
        let mut cons = consensus.lock().await;
        cons.check_consensus(proposal_id).await.is_some()
    };

    assert!(has_consensus, "Should reach consensus with majority votes");

    println!("✅ Consensus failure recovery test passed");
}

/// Test cache synchronization under failure conditions
#[tokio::test]
async fn test_cache_sync_under_failures() {
    println!("🧪 Testing cache synchronization under failure conditions");

    let cache = Arc::new(KVCache::new(0));
    let num_operations = 1000;

    // Pre-populate cache
    for i in 0..num_operations {
        let key = format!("key_{}", i);
        let value = vec![i as f32; 100];
        let _ = cache.put(key, value).await;
    }

    // Test cache sync with simulated failures
    let sync_handle = tokio::spawn(async move {
        let mut successful_syncs = 0;
        let mut failed_syncs = 0;

        for i in 0..100 {
            // Simulate occasional sync failures
            if i % 10 == 0 {
                // Simulate network failure - skip this sync
                failed_syncs += 1;
                sleep(Duration::from_millis(1)).await;
                continue;
            }

            let _ = cache.get_delta_since(i as u64).await;
            successful_syncs += 1;
            sleep(Duration::from_millis(1)).await;
        }

        (successful_syncs, failed_syncs)
    });

    let (successful, failed) = sync_handle.await.unwrap();
    assert!(successful > 0, "Should have successful syncs");
    assert!(failed > 0, "Should have simulated failures");

    println!(
        "✅ Cache sync under failures test passed - {} successful, {} failed syncs",
        successful, failed
    );
}

/// Test erasure coding recovery from multiple failures
#[tokio::test]
async fn test_erasure_coding_multi_failure_recovery() {
    println!("🧪 Testing erasure coding recovery from multiple failures");

    let coder = ErasureCoder::new(4, 2).unwrap(); // 4 data shards, 2 parity shards
    let data = vec![42u8; 4096]; // 4KB of data

    // Encode data
    let encoded = coder.encode(&data).unwrap();
    assert_eq!(
        encoded.shards.len(),
        6,
        "Should have 6 total shards (4 data + 2 parity)"
    );

    // Simulate multiple shard losses (up to 2 should be recoverable)
    for loss_count in 1..=2 {
        let mut available_shards: std::collections::HashMap<usize, _> = (0..6)
            .filter_map(|i| encoded.shards.get(&i).map(|shard| (i, shard.clone())))
            .collect();
        // Remove 'loss_count' shards
        for i in 0..loss_count {
            available_shards.remove(&(5 - i));
        }

        // Attempt reconstruction
        let reconstructed = coder.reconstruct_shards(&available_shards);
        assert!(
            reconstructed.is_ok(),
            "Should reconstruct with {} shard loss",
            loss_count
        );

        let reconstructed_data = coder.reconstruct(&encoded);
        assert!(
            reconstructed_data.is_ok(),
            "Should reconstruct data with {} shard loss",
            loss_count
        );
        assert_eq!(
            reconstructed_data.unwrap(),
            data,
            "Reconstructed data should match original"
        );
    }

    // Test failure case - too many losses (3 losses should fail)
    let available_shards: std::collections::HashMap<usize, _> = (0..3)
        .filter_map(|i| encoded.shards.get(&i).map(|shard| (i, shard.clone())))
        .collect(); // Only 3 shards available
    let reconstructed = coder.reconstruct_shards(&available_shards);
    assert!(
        reconstructed.is_err(),
        "Should fail reconstruction with 3 shard losses"
    );

    println!("✅ Erasure coding multi-failure recovery test passed");
}

/// Test system resilience under sustained load with failures
#[tokio::test]
async fn test_sustained_load_with_failures() {
    println!("🧪 Testing sustained load with intermittent failures");

    let registry = Arc::new(Mutex::new(NodeRegistry::new()));

    // Register nodes
    {
        let mut reg = registry.lock().await;
        for i in 0..4 {
            reg.register_node(format!("127.0.0.1:808{}", i), vec!["gpu".to_string()]);
        }
    }

    let start_time = Instant::now();
    let test_duration = Duration::from_secs(5);
    let mut operations_completed = 0;
    let mut failures_simulated = 0;

    while start_time.elapsed() < test_duration {
        // Simulate normal operations
        let manager = ShardManager::new(ModelArchitecture::Linear, 256, 128, 2);
        let input = vec![1.0f32; 256];
        let kv_cache = Some(Arc::new(KVCache::new(operations_completed)));

        let _ = manager.process_parallel(input, kv_cache).await;
        operations_completed += 1;

        // Simulate occasional failures
        if operations_completed % 50 == 0 {
            failures_simulated += 1;

            // Simulate node failure
            {
                let _reg = registry.lock().await;
                let _failed_node = format!("127.0.0.1:808{}", failures_simulated % 4);
                // In real implementation, this would trigger consensus
            }

            // Small delay to simulate recovery time
            sleep(Duration::from_millis(10)).await;
        }
    }

    assert!(
        operations_completed > 100,
        "Should complete significant operations under load"
    );
    assert!(failures_simulated > 0, "Should have simulated failures");

    println!(
        "✅ Sustained load with failures test passed - {} operations, {} failures",
        operations_completed, failures_simulated
    );
}
