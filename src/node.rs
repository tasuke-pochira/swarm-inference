use crate::checkpoint::{CheckpointManager, InferenceCheckpoint};
use crate::kv_cache::KVCache;
use crate::metrics::{
    CONNECTIONS_RECONNECTED, CONNECTIONS_REUSED, ERRORS_COUNT, MESSAGES_RECEIVED, MESSAGES_SENT,
    PROCESSING_TIME_MS,
};
use crate::model::{GpuShard, ModelShard, RealShard};
use crate::network::{
    BiStream, CompressedData, Message, create_client_endpoint, create_server_endpoint,
    dequantize_u8_to_f32, open_bi_stream, quic_accept, quic_connect, receive_message, send_message,
};
use anyhow::Result;
use quinn::Connection;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::{Mutex, Notify};
use tokio::time::{Duration, interval};
use tracing::{error, info};

static THROTTLE_INTERVAL_MS: AtomicU64 = AtomicU64::new(10); // Adaptive throttling: starts at 10ms, increases on errors
use std::collections::VecDeque;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Priority {
    High,
    Normal,
}

struct PrioritizedQueue {
    state: Mutex<PrioritizedQueueState>,
    notify: Notify,
}

struct PrioritizedQueueState {
    high: VecDeque<Message>,
    normal: VecDeque<Message>,
}

impl PrioritizedQueue {
    fn new() -> Self {
        Self {
            state: Mutex::new(PrioritizedQueueState {
                high: VecDeque::new(),
                normal: VecDeque::new(),
            }),
            notify: Notify::new(),
        }
    }

    async fn push(&self, priority: Priority, msg: Message) {
        let mut state = self.state.lock().await;
        match priority {
            Priority::High => state.high.push_back(msg),
            Priority::Normal => state.normal.push_back(msg),
        }
        drop(state);
        self.notify.notify_one();
    }

    async fn recv(&self) -> Message {
        loop {
            let mut state = self.state.lock().await;
            if let Some(msg) = state.high.pop_front() {
                return msg;
            }
            if let Some(msg) = state.normal.pop_front() {
                return msg;
            }
            drop(state);
            self.notify.notified().await;
        }
    }
}

pub struct Node {
    id: usize,
    shard: Arc<dyn ModelShard + Send + Sync + 'static>,
    kv_cache: Arc<KVCache>,
    checkpoint_manager: Arc<Mutex<CheckpointManager>>,
    next_addr: Option<String>,
    next_conn: Arc<Mutex<Option<Connection>>>,
    next_queue: Option<Arc<PrioritizedQueue>>,
    coordinator_addr: String,
    coordinator_conn: Arc<Mutex<Option<Connection>>>,
    coordinator_queue: Arc<PrioritizedQueue>,
}

impl Node {
    pub fn new(id: usize, next_addr: Option<String>, coordinator_addr: String) -> Self {
        let use_gpu = std::env::var("SWARM_USE_GPU").is_ok();
        let shard: Arc<dyn ModelShard + Send + Sync + 'static> = if use_gpu {
            Arc::new(GpuShard::new(10, 10))
        } else {
            Arc::new(RealShard::new(
                crate::model::ModelArchitecture::Linear,
                10,
                10,
            ))
        };

        let kv_cache = Arc::new(KVCache::new(id));
        let checkpoint_manager = Arc::new(Mutex::new(CheckpointManager::new(100))); // Keep up to 100 checkpoints
        Self {
            id,
            shard,
            kv_cache,
            checkpoint_manager,
            next_addr: next_addr.clone(),
            next_conn: Arc::new(Mutex::new(None)),
            next_queue: next_addr
                .as_ref()
                .map(|_| Arc::new(PrioritizedQueue::new())),
            coordinator_addr,
            coordinator_conn: Arc::new(Mutex::new(None)),
            coordinator_queue: Arc::new(PrioritizedQueue::new()),
        }
    }

    pub async fn run(&self, addr: &str) -> Result<()> {
        let endpoint = create_server_endpoint(addr).await?;
        info!("Node {} listening on {}", self.id, addr);

        // Spawn worker tasks for prioritized outbound messaging.
        {
            let coordinator_queue = Arc::clone(&self.coordinator_queue);
            let coordinator_conn = Arc::clone(&self.coordinator_conn);
            let coordinator_addr = self.coordinator_addr.clone();
            tokio::spawn(async move {
                loop {
                    let msg = coordinator_queue.recv().await;
                    if let Err(e) = send_via_pool(&coordinator_conn, &coordinator_addr, &msg).await
                    {
                        error!("Coordinator send failed: {}", e);
                        ERRORS_COUNT.fetch_add(1, Ordering::Relaxed);
                    } else {
                        MESSAGES_SENT.fetch_add(1, Ordering::Relaxed);
                    }
                }
            });
        }

        if let (Some(next_addr), Some(next_queue)) =
            (self.next_addr.clone(), self.next_queue.clone())
        {
            let next_conn = Arc::clone(&self.next_conn);
            let heartbeat_queue = Arc::clone(&next_queue);
            tokio::spawn(async move {
                loop {
                    let msg = next_queue.recv().await;
                    if let Err(e) = send_via_pool(&next_conn, &next_addr, &msg).await {
                        error!("Next node send failed: {}", e);
                        ERRORS_COUNT.fetch_add(1, Ordering::Relaxed);
                    } else {
                        MESSAGES_SENT.fetch_add(1, Ordering::Relaxed);
                    }
                }
            });

            tokio::spawn(async move {
                let mut interval = interval(Duration::from_secs(5));
                loop {
                    interval.tick().await;
                    heartbeat_queue
                        .push(Priority::High, Message::Heartbeat)
                        .await;
                }
            });
        }

        loop {
            let conn = quic_accept(&endpoint).await?;
            let mut stream = open_bi_stream(&conn).await?;
            let shard = Arc::clone(&self.shard);
            let kv_cache = Arc::clone(&self.kv_cache);
            let checkpoint_manager = Arc::clone(&self.checkpoint_manager);
            let next_addr = self.next_addr.clone();
            let next_queue = self.next_queue.clone();
            let coordinator_queue = Arc::clone(&self.coordinator_queue);
            tokio::spawn(async move {
                if let Err(e) = handle_connection(
                    &mut stream,
                    shard,
                    kv_cache,
                    checkpoint_manager,
                    next_addr,
                    next_queue,
                    coordinator_queue,
                )
                .await
                {
                    error!("Error handling connection: {}", e);
                    ERRORS_COUNT.fetch_add(1, Ordering::Relaxed);
                }
            });
        }
    }
}

async fn send_via_pool(
    pool: &Arc<Mutex<Option<Connection>>>,
    addr: &str,
    msg: &Message,
) -> Result<()> {
    let span = tracing::trace_span!("send_via_pool", addr = %addr, msg = ?msg);
    let _enter = span.enter();

    let mut lock = pool.lock().await;
    let already_connected = lock.is_some();

    if lock.is_none() {
        let endpoint = create_client_endpoint().await?;
        let conn = quic_connect(&endpoint, addr).await?;
        *lock = Some(conn);
    }

    if let Some(conn) = lock.as_ref() {
        let mut stream = open_bi_stream(conn).await?;
        tokio::time::sleep(Duration::from_millis(
            THROTTLE_INTERVAL_MS.load(Ordering::Relaxed),
        ))
        .await;
        match send_message(&mut stream, msg).await {
            Ok(_) => {
                if already_connected {
                    CONNECTIONS_REUSED.fetch_add(1, Ordering::Relaxed);
                }
            }
            Err(e) => {
                error!("send_via_pool: send failed to {}: {}", addr, e);
                *lock = None;
                CONNECTIONS_RECONNECTED.fetch_add(1, Ordering::Relaxed);
                // Adaptive throttling: increase interval on errors
                let current = THROTTLE_INTERVAL_MS.load(Ordering::Relaxed);
                if current < 100 {
                    THROTTLE_INTERVAL_MS.store(current + 5, Ordering::Relaxed);
                }

                let endpoint = create_client_endpoint().await?;
                let conn = quic_connect(&endpoint, addr).await?;
                *lock = Some(conn);
                let mut stream = open_bi_stream(lock.as_ref().unwrap()).await?;
                tokio::time::sleep(Duration::from_millis(
                    THROTTLE_INTERVAL_MS.load(Ordering::Relaxed),
                ))
                .await;
                send_message(&mut stream, msg).await?;
            }
        }
    }

    Ok(())
}

async fn handle_connection(
    stream: &mut BiStream,
    shard: Arc<dyn ModelShard + Send + Sync + 'static>,
    kv_cache: Arc<KVCache>,
    checkpoint_manager: Arc<Mutex<CheckpointManager>>,
    next_addr: Option<String>,
    next_queue: Option<Arc<PrioritizedQueue>>,
    coordinator_queue: Arc<PrioritizedQueue>,
) -> Result<()> {
    let span = tracing::info_span!("handle_connection");
    let _enter = span.enter();

    let msg = receive_message(stream).await?;
    tracing::info!(?msg, "Received message");
    MESSAGES_RECEIVED.fetch_add(1, Ordering::Relaxed);
    let start = std::time::Instant::now();

    match msg {
        Message::Prompt { text } => {
            let input: Vec<f32> = text.chars().map(|c| c as u32 as f32).collect();
            let output = shard.process(input, Some(kv_cache.clone())).await?;
            let compressed = CompressedData::compress(&output)?;
            let outgoing = if next_addr.is_some() {
                Message::Intermediate { data: compressed }
            } else {
                Message::Result {
                    text: format!("Processed: {:?}", output),
                }
            };

            if let Some(queue) = next_queue {
                queue.push(Priority::Normal, outgoing).await;
            } else {
                coordinator_queue.push(Priority::Normal, outgoing).await;
            }
        }
        Message::Intermediate { data } => {
            let input = data.decompress()?;
            let output = shard.process(input, Some(kv_cache.clone())).await?;
            let compressed = CompressedData::compress(&output)?;
            let outgoing = if next_addr.is_some() {
                Message::Intermediate { data: compressed }
            } else {
                Message::Result {
                    text: format!("Processed: {:?}", output),
                }
            };

            if let Some(queue) = next_queue {
                queue.push(Priority::Normal, outgoing).await;
            } else {
                coordinator_queue.push(Priority::Normal, outgoing).await;
            }
        }
        Message::InferenceRequest {
            prompt_id,
            quantized_input,
        } => {
            let input_data = dequantize_u8_to_f32(&quantized_input);
            let output = shard.process(input_data, Some(kv_cache.clone())).await?;
            let latency = start.elapsed().as_millis() as u64;
            crate::metrics::LATENCY_MS.store(latency, Ordering::Relaxed);
            let result_msg = Message::InferenceResult { prompt_id, output };
            coordinator_queue.push(Priority::Normal, result_msg).await;
        }
        Message::Heartbeat => {
            send_message(stream, &Message::HeartbeatAck).await?;
        }
        Message::HeartbeatAck => {
            info!("Received heartbeat ack");
        }
        Message::CacheSync(sync_msg) => {
            use crate::kv_cache::CacheSyncMessage::*;
            match sync_msg {
                FullSync { shard_id, entries } => {
                    if shard_id == kv_cache.get_shard_id() {
                        // Clear existing cache and apply full sync
                        // Note: In a real implementation, we'd need to clear the cache
                        // For now, we'll just apply the entries
                        for (key, entry) in entries {
                            // This is a simplified version - in practice we'd need to handle versioning
                            let _ = kv_cache.put(key, entry.value).await;
                        }
                    }
                }
                DeltaSync(delta) => {
                    if let Err(e) = kv_cache.apply_delta(&delta).await {
                        error!("Failed to apply cache delta: {}", e);
                    }
                }
                #[allow(clippy::collapsible_if)]
                Invalidate { shard_id, keys } => {
                    if shard_id == kv_cache.get_shard_id() {
                        if let Err(e) = kv_cache.invalidate_keys(&keys).await {
                            error!("Failed to invalidate cache keys: {}", e);
                        }
                    }
                }
                VersionRequest { shard_id } => {
                    if shard_id == kv_cache.get_shard_id() {
                        let version = kv_cache.get_current_version();
                        let response = Message::CacheSync(VersionResponse { shard_id, version });
                        send_message(stream, &response).await?;
                    }
                }
                VersionResponse { .. } => {
                    // Handle version response if needed
                }
            }
        }
        Message::ErasureSync {
            shard_id,
            erasure_data,
        } => {
            // Handle erasure-coded data synchronization
            // In a full implementation, this would update the node's shard manager
            // with the erasure-coded data for redundancy
            info!(
                "Received erasure sync for shard {} with {} total shards",
                shard_id, erasure_data.total_shards
            );
        }
        Message::CheckpointSave {
            inference_id,
            checkpoint_data,
        } => {
            let checkpoint = InferenceCheckpoint::from_bytes(&checkpoint_data)?;
            let mut manager = checkpoint_manager.lock().await;
            manager.save_checkpoint(checkpoint)?;
            info!("Saved checkpoint for inference {}", inference_id);
        }
        Message::CheckpointLoad { inference_id } => {
            let manager = checkpoint_manager.lock().await;
            let checkpoint_data = manager
                .load_checkpoint(&inference_id)
                .and_then(|cp| cp.to_bytes().ok());
            let response = Message::CheckpointResponse {
                inference_id,
                checkpoint_data,
            };
            send_message(stream, &response).await?;
        }
        Message::CheckpointResponse {
            inference_id,
            checkpoint_data,
        } => {
            // Handle checkpoint response - this would be used by coordinators/nodes
            // to resume from checkpoints
            if let Some(data) = checkpoint_data {
                let checkpoint = InferenceCheckpoint::from_bytes(&data)?;
                info!(
                    "Received checkpoint for inference {} at step {}",
                    inference_id, checkpoint.current_step
                );
            } else {
                info!("No checkpoint found for inference {}", inference_id);
            }
        }
        _ => {}
    }

    PROCESSING_TIME_MS.fetch_add(start.elapsed().as_millis() as u64, Ordering::Relaxed);
    Ok(())
}
