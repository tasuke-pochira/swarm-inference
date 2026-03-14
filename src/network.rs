use crate::{AuditResult, get_audit_logger};
use anyhow::Result;
use quinn::{Connection, Endpoint, RecvStream, SendStream};
use rcgen::{Certificate, CertificateParams};
use rustls::client::ServerCertVerified;
use rustls::client::ServerCertVerifier;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

pub struct BiStream {
    send: SendStream,
    recv: RecvStream,
}

impl AsyncRead for BiStream {
    fn poll_read(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &mut tokio::io::ReadBuf<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        std::pin::Pin::new(&mut self.get_mut().recv).poll_read(cx, buf)
    }
}

impl AsyncWrite for BiStream {
    fn poll_write(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &[u8],
    ) -> std::task::Poll<Result<usize, std::io::Error>> {
        std::pin::Pin::new(&mut self.get_mut().send).poll_write(cx, buf)
    }

    fn poll_flush(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), std::io::Error>> {
        std::pin::Pin::new(&mut self.get_mut().send).poll_flush(cx)
    }

    fn poll_shutdown(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), std::io::Error>> {
        std::pin::Pin::new(&mut self.get_mut().send).poll_shutdown(cx)
    }
}

pub fn generate_self_signed_cert() -> Result<(rustls::Certificate, rustls::PrivateKey)> {
    let mut params = CertificateParams::new(vec!["localhost".to_string()]);
    params.is_ca = rcgen::IsCa::Ca(rcgen::BasicConstraints::Unconstrained);
    let cert = Certificate::from_params(params)?;
    let cert_der = cert.serialize_der()?;
    let key_der = cert.serialize_private_key_der();
    let cert = rustls::Certificate(cert_der);
    let key = rustls::PrivateKey(key_der);
    Ok((cert, key))
}

struct SkipServerVerification;

impl SkipServerVerification {
    fn new() -> Arc<Self> {
        Arc::new(Self)
    }
}

impl ServerCertVerifier for SkipServerVerification {
    fn verify_server_cert(
        &self,
        _end_entity: &rustls::Certificate,
        _intermediates: &[rustls::Certificate],
        _server_name: &rustls::ServerName,
        _scts: &mut dyn Iterator<Item = &[u8]>,
        _ocsp_response: &[u8],
        _now: std::time::SystemTime,
    ) -> Result<ServerCertVerified, rustls::Error> {
        Ok(ServerCertVerified::assertion())
    }
}

pub fn server_config(
    cert: rustls::Certificate,
    key: rustls::PrivateKey,
) -> Result<quinn::ServerConfig> {
    let server_config = quinn::ServerConfig::with_single_cert(vec![cert], key)?;
    Ok(server_config)
}

pub fn client_config() -> Result<quinn::ClientConfig> {
    let crypto = rustls::ClientConfig::builder()
        .with_safe_defaults()
        .with_custom_certificate_verifier(SkipServerVerification::new())
        .with_no_client_auth();
    Ok(quinn::ClientConfig::new(Arc::new(crypto)))
}

pub async fn create_server_endpoint(addr: &str) -> Result<Endpoint> {
    let (cert, key) = generate_self_signed_cert()?;
    let server_config = server_config(cert, key)?;
    let endpoint = Endpoint::server(server_config, addr.parse()?)?;
    Ok(endpoint)
}

pub async fn create_client_endpoint() -> Result<Endpoint> {
    let client_config = client_config()?;
    let mut endpoint = Endpoint::client("0.0.0.0:0".parse()?)?;
    endpoint.set_default_client_config(client_config);
    Ok(endpoint)
}

pub async fn quic_connect(endpoint: &Endpoint, addr: &str) -> Result<Connection> {
    let connection = endpoint.connect(addr.parse()?, "localhost")?.await?;

    // Audit the connection attempt
    get_audit_logger().log_access_control(
        "network",
        None,
        addr,
        "connect",
        AuditResult::Success,
        serde_json::json!({
            "protocol": "QUIC"
        }),
    );

    Ok(connection)
}

pub async fn quic_accept(endpoint: &Endpoint) -> Result<Connection> {
    let incoming = endpoint
        .accept()
        .await
        .ok_or(anyhow::anyhow!("No incoming connection"))?;
    let connection = incoming.await?;
    Ok(connection)
}

pub async fn open_bi_stream(conn: &Connection) -> Result<BiStream> {
    let (send, recv) = conn.open_bi().await?;
    Ok(BiStream { send, recv })
}

#[derive(Serialize, Deserialize, Debug)]
pub enum Message {
    Prompt {
        text: String,
    },
    Intermediate {
        data: CompressedData,
    },
    Result {
        text: String,
    },
    Heartbeat,
    HeartbeatAck,
    InferenceRequest {
        prompt_id: u64,
        quantized_input: Vec<u8>,
    },
    InferenceResult {
        prompt_id: u64,
        output: Vec<f32>,
    },
    CacheSync(crate::kv_cache::CacheSyncMessage),
    ErasureSync {
        shard_id: usize,
        erasure_data: crate::erasure::ErasureCodedData,
    },
    CheckpointSave {
        inference_id: String,
        checkpoint_data: Vec<u8>,
    },
    CheckpointLoad {
        inference_id: String,
    },
    CheckpointResponse {
        inference_id: String,
        checkpoint_data: Option<Vec<u8>>,
    },
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CompressedData {
    pub compressed: Vec<u8>,
    pub original_len: usize,
}

impl CompressedData {
    pub fn compress(data: &[f32]) -> Result<Self> {
        let bytes = bincode::serialize(data)?;
        let compressed = zstd::encode_all(bytes.as_slice(), 3)?;
        Ok(Self {
            compressed,
            original_len: data.len(),
        })
    }

    pub fn decompress(&self) -> Result<Vec<f32>> {
        let bytes = zstd::decode_all(self.compressed.as_slice())?;
        let data: Vec<f32> = bincode::deserialize(&bytes)?;
        Ok(data)
    }
}

pub fn quantize_f32_to_u8(data: &[f32]) -> Vec<u8> {
    data.iter()
        .map(|&x| {
            let clamped = x.clamp(-10.0, 10.0);
            ((clamped + 10.0) / 20.0 * 255.0) as u8
        })
        .collect()
}

pub fn dequantize_u8_to_f32(data: &[u8]) -> Vec<f32> {
    data.iter()
        .map(|&x| (x as f32) / 255.0 * 20.0 - 10.0)
        .collect()
}

pub async fn send_message(stream: &mut BiStream, msg: &Message) -> Result<()> {
    let span = tracing::trace_span!("send_message", msg = ?msg);
    let _enter = span.enter();
    let encoded = bincode::serialize(msg)?;
    let len = encoded.len() as u32;
    stream.write_all(&len.to_be_bytes()).await?;
    stream.write_all(&encoded).await?;
    tracing::trace!(len = len, "message sent");
    Ok(())
}

pub async fn receive_message(stream: &mut BiStream) -> Result<Message> {
    let span = tracing::trace_span!("receive_message");
    let _enter = span.enter();
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf).await?;
    let len = u32::from_be_bytes(len_buf) as usize;
    let mut buf = vec![0u8; len];
    stream.read_exact(&mut buf).await?;
    let msg: Message = bincode::deserialize(&buf)?;
    tracing::trace!(?msg, "message received");
    Ok(msg)
}

pub struct LatencyPredictor {
    history: Vec<f32>,
    alpha: f32, // smoothing factor for EMA
}

#[allow(dead_code)]
impl LatencyPredictor {
    pub fn new(alpha: f32) -> Self {
        Self {
            history: Vec::new(),
            alpha,
        }
    }

    #[allow(dead_code)]
    pub fn add_measurement(&mut self, latency_ms: f32) {
        self.history.push(latency_ms);
        if self.history.len() > 100 {
            self.history.remove(0);
        }
    }

    #[allow(dead_code)]
    pub fn predict(&self) -> f32 {
        if self.history.is_empty() {
            return 0.0;
        }
        let mut ema = self.history[0];
        for &lat in &self.history[1..] {
            ema = self.alpha * lat + (1.0 - self.alpha) * ema;
        }
        ema
    }
}

pub struct RoutingTable {
    shard_to_nodes: HashMap<usize, Vec<String>>,
    node_locations: HashMap<String, NodeLocation>,
}

#[derive(Debug, Clone)]
pub struct NodeLocation {
    pub latitude: f32,
    pub longitude: f32,
}

#[allow(dead_code)]
impl RoutingTable {
    pub fn new() -> Self {
        Self {
            shard_to_nodes: HashMap::new(),
            node_locations: HashMap::new(),
        }
    }

    pub fn add_route(&mut self, shard_id: usize, node_addr: String) {
        self.shard_to_nodes
            .entry(shard_id)
            .or_default()
            .push(node_addr);
    }

    pub fn add_node_location(&mut self, addr: String, location: NodeLocation) {
        self.node_locations.insert(addr, location);
    }

    pub fn get_nodes(&self, shard_id: usize) -> Option<&Vec<String>> {
        self.shard_to_nodes.get(&shard_id)
    }

    pub fn get_best_node(&self, shard_id: usize, _predictor: &LatencyPredictor) -> Option<String> {
        // Basic adaptive routing: return first node, but could use predictor
        self.get_nodes(shard_id)
            .and_then(|nodes| nodes.first().cloned())
    }

    pub fn get_closest_node(
        &self,
        shard_id: usize,
        client_location: &NodeLocation,
    ) -> Option<String> {
        self.get_nodes(shard_id).and_then(|nodes| {
            nodes
                .iter()
                .min_by(|a, b| {
                    let dist_a = self.node_locations.get(*a).map_or(f32::INFINITY, |loc| {
                        haversine_distance(client_location, loc)
                    });
                    let dist_b = self.node_locations.get(*b).map_or(f32::INFINITY, |loc| {
                        haversine_distance(client_location, loc)
                    });
                    dist_a.partial_cmp(&dist_b).unwrap()
                })
                .cloned()
        })
    }
}

impl Default for RoutingTable {
    fn default() -> Self {
        Self::new()
    }
}

fn haversine_distance(loc1: &NodeLocation, loc2: &NodeLocation) -> f32 {
    let dlat = (loc2.latitude - loc1.latitude).to_radians();
    let dlon = (loc2.longitude - loc1.longitude).to_radians();
    let a = (dlat / 2.0).sin().powi(2)
        + loc1.latitude.to_radians().cos()
            * loc2.latitude.to_radians().cos()
            * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());
    6371.0 * c // Earth radius in km
}

#[derive(Debug, Clone)]
pub struct NodeInfo {
    #[allow(dead_code)]
    pub addr: String,
    pub capabilities: Vec<String>,
    pub last_seen: std::time::Instant,
}

#[allow(dead_code)]
pub struct NodeRegistry {
    nodes: HashMap<String, NodeInfo>,
}

#[allow(dead_code)]
impl NodeRegistry {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
        }
    }

    pub fn register_node(&mut self, addr: String, capabilities: Vec<String>) {
        let info = NodeInfo {
            addr: addr.clone(),
            capabilities,
            last_seen: std::time::Instant::now(),
        };
        self.nodes.insert(addr, info);
    }

    pub fn update_heartbeat(&mut self, addr: &str) {
        if let Some(info) = self.nodes.get_mut(addr) {
            info.last_seen = std::time::Instant::now();
        }
    }

    pub fn get_active_nodes(&self) -> Vec<String> {
        let now = std::time::Instant::now();
        self.nodes
            .iter()
            .filter(|(_, info)| now.duration_since(info.last_seen).as_secs() < 60)
            .map(|(addr, _)| addr.clone())
            .collect()
    }

    pub fn get_nodes_with_capability(&self, capability: &str) -> Vec<String> {
        self.get_active_nodes()
            .into_iter()
            .filter(|addr| {
                self.nodes
                    .get(addr)
                    .unwrap()
                    .capabilities
                    .contains(&capability.to_string())
            })
            .collect()
    }
}

impl Default for NodeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardAssignment {
    #[allow(dead_code)]
    pub shard_id: usize,
    #[allow(dead_code)]
    pub node_addr: String,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusProposal {
    pub id: u64,
    #[allow(dead_code)]
    pub assignments: Vec<ShardAssignment>,
    #[allow(dead_code)]
    pub proposer: String,
    #[allow(dead_code)]
    pub timestamp: u64,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsensusMessage {
    Propose(ConsensusProposal),
    Vote {
        proposal_id: u64,
        voter: String,
        approve: bool,
    },
    Commit {
        proposal_id: u64,
    },
}

#[allow(dead_code)]
pub struct ConsensusManager {
    node_addr: String,
    current_assignments: HashMap<usize, String>, // shard_id -> node_addr
    proposals: HashMap<u64, ConsensusProposal>,
    votes: HashMap<u64, HashMap<String, bool>>, // proposal_id -> (voter -> vote)
    next_proposal_id: u64,
    registry: Arc<tokio::sync::Mutex<NodeRegistry>>,
}

#[allow(dead_code)]
impl ConsensusManager {
    pub fn new(node_addr: String, registry: Arc<tokio::sync::Mutex<NodeRegistry>>) -> Self {
        Self {
            node_addr,
            current_assignments: HashMap::new(),
            proposals: HashMap::new(),
            votes: HashMap::new(),
            next_proposal_id: 1,
            registry,
        }
    }

    pub fn get_current_assignments(&self) -> &HashMap<usize, String> {
        &self.current_assignments
    }

    pub fn propose_shard_reassignment(&mut self, assignments: Vec<ShardAssignment>) -> u64 {
        let proposal_id = self.next_proposal_id;
        self.next_proposal_id += 1;

        let proposal = ConsensusProposal {
            id: proposal_id,
            assignments,
            proposer: self.node_addr.clone(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };

        self.proposals.insert(proposal_id, proposal.clone());
        self.votes.insert(proposal_id, HashMap::new());

        // Auto-vote for own proposal
        let _ = self.vote_on_proposal(proposal_id, self.node_addr.clone(), true);

        proposal_id
    }

    pub fn vote_on_proposal(
        &mut self,
        proposal_id: u64,
        voter: String,
        approve: bool,
    ) -> Result<()> {
        if let Some(votes) = self.votes.get_mut(&proposal_id) {
            votes.insert(voter, approve);
        }
        Ok(())
    }

    pub async fn check_consensus(&mut self, proposal_id: u64) -> Option<&ConsensusProposal> {
        let registry = self.registry.lock().await;
        let active_nodes = registry.get_active_nodes();
        let total_nodes = active_nodes.len();

        if let Some(votes) = self.votes.get(&proposal_id) {
            let approve_votes = votes.values().filter(|&&v| v).count();
            let majority = (total_nodes / 2) + 1;

            #[allow(clippy::collapsible_if)]
            if approve_votes >= majority {
                if let Some(proposal) = self.proposals.get(&proposal_id) {
                    // Apply the assignments
                    for assignment in &proposal.assignments {
                        self.current_assignments
                            .insert(assignment.shard_id, assignment.node_addr.clone());
                    }
                    return Some(proposal);
                }
            }
        }
        None
    }

    pub fn reassign_shards_on_node_failure(
        &mut self,
        failed_node: &str,
        available_nodes: &[String],
    ) -> Vec<ShardAssignment> {
        let mut new_assignments = Vec::new();
        if available_nodes.is_empty() {
            return new_assignments;
        }

        // Find shards assigned to the failed node
        let mut failed_shards: Vec<usize> = self
            .current_assignments
            .iter()
            .filter(|(_, node)| *node == failed_node)
            .map(|(shard_id, _)| *shard_id)
            .collect();

        // Keep reassignment deterministic so tests can rely on stable ordering
        failed_shards.sort_unstable();

        // Reassign to available nodes in round-robin fashion and update current assignments
        for (i, shard_id) in failed_shards.into_iter().enumerate() {
            let target_node = &available_nodes[i % available_nodes.len()];
            self.current_assignments
                .insert(shard_id, target_node.clone());
            new_assignments.push(ShardAssignment {
                shard_id,
                node_addr: target_node.clone(),
            });
        }

        new_assignments
    }

    pub fn get_shard_location(&self, shard_id: usize) -> Option<&String> {
        self.current_assignments.get(&shard_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_compression_decompression() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let compressed = CompressedData::compress(&data).unwrap();
        let decompressed = compressed.decompress().unwrap();
        assert_eq!(data, decompressed);
    }

    #[test]
    fn test_quantize_dequantize() {
        let data = vec![-5.0f32, 0.0, 5.0, 10.0];
        let quantized = quantize_f32_to_u8(&data);
        let dequantized = dequantize_u8_to_f32(&quantized);
        // Check approximate equality due to quantization
        for (orig, deq) in data.iter().zip(dequantized.iter()) {
            assert!((orig - deq).abs() < 0.5); // within 0.5 for 8bit quantization
        }
    }

    #[test]
    fn test_latency_predictor() {
        let mut predictor = LatencyPredictor::new(0.1);
        predictor.add_measurement(10.0);
        predictor.add_measurement(12.0);
        predictor.add_measurement(11.0);
        let prediction = predictor.predict();
        assert!(prediction > 10.0 && prediction < 12.0); // EMA should be between values
    }

    #[test]
    fn test_routing_table() {
        let mut table = RoutingTable::new();
        table.add_route(0, "127.0.0.1:8081".to_string());
        table.add_route(0, "127.0.0.1:8082".to_string());
        let nodes = table.get_nodes(0).unwrap();
        assert_eq!(nodes.len(), 2);
        let predictor = LatencyPredictor::new(0.1);
        let best = table.get_best_node(0, &predictor).unwrap();
        assert_eq!(best, "127.0.0.1:8081");

        // Geographic routing
        table.add_node_location(
            "127.0.0.1:8081".to_string(),
            NodeLocation {
                latitude: 40.0,
                longitude: -74.0,
            },
        );
        table.add_node_location(
            "127.0.0.1:8082".to_string(),
            NodeLocation {
                latitude: 34.0,
                longitude: -118.0,
            },
        );
        let client_loc = NodeLocation {
            latitude: 37.0,
            longitude: -122.0,
        }; // Near LA
        let closest = table.get_closest_node(0, &client_loc).unwrap();
        assert_eq!(closest, "127.0.0.1:8082"); // Closer to LA
    }

    #[test]
    fn test_node_registry() {
        let mut registry = NodeRegistry::new();
        registry.register_node(
            "127.0.0.1:8081".to_string(),
            vec!["linear".to_string(), "gpu".to_string()],
        );
        registry.register_node(
            "127.0.0.1:8082".to_string(),
            vec!["transformer".to_string(), "cpu".to_string()],
        );

        let active = registry.get_active_nodes();
        assert_eq!(active.len(), 2);

        let gpu_nodes = registry.get_nodes_with_capability("gpu");
        assert_eq!(gpu_nodes, vec!["127.0.0.1:8081"]);
    }

    #[tokio::test]
    async fn test_consensus_manager_proposal() {
        let registry = Arc::new(tokio::sync::Mutex::new(NodeRegistry::new()));
        let mut consensus = ConsensusManager::new("127.0.0.1:8081".to_string(), registry.clone());

        let assignments = vec![
            ShardAssignment {
                shard_id: 0,
                node_addr: "127.0.0.1:8081".to_string(),
            },
            ShardAssignment {
                shard_id: 1,
                node_addr: "127.0.0.1:8082".to_string(),
            },
        ];

        let proposal_id = consensus.propose_shard_reassignment(assignments.clone());
        assert_eq!(proposal_id, 1);

        // Check that proposal was stored
        assert!(consensus.proposals.contains_key(&proposal_id));
    }

    #[tokio::test]
    async fn test_consensus_manager_voting() {
        let registry = Arc::new(tokio::sync::Mutex::new(NodeRegistry::new()));
        {
            let mut reg = registry.lock().await;
            reg.register_node("127.0.0.1:8081".to_string(), vec![]);
            reg.register_node("127.0.0.1:8082".to_string(), vec![]);
            reg.register_node("127.0.0.1:8083".to_string(), vec![]);
        }
        let mut consensus = ConsensusManager::new("127.0.0.1:8081".to_string(), registry.clone());

        let assignments = vec![ShardAssignment {
            shard_id: 0,
            node_addr: "127.0.0.1:8081".to_string(),
        }];
        let proposal_id = consensus.propose_shard_reassignment(assignments);

        // Vote yes from another node (simulate)
        let _ = consensus.vote_on_proposal(proposal_id, "127.0.0.1:8082".to_string(), true);

        // With 3 nodes, majority is 2. We have 1 auto-vote + 1 manual = 2, should reach consensus
        let result = consensus.check_consensus(proposal_id).await;
        assert!(result.is_some());

        // Check assignment was applied
        assert_eq!(
            consensus.get_shard_location(0),
            Some(&"127.0.0.1:8081".to_string())
        );
    }

    #[tokio::test]
    async fn test_shard_reassignment_on_failure() {
        let registry = Arc::new(tokio::sync::Mutex::new(NodeRegistry::new()));
        {
            let mut reg = registry.lock().await;
            reg.register_node("127.0.0.1:8081".to_string(), vec![]);
            reg.register_node("127.0.0.1:8082".to_string(), vec![]);
        }
        let mut consensus = ConsensusManager::new("127.0.0.1:8081".to_string(), registry.clone());

        // Set up initial assignments
        consensus
            .current_assignments
            .insert(0, "127.0.0.1:8081".to_string());
        consensus
            .current_assignments
            .insert(1, "127.0.0.1:8082".to_string());
        consensus
            .current_assignments
            .insert(2, "127.0.0.1:8082".to_string());

        // Simulate node failure by removing from registry
        {
            let mut reg = registry.lock().await;
            reg.nodes.remove("127.0.0.1:8082");
            // Update heartbeat for remaining node to ensure it's active
            reg.update_heartbeat("127.0.0.1:8081");
        }

        // Simulate node failure
        let available_nodes = vec!["127.0.0.1:8081".to_string()];
        let new_assignments =
            consensus.reassign_shards_on_node_failure("127.0.0.1:8082", &available_nodes);

        // Should reassign shards 1 and 2 to remaining node
        eprintln!("new_assignments={:?}", new_assignments);
        assert_eq!(new_assignments.len(), 2);
        assert_eq!(new_assignments[0].shard_id, 1);
        assert_eq!(new_assignments[1].shard_id, 2);
        assert_eq!(new_assignments[0].node_addr, "127.0.0.1:8081");
        assert_eq!(new_assignments[1].node_addr, "127.0.0.1:8081");
    }
}
