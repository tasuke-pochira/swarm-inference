use crate::erasure::ErasureCoder;
use crate::gpu::create_gpu_backend;
use crate::kv_cache::KVCache;
use crate::metrics::{GPU_OP_TIME_MS, TENSOR_OP_TIME_MS};
use anyhow::Result;
use async_trait::async_trait;
use std::sync::Arc;
use tracing::warn;

#[async_trait]
pub trait ModelShard: Send + Sync + 'static {
    async fn process(&self, input: Vec<f32>, kv_cache: Option<Arc<KVCache>>) -> Result<Vec<f32>>;
}

#[derive(Debug, Clone)]
pub enum ModelArchitecture {
    Linear,
    #[allow(dead_code)]
    Transformer,
    #[allow(dead_code)]
    Mock,
}

#[allow(dead_code)]
pub struct MockShard {
    pub id: usize,
}

#[async_trait]
impl ModelShard for MockShard {
    async fn process(&self, input: Vec<f32>, _kv_cache: Option<Arc<KVCache>>) -> Result<Vec<f32>> {
        // Mock: add id to each element
        Ok(input.into_iter().map(|x| x + self.id as f32).collect())
    }
}

pub struct LinearShard {
    weights: Vec<f32>,
    bias: Vec<f32>,
    in_dim: usize,
    out_dim: usize,
}

impl LinearShard {
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        // Simple deterministic initialization
        let mut weights = Vec::with_capacity(in_dim * out_dim);
        for i in 0..(in_dim * out_dim) {
            weights.push(((i % 10) as f32) * 0.01);
        }
        let bias = vec![0.0; out_dim];
        Self {
            weights,
            bias,
            in_dim,
            out_dim,
        }
    }

    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let start = std::time::Instant::now();
        let mut out = vec![0.0; self.out_dim];
        for (o, out_val) in out.iter_mut().enumerate().take(self.out_dim) {
            let mut sum = self.bias[o];
            let weight_slice = &self.weights[o * self.in_dim..(o + 1) * self.in_dim];
            sum += input
                .iter()
                .zip(weight_slice.iter())
                .map(|(&input_val, &weight_val)| input_val * weight_val)
                .sum::<f32>();
            *out_val = sum;
        }
        let elapsed = start.elapsed().as_millis() as u64;
        TENSOR_OP_TIME_MS.fetch_add(elapsed, std::sync::atomic::Ordering::Relaxed);
        out
    }
}

pub struct RealShard {
    pub architecture: ModelArchitecture,
    pub layer: LinearShard,
}

impl RealShard {
    pub fn new(architecture: ModelArchitecture, in_dim: usize, out_dim: usize) -> Self {
        Self {
            architecture,
            layer: LinearShard::new(in_dim, out_dim),
        }
    }
}

pub struct GpuShard {
    cpu_layer: LinearShard,
    backend: Option<Arc<dyn crate::gpu::GpuCompute>>,
}

impl GpuShard {
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        let backend = match create_gpu_backend() {
            Ok(b) => Some(Arc::from(b)),
            Err(e) => {
                warn!("GPU backend init failed; falling back to CPU: {}", e);
                None
            }
        };

        Self {
            cpu_layer: LinearShard::new(in_dim, out_dim),
            backend,
        }
    }

    pub fn gpu_forward(&self, input: &[f32]) -> Vec<f32> {
        let start = std::time::Instant::now();
        let result = if let Some(backend) = &self.backend {
            backend
                .matmul(
                    &self.cpu_layer.weights,
                    input,
                    self.cpu_layer.in_dim,
                    self.cpu_layer.out_dim,
                    &self.cpu_layer.bias,
                )
                .unwrap_or_else(|e| {
                    warn!("GPU matmul failed; falling back to CPU: {}", e);
                    self.cpu_layer.forward(input)
                })
        } else {
            self.cpu_layer.forward(input)
        };

        let elapsed = start.elapsed().as_millis() as u64;
        GPU_OP_TIME_MS.fetch_add(elapsed, std::sync::atomic::Ordering::Relaxed);
        result
    }
}

#[async_trait]
impl ModelShard for GpuShard {
    async fn process(&self, input: Vec<f32>, _kv_cache: Option<Arc<KVCache>>) -> Result<Vec<f32>> {
        Ok(self.gpu_forward(&input))
    }
}

#[async_trait]
impl ModelShard for RealShard {
    async fn process(&self, input: Vec<f32>, _kv_cache: Option<Arc<KVCache>>) -> Result<Vec<f32>> {
        match self.architecture {
            ModelArchitecture::Linear => Ok(self.layer.forward(&input)),
            ModelArchitecture::Transformer => {
                // Simple transformer-like processing: single layer with ReLU activation
                let hidden = self.layer.forward(&input);
                Ok(hidden.iter().map(|&x| x.max(0.0)).collect())
            }
            ModelArchitecture::Mock => {
                // Use MockShard logic
                Ok(input.into_iter().map(|x| x + 1.0).collect())
            }
        }
    }
}

pub struct ShardManager {
    pub shards: Vec<Arc<dyn ModelShard>>,
    #[allow(dead_code)]
    pub erasure_coder: Option<ErasureCoder>,
    #[allow(dead_code)]
    pub coded_shards: Option<crate::erasure::ErasureCodedData>,
}

#[allow(dead_code)]
impl ShardManager {
    /// Create a shard manager with automatic splitting
    pub fn new(
        base_architecture: ModelArchitecture,
        in_dim: usize,
        out_dim: usize,
        num_shards: usize,
    ) -> Self {
        let mut shards = Vec::new();
        let shard_out_dim = out_dim / num_shards;
        for _ in 0..num_shards {
            let shard = RealShard::new(base_architecture.clone(), in_dim, shard_out_dim);
            shards.push(Arc::new(shard) as Arc<dyn ModelShard>);
        }
        Self {
            shards,
            erasure_coder: None,
            coded_shards: None,
        }
    }

    /// Create a shard manager with erasure coding for redundancy
    pub fn new_with_erasure_coding(
        base_architecture: ModelArchitecture,
        in_dim: usize,
        out_dim: usize,
        data_shards: usize,
        parity_shards: usize,
    ) -> Result<Self> {
        let erasure_coder = ErasureCoder::new(data_shards, parity_shards)?;
        let mut shards = Vec::new();
        let shard_out_dim = out_dim / data_shards;

        for _ in 0..data_shards {
            let shard = RealShard::new(base_architecture.clone(), in_dim, shard_out_dim);
            shards.push(Arc::new(shard) as Arc<dyn ModelShard>);
        }

        Ok(Self {
            shards,
            erasure_coder: Some(erasure_coder),
            coded_shards: None,
        })
    }

    pub fn get_shard(&self, index: usize) -> Option<Arc<dyn ModelShard>> {
        self.shards.get(index).cloned()
    }

    #[allow(dead_code)]
    pub fn list_checkpoints(&self) -> Vec<String> {
        // Placeholder for actual checkpoint listing logic
        vec![]
    }

    /// Encode model data with erasure coding for redundancy
    #[allow(dead_code)]
    pub fn cleanup_old_checkpoints(&mut self, max_age_seconds: u64) -> Result<()> {
        // Placeholder for actual cleanup logic
        let _ = max_age_seconds; // Suppress unused variable warning
        Ok(())
    }

    pub fn num_shards(&self) -> usize {
        self.shards.len()
    }

    /// Encode model data with erasure coding for redundancy
    pub fn encode_model_data(&mut self, model_data: &[u8]) -> Result<()> {
        if let Some(coder) = &self.erasure_coder {
            self.coded_shards = Some(coder.encode(model_data)?);
        }
        Ok(())
    }

    /// Reconstruct model data from available shards
    pub fn reconstruct_model_data(&self) -> Result<Vec<u8>> {
        if let (Some(coder), Some(coded_data)) = (&self.erasure_coder, &self.coded_shards) {
            coder.reconstruct(coded_data)
        } else {
            Err(anyhow::anyhow!(
                "No erasure coding configured or no coded data available"
            ))
        }
    }

    /// Check if we can reconstruct data with current available shards
    pub fn can_reconstruct(&self, available_shard_indices: &[usize]) -> bool {
        if let Some(coder) = &self.erasure_coder {
            let available_count = available_shard_indices.len();
            available_count >= coder.data_shards()
        } else {
            false
        }
    }

    /// Reconstruct missing shards from available ones
    pub fn reconstruct_missing_shards(
        &self,
        available_shard_indices: &[usize],
    ) -> Result<Vec<usize>> {
        if let (Some(coder), Some(coded_data)) = (&self.erasure_coder, &self.coded_shards) {
            let mut available_shards = std::collections::HashMap::new();
            for &idx in available_shard_indices {
                if let Some(shard) = coded_data.shards.get(&idx) {
                    available_shards.insert(idx, shard.clone());
                }
            }

            let reconstructed = coder.reconstruct_shards(&available_shards)?;
            let missing_indices: Vec<usize> = (0..coded_data.total_shards)
                .filter(|i| !available_shard_indices.contains(i))
                .collect();

            Ok(missing_indices
                .into_iter()
                .filter(|i| reconstructed.contains_key(i))
                .collect())
        } else {
            Err(anyhow::anyhow!("No erasure coding configured"))
        }
    }

    /// Process all shards in parallel for the given input.
    ///
    /// Returns a vector of results, one for each shard.
    #[allow(dead_code)]
    pub async fn process_parallel(
        &self,
        input: Vec<f32>,
        kv_cache: Option<Arc<KVCache>>,
    ) -> Result<Vec<Vec<f32>>> {
        let futures = self.shards.iter().map(|shard| {
            let input_clone = input.clone();
            let kv_cache_clone = kv_cache.clone();
            async move { shard.process(input_clone, kv_cache_clone).await }
        });

        let results = futures::future::join_all(futures).await;
        results.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_linear_shard() {
        let shard = LinearShard::new(2, 2);
        let input = vec![1.0, 2.0];
        let output = shard.forward(&input);
        assert_eq!(output.len(), 2);
        // Deterministic weights: [0.00, 0.01, 0.02, 0.03] for 2x2
        // out[0] = 1*0.00 + 2*0.01 + 0.0 = 0.02
        // out[1] = 1*0.02 + 2*0.03 + 0.0 = 0.08
        assert!((output[0] - 0.02).abs() < 1e-6);
        assert!((output[1] - 0.08).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_real_shard() {
        let shard = RealShard::new(ModelArchitecture::Linear, 2, 2);
        let input = vec![1.0, 2.0];
        let output = shard.process(input, None).await.unwrap();
        assert_eq!(output.len(), 2);
    }

    #[tokio::test]
    async fn test_transformer_shard() {
        let shard = RealShard::new(ModelArchitecture::Transformer, 2, 2);
        let input = vec![1.0, 2.0];
        let output = shard.process(input, None).await.unwrap();
        assert_eq!(output.len(), 2);
    }

    #[tokio::test]
    async fn test_shard_manager() {
        let manager = ShardManager::new(ModelArchitecture::Linear, 2, 4, 2);
        assert_eq!(manager.num_shards(), 2);
        let shard0 = manager.get_shard(0).unwrap();
        let input = vec![1.0, 2.0];
        let output = shard0.process(input, None).await.unwrap();
        assert_eq!(output.len(), 2); // out_dim / num_shards = 4/2 = 2
    }

    #[test]
    fn test_shard_manager_with_erasure_coding() {
        let manager =
            ShardManager::new_with_erasure_coding(ModelArchitecture::Linear, 4, 8, 3, 1).unwrap();

        assert_eq!(manager.num_shards(), 3); // Only data shards
        assert!(manager.erasure_coder.is_some());
        assert!(manager.coded_shards.is_none()); // Not encoded yet
    }

    #[test]
    fn test_erasure_coding_model_data() {
        let mut manager =
            ShardManager::new_with_erasure_coding(ModelArchitecture::Linear, 4, 8, 2, 1).unwrap();

        let model_data = b"model weights and parameters";
        manager.encode_model_data(model_data).unwrap();

        assert!(manager.coded_shards.is_some());

        if let Some(coded) = &manager.coded_shards {
            assert_eq!(coded.data_shards, 2);
            assert_eq!(coded.parity_shards, 1);
            assert_eq!(coded.total_shards, 3);
            assert_eq!(coded.original_size, model_data.len());
        }

        // Test reconstruction
        let reconstructed = manager.reconstruct_model_data().unwrap();
        assert_eq!(reconstructed, model_data);
    }

    #[test]
    fn test_erasure_coding_reconstruction() {
        let mut manager =
            ShardManager::new_with_erasure_coding(ModelArchitecture::Linear, 4, 8, 3, 2).unwrap();

        let model_data = b"test model data for reconstruction";
        manager.encode_model_data(model_data).unwrap();

        // Test can_reconstruct
        assert!(manager.can_reconstruct(&[0, 1, 2])); // All data shards
        assert!(manager.can_reconstruct(&[0, 1, 4])); // 2 data + 1 parity
        assert!(!manager.can_reconstruct(&[0, 1])); // Only 2 shards, need 3

        // Test reconstruct_missing_shards
        let missing = manager.reconstruct_missing_shards(&[0, 1, 2]).unwrap();
        assert_eq!(missing.len(), 2); // Should reconstruct parity shards 3 and 4
        assert!(missing.contains(&3));
        assert!(missing.contains(&4));
    }
}
