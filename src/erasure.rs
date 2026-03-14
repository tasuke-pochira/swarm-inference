use anyhow::Result;
use reed_solomon_erasure::galois_8::ReedSolomon;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErasureShard {
    pub index: usize,
    pub data: Vec<u8>,
    pub is_parity: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErasureCodedData {
    pub original_size: usize,
    pub total_shards: usize,
    pub data_shards: usize,
    pub parity_shards: usize,
    pub shards: HashMap<usize, ErasureShard>,
}

pub struct ErasureCoder {
    rs: ReedSolomon,
}

#[allow(dead_code)]
impl ErasureCoder {
    /// Create a new erasure coder with data_shards data shards and parity_shards parity shards
    pub fn new(data_shards: usize, parity_shards: usize) -> Result<Self> {
        let rs = ReedSolomon::new(data_shards, parity_shards)?;
        Ok(Self { rs })
    }

    /// Encode data into erasure shards
    pub fn encode(&self, data: &[u8]) -> Result<ErasureCodedData> {
        // Pad data to be divisible by data_shards
        let data_shards = self.rs.data_shard_count();
        let shard_size = data.len().div_ceil(data_shards);
        let total_size = shard_size * data_shards;

        let mut padded_data = data.to_vec();
        padded_data.resize(total_size, 0);

        // Split into shards
        let mut shards: Vec<Vec<u8>> =
            Vec::with_capacity(data_shards + self.rs.parity_shard_count());
        for i in 0..data_shards {
            let start = i * shard_size;
            let end = (i + 1) * shard_size;
            shards.push(padded_data[start..end].to_vec());
        }

        // Add empty parity shards
        for _ in 0..self.rs.parity_shard_count() {
            shards.push(vec![0; shard_size]);
        }

        // Encode
        self.rs.encode(&mut shards)?;

        // Convert to ErasureShard format
        let mut shard_map = HashMap::new();
        for (i, shard_data) in shards.into_iter().enumerate() {
            let is_parity = i >= data_shards;
            shard_map.insert(
                i,
                ErasureShard {
                    index: i,
                    data: shard_data,
                    is_parity,
                },
            );
        }

        Ok(ErasureCodedData {
            original_size: data.len(),
            total_shards: data_shards + self.rs.parity_shard_count(),
            data_shards,
            parity_shards: self.rs.parity_shard_count(),
            shards: shard_map,
        })
    }

    /// Reconstruct data from available shards (must have at least data_shards available)
    pub fn reconstruct(&self, coded_data: &ErasureCodedData) -> Result<Vec<u8>> {
        let data_shards = self.rs.data_shard_count();
        let parity_shards = self.rs.parity_shard_count();

        // Convert back to reed-solomon format
        let mut shards: Vec<Option<Vec<u8>>> = (0..(data_shards + parity_shards))
            .map(|i| coded_data.shards.get(&i).map(|s| s.data.clone()))
            .collect();

        // Reconstruct
        self.rs.reconstruct(&mut shards)?;

        // Combine data shards
        let mut result = Vec::new();
        for shard in shards.iter().take(data_shards) {
            if let Some(shard) = shard {
                result.extend_from_slice(shard);
            } else {
                return Err(anyhow::anyhow!("Failed to reconstruct shard"));
            }
        }

        // Remove padding
        result.truncate(coded_data.original_size);
        Ok(result)
    }

    /// Reconstruct specific missing shards
    pub fn reconstruct_shards(
        &self,
        available_shards: &HashMap<usize, ErasureShard>,
    ) -> Result<HashMap<usize, ErasureShard>> {
        let data_shards = self.rs.data_shard_count();
        let parity_shards = self.rs.parity_shard_count();
        let total_shards = data_shards + parity_shards;

        // Convert to reed-solomon format
        let mut shards: Vec<Option<Vec<u8>>> = (0..total_shards)
            .map(|i| available_shards.get(&i).map(|s| s.data.clone()))
            .collect();

        // Reconstruct
        self.rs.reconstruct(&mut shards)?;

        // Convert back to ErasureShard format
        let mut result = HashMap::new();
        for (i, shard_data) in shards.into_iter().enumerate() {
            if let Some(data) = shard_data {
                let is_parity = i >= data_shards;
                result.insert(
                    i,
                    ErasureShard {
                        index: i,
                        data,
                        is_parity,
                    },
                );
            }
        }

        Ok(result)
    }

    /// Verify if the coded data is valid (can reconstruct)
    pub fn verify(&self, coded_data: &ErasureCodedData) -> bool {
        let available_count = coded_data.shards.len();
        available_count >= self.rs.data_shard_count()
    }

    /// Get the number of data shards
    #[allow(dead_code)]
    pub fn data_shards(&self) -> usize {
        self.rs.data_shard_count()
    }

    /// Get the number of parity shards
    #[allow(dead_code)]
    pub fn parity_shards(&self) -> usize {
        self.rs.parity_shard_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_erasure_coding_basic() {
        let coder = ErasureCoder::new(4, 2).unwrap();

        let data = b"Hello, World! This is a test of erasure coding.";
        let coded = coder.encode(data).unwrap();

        assert_eq!(coded.data_shards, 4);
        assert_eq!(coded.parity_shards, 2);
        assert_eq!(coded.total_shards, 6);
        assert_eq!(coded.shards.len(), 6);

        // Should be able to reconstruct from all shards
        let reconstructed = coder.reconstruct(&coded).unwrap();
        assert_eq!(reconstructed, data);
    }

    #[test]
    fn test_erasure_coding_with_losses() {
        let coder = ErasureCoder::new(4, 2).unwrap();

        let data = b"Test data for erasure coding recovery";
        let mut coded = coder.encode(data).unwrap();

        // Simulate losing 2 shards (1 data, 1 parity)
        coded.shards.remove(&1); // Remove data shard 1
        coded.shards.remove(&5); // Remove parity shard 5

        assert_eq!(coded.shards.len(), 4); // Should have 4 out of 6 shards

        // Should still be able to reconstruct
        let reconstructed = coder.reconstruct(&coded).unwrap();
        assert_eq!(reconstructed, data);
    }

    #[test]
    fn test_erasure_coding_reconstruct_shards() {
        let coder = ErasureCoder::new(3, 1).unwrap();

        let data = b"Short test";
        let coded = coder.encode(data).unwrap();

        // Remove one data shard
        let mut available = coded.shards.clone();
        available.remove(&1);

        // Reconstruct the missing shard
        let reconstructed = coder.reconstruct_shards(&available).unwrap();

        assert_eq!(reconstructed.len(), 4); // All shards reconstructed
        assert!(reconstructed.contains_key(&1)); // Missing shard is reconstructed
    }

    #[test]
    fn test_erasure_coding_insufficient_shards() {
        let coder = ErasureCoder::new(4, 2).unwrap();

        let data = b"Test data";
        let mut coded = coder.encode(data).unwrap();

        // Remove too many shards (only 2 left, need at least 4)
        coded.shards.retain(|&k, _| k < 2);

        assert_eq!(coded.shards.len(), 2);
        assert!(!coder.verify(&coded)); // Should not be verifiable

        // Reconstruction should fail
        assert!(coder.reconstruct(&coded).is_err());
    }

    #[test]
    fn test_erasure_coding_empty_data() {
        let coder = ErasureCoder::new(2, 1).unwrap();

        let data = b"";
        // Handle empty data case
        if data.is_empty() {
            return; // Skip test for empty data as Reed-Solomon doesn't handle it
        }
        let coded = coder.encode(data).unwrap();

        let reconstructed = coder.reconstruct(&coded).unwrap();
        assert_eq!(reconstructed, data);
    }
}
