use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};

/// Represents the state of an ongoing inference that can be checkpointed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceCheckpoint {
    /// Unique identifier for this inference session
    pub inference_id: String,
    /// Current step in the inference chain (0 = coordinator, 1 = first node, etc.)
    pub current_step: usize,
    /// Total number of steps in the chain
    pub total_steps: usize,
    /// Intermediate results from previous steps
    pub intermediate_results: Vec<Vec<f32>>,
    /// Current input data being processed
    pub current_input: Vec<f32>,
    /// Timestamp when checkpoint was created
    pub timestamp: u64,
    /// Version of the checkpoint format
    pub version: u32,
    /// Metadata about the inference (prompt_id, etc.)
    pub metadata: HashMap<String, String>,
}

impl InferenceCheckpoint {
    /// Create a new checkpoint
    pub fn new(inference_id: String, total_steps: usize) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            inference_id,
            current_step: 0,
            total_steps,
            intermediate_results: Vec::new(),
            current_input: Vec::new(),
            timestamp,
            version: 1,
            metadata: HashMap::new(),
        }
    }

    /// Update the checkpoint with new intermediate results
    pub fn update_progress(
        &mut self,
        step: usize,
        intermediate_result: Vec<f32>,
        next_input: Vec<f32>,
    ) {
        self.current_step = step;
        self.intermediate_results.push(intermediate_result);
        self.current_input = next_input;
        self.timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
    }

    /// Check if the inference is complete
    #[allow(dead_code)]
    pub fn is_complete(&self) -> bool {
        self.current_step >= self.total_steps
    }

    /// Get the final result if inference is complete
    #[allow(dead_code)]
    pub fn get_final_result(&self) -> Option<&Vec<f32>> {
        if self.is_complete() && !self.intermediate_results.is_empty() {
            self.intermediate_results.last()
        } else {
            None
        }
    }

    /// Serialize checkpoint to bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        bincode::serialize(self).map_err(Into::into)
    }

    /// Deserialize checkpoint from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        bincode::deserialize(data).map_err(Into::into)
    }
}

/// Manager for handling checkpoints with persistence
pub struct CheckpointManager {
    checkpoints: HashMap<String, InferenceCheckpoint>,
    insertion_order: VecDeque<String>,
    max_checkpoints: usize,
}

impl CheckpointManager {
    pub fn new(max_checkpoints: usize) -> Self {
        Self {
            checkpoints: HashMap::new(),
            insertion_order: VecDeque::new(),
            max_checkpoints,
        }
    }

    /// Save a checkpoint.
    ///
    /// Evicts the oldest checkpoint when capacity is exceeded.
    pub fn save_checkpoint(&mut self, checkpoint: InferenceCheckpoint) -> Result<()> {
        let id = checkpoint.inference_id.clone();

        // If a checkpoint already exists, remove it from insertion order so we can reinsert at the back.
        if self.checkpoints.contains_key(&id) {
            self.insertion_order.retain(|x| x != &id);
        }

        // Evict oldest if at capacity
        #[allow(clippy::collapsible_if)]
        if self.checkpoints.len() >= self.max_checkpoints {
            if let Some(oldest_id) = self.insertion_order.pop_front() {
                self.checkpoints.remove(&oldest_id);
            }
        }

        self.checkpoints.insert(id.clone(), checkpoint);
        self.insertion_order.push_back(id);
        Ok(())
    }

    /// Load a checkpoint
    pub fn load_checkpoint(&self, inference_id: &str) -> Option<&InferenceCheckpoint> {
        self.checkpoints.get(inference_id)
    }

    /// Remove a checkpoint
    #[allow(dead_code)]
    pub fn remove_checkpoint(&mut self, inference_id: &str) -> bool {
        self.checkpoints.remove(inference_id).is_some()
    }

    /// List all checkpoint IDs
    #[allow(dead_code)]
    pub fn list_checkpoints(&self) -> Vec<String> {
        self.checkpoints.keys().cloned().collect()
    }

    /// Clean up old checkpoints based on age
    #[allow(dead_code)]
    pub fn cleanup_old_checkpoints(&mut self, max_age_seconds: u64) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        self.checkpoints
            .retain(|_, cp| now - cp.timestamp < max_age_seconds);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_creation_and_update() {
        let mut checkpoint = InferenceCheckpoint::new("test_inference".to_string(), 3);

        assert_eq!(checkpoint.inference_id, "test_inference");
        assert_eq!(checkpoint.current_step, 0);
        assert_eq!(checkpoint.total_steps, 3);
        assert!(!checkpoint.is_complete());

        checkpoint.update_progress(1, vec![1.0, 2.0], vec![3.0, 4.0]);
        assert_eq!(checkpoint.current_step, 1);
        assert_eq!(checkpoint.intermediate_results.len(), 1);
        assert_eq!(checkpoint.current_input, vec![3.0, 4.0]);

        checkpoint.update_progress(2, vec![5.0, 6.0], vec![7.0, 8.0]);
        assert_eq!(checkpoint.current_step, 2);
        assert_eq!(checkpoint.intermediate_results.len(), 2);
    }

    #[test]
    fn test_checkpoint_completion() {
        let mut checkpoint = InferenceCheckpoint::new("test_inference".to_string(), 2);

        checkpoint.update_progress(1, vec![1.0], vec![2.0]);
        assert!(!checkpoint.is_complete());

        checkpoint.update_progress(2, vec![3.0], vec![4.0]);
        assert!(checkpoint.is_complete());
        assert_eq!(checkpoint.get_final_result(), Some(&vec![3.0]));
    }

    #[test]
    fn test_checkpoint_serialization() {
        let mut checkpoint = InferenceCheckpoint::new("test_inference".to_string(), 2);
        checkpoint.update_progress(1, vec![1.0, 2.0], vec![3.0, 4.0]);

        let serialized = checkpoint.to_bytes().unwrap();
        let deserialized = InferenceCheckpoint::from_bytes(&serialized).unwrap();

        assert_eq!(deserialized.inference_id, checkpoint.inference_id);
        assert_eq!(deserialized.current_step, checkpoint.current_step);
        assert_eq!(
            deserialized.intermediate_results,
            checkpoint.intermediate_results
        );
    }

    #[test]
    fn test_checkpoint_manager() {
        let mut manager = CheckpointManager::new(2);

        let cp1 = InferenceCheckpoint::new("inference_1".to_string(), 3);
        let cp2 = InferenceCheckpoint::new("inference_2".to_string(), 3);
        let cp3 = InferenceCheckpoint::new("inference_3".to_string(), 3);

        manager.save_checkpoint(cp1).unwrap();
        manager.save_checkpoint(cp2).unwrap();
        manager.save_checkpoint(cp3).unwrap(); // Should evict oldest

        assert!(manager.load_checkpoint("inference_1").is_none()); // Should be evicted
        assert!(manager.load_checkpoint("inference_2").is_some());
        assert!(manager.load_checkpoint("inference_3").is_some());

        assert!(manager.remove_checkpoint("inference_2"));
        assert!(manager.load_checkpoint("inference_2").is_none());
    }
}
