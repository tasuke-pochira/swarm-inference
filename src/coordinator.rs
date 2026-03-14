use crate::auto_scaling::{AutoScaler, ScalingService};
use crate::checkpoint::{CheckpointManager, InferenceCheckpoint};
use crate::config::Config;
use crate::network::{
    Message, create_client_endpoint, create_server_endpoint, open_bi_stream, quantize_f32_to_u8,
    quic_accept, quic_connect, receive_message, send_message,
};
use crate::{AuditResult, get_audit_logger};
use anyhow::Result;
use std::sync::Arc;
use tokio::sync::Mutex;

pub struct Coordinator {
    pub first_node_addr: String,
    pub listen_addr: String,
    pub checkpoint_manager: Arc<Mutex<CheckpointManager>>,
    pub scaling_service: Option<ScalingService>,
}

impl Coordinator {
    pub async fn new(
        first_node_addr: String,
        listen_addr: String,
        config: &Config,
    ) -> Result<Self> {
        let checkpoint_manager = Arc::new(Mutex::new(CheckpointManager::new(50))); // Keep up to 50 checkpoints

        // Initialize auto-scaling service if enabled
        let scaling_service = if config.auto_scaling.enabled {
            let auto_scaler = Arc::new(
                AutoScaler::new(config.auto_scaling.clone(), config.kubernetes.clone()).await?,
            );
            Some(ScalingService::new(
                auto_scaler,
                std::time::Duration::from_secs(config.auto_scaling.interval_seconds),
            ))
        } else {
            None
        };

        Ok(Self {
            first_node_addr,
            listen_addr,
            checkpoint_manager,
            scaling_service,
        })
    }

    /// Start the auto-scaling service if enabled
    pub async fn start_scaling_service(&self) -> Result<()> {
        if let Some(service) = &self.scaling_service {
            let service_clone = service.clone();
            tokio::spawn(async move {
                if let Err(e) = service_clone.run().await {
                    tracing::error!("Auto-scaling service error: {}", e);
                }
            });
        }
        Ok(())
    }

    pub async fn run_inference(&self, prompt: &str) -> Result<Vec<f32>> {
        let inference_id = format!(
            "inference_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis()
        );

        // Audit the inference request
        get_audit_logger().log_access_control(
            "coordinator",
            None,
            "inference",
            "start",
            AuditResult::Success,
            serde_json::json!({
                "inference_id": &inference_id,
                "prompt_length": prompt.len()
            }),
        );

        // Create initial checkpoint
        let mut checkpoint = InferenceCheckpoint::new(inference_id.clone(), 1); // Assume single step for now
        checkpoint
            .metadata
            .insert("prompt".to_string(), prompt.to_string());

        // For now, use dummy data - in a real implementation this would process the prompt
        let input_data = vec![1.0, 2.0]; // dummy
        let prompt_id = 0;
        let quantized_input = quantize_f32_to_u8(&input_data);

        // Save initial checkpoint
        {
            let mut manager = self.checkpoint_manager.lock().await;
            manager.save_checkpoint(checkpoint.clone())?;
        }

        let request = Message::InferenceRequest {
            prompt_id,
            quantized_input,
        };
        let endpoint = create_client_endpoint().await?;
        let conn = quic_connect(&endpoint, &self.first_node_addr).await?;
        let mut stream = open_bi_stream(&conn).await?;
        send_message(&mut stream, &request).await?;

        // Listen for result
        let endpoint = create_server_endpoint(&self.listen_addr).await?;
        let conn = quic_accept(&endpoint).await?;
        let mut stream = open_bi_stream(&conn).await?;
        let response = receive_message(&mut stream).await?;
        match response {
            Message::InferenceResult {
                prompt_id: resp_id,
                output,
            } if resp_id == prompt_id => {
                // Update checkpoint with completion
                checkpoint.update_progress(1, output.clone(), vec![]);
                {
                    let mut manager = self.checkpoint_manager.lock().await;
                    manager.save_checkpoint(checkpoint)?;
                }

                // Audit the successful inference completion
                get_audit_logger().log_access_control(
                    "coordinator",
                    None,
                    "inference",
                    "complete",
                    AuditResult::Success,
                    serde_json::json!({
                        "inference_id": &inference_id,
                        "output_length": output.len()
                    }),
                );

                Ok(output)
            }
            _ => Err(anyhow::anyhow!("Unexpected response")),
        }
    }

    /// Resume inference from a checkpoint
    #[allow(dead_code)]
    pub async fn resume_inference(&self, inference_id: &str) -> Result<Option<Vec<f32>>> {
        let manager = self.checkpoint_manager.lock().await;
        if let Some(checkpoint) = manager.load_checkpoint(inference_id) {
            if checkpoint.is_complete() {
                return Ok(checkpoint.get_final_result().cloned());
            }
            // For now, just return None - in a full implementation this would
            // resume the inference from the checkpoint
            Ok(None)
        } else {
            Ok(None)
        }
    }

    /// List available checkpoints
    #[allow(dead_code)]
    pub async fn list_checkpoints(&self) -> Result<Vec<String>> {
        let manager = self.checkpoint_manager.lock().await;
        Ok(manager.list_checkpoints())
    }
}
