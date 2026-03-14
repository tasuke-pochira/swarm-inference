# API Reference

*Work in Progress*

Swarm Inference provides multiple interfaces for interaction, depending on the role (Coordinator vs Node) and the client type.

## Inter-Node Protocol (QUIC)

The primary communication channel between nodes is a high-performance QUIC protocol.

### Message Types

| Message | Description |
| --- | --- |
| `Prompt` | Initial request from Coordinator to Node. |
| `Intermediate` | Passing tensor data (compressed) between nodes. |
| `Result` | Final output result sent back to the Coordinator. |
| `Heartbeat` | Periodic health status signal. |

## REST API (Coordinator)

Exposed on the Coordinator for high-level management and inference requests.

- `POST /v1/inference`: Run an inference task.
- `GET /v1/health`: Cluster-wide health status.
- `GET /v1/nodes`: List of active nodes in the swarm.

## WebSocket API (Streaming)

For real-time streaming of tokens or intermediate results.

- `WS /v1/stream`: Persistent connection for bidirectional communication.

## Metrics (Prometheus)

All nodes expose a `/metrics` endpoint for Prometheus scraping.

Key metrics:
- `swarm_inference_requests_total`: Total count of inference requests.
- `swarm_inference_latency_ms`: Histograms of inference duration.
- `swarm_gpu_memory_usage`: Real-time GPU memory tracking.
