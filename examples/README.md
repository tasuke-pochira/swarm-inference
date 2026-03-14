# Swarm Inference Examples

This directory contains example applications and tutorials for using the Swarm Inference Protocol.

## Basic Inference Example

A simple client that sends inference requests to a swarm cluster.

### Python Client

```python
#!/usr/bin/env python3
"""
Basic Swarm Inference Client Example

This example demonstrates how to connect to a Swarm Inference cluster
and run inference requests using the REST API.
"""

import requests
import json
import time
from typing import Dict, Any

class SwarmClient:
    def __init__(self, coordinator_url: str = "http://localhost:8080"):
        self.coordinator_url = coordinator_url.rstrip('/')
        self.session = requests.Session()

    def run_inference(self, model: str, input_text: str, **parameters) -> Dict[str, Any]:
        """
        Run inference on the swarm cluster.

        Args:
            model: Model identifier (e.g., 'llama-2-7b', 'gpt-2-medium')
            input_text: Input text for inference
            **parameters: Additional model parameters

        Returns:
            Dictionary containing inference results
        """
        payload = {
            "model": model,
            "input": input_text,
            "parameters": parameters
        }

        response = self.session.post(
            f"{self.coordinator_url}/api/v1/inference",
            json=payload,
            timeout=30
        )
        response.raise_for_status()

        return response.json()

    def get_cluster_status(self) -> Dict[str, Any]:
        """Get the current status of the swarm cluster."""
        response = self.session.get(f"{self.coordinator_url}/api/v1/status")
        response.raise_for_status()
        return response.json()

def main():
    # Initialize client
    client = SwarmClient("http://localhost:8080")

    # Check cluster status
    try:
        status = client.get_cluster_status()
        print(f"Cluster status: {status}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to connect to cluster: {e}")
        return

    # Example inference requests
    examples = [
        {
            "model": "llama-2-7b-chat",
            "input": "Explain quantum computing in simple terms",
            "temperature": 0.7,
            "max_tokens": 200
        },
        {
            "model": "gpt-2-medium",
            "input": "Write a haiku about artificial intelligence",
            "temperature": 0.8,
            "max_tokens": 50
        }
    ]

    for i, example in enumerate(examples, 1):
        print(f"\n--- Example {i} ---")
        print(f"Model: {example['model']}")
        print(f"Input: {example['input']}")

        try:
            start_time = time.time()
            result = client.run_inference(**example)
            end_time = time.time()

            print(f"Response time: {end_time - start_time:.2f}s")
            print(f"Output: {result.get('output', 'N/A')}")
            print(f"Confidence: {result.get('confidence', 'N/A')}")

        except requests.exceptions.RequestException as e:
            print(f"Inference failed: {e}")
        except KeyError as e:
            print(f"Unexpected response format: {e}")

if __name__ == "__main__":
    main()
```

### JavaScript/Node.js Client

```javascript
const axios = require('axios');

class SwarmInferenceClient {
  constructor(coordinatorUrl = 'http://localhost:8080') {
    this.client = axios.create({
      baseURL: coordinatorUrl,
      timeout: 30000,
    });
  }

  async runInference(model, input, parameters = {}) {
    try {
      const response = await this.client.post('/api/v1/inference', {
        model,
        input,
        parameters,
      });

      return response.data;
    } catch (error) {
      throw new Error(`Inference failed: ${error.message}`);
    }
  }

  async getClusterStatus() {
    try {
      const response = await this.client.get('/api/v1/status');
      return response.data;
    } catch (error) {
      throw new Error(`Status check failed: ${error.message}`);
    }
  }
}

// Usage example
async function main() {
  const client = new SwarmInferenceClient();

  try {
    // Check cluster status
    const status = await client.getClusterStatus();
    console.log('Cluster status:', status);

    // Run inference
    const result = await client.runInference(
      'llama-2-7b-chat',
      'What is the capital of France?',
      { temperature: 0.1, max_tokens: 50 }
    );

    console.log('Inference result:', result);
  } catch (error) {
    console.error('Error:', error.message);
  }
}

main();
```

### Go Client

```go
package main

import (
    "bytes"
    "encoding/json"
    "fmt"
    "io"
    "net/http"
    "time"
)

type SwarmClient struct {
    baseURL    string
    httpClient *http.Client
}

type InferenceRequest struct {
    Model      string                 `json:"model"`
    Input      string                 `json:"input"`
    Parameters map[string]interface{} `json:"parameters,omitempty"`
}

type InferenceResponse struct {
    Output     string                 `json:"output"`
    Confidence float64                `json:"confidence,omitempty"`
    Metadata   map[string]interface{} `json:"metadata,omitempty"`
}

func NewSwarmClient(baseURL string) *SwarmClient {
    return &SwarmClient{
        baseURL: baseURL,
        httpClient: &http.Client{Timeout: 30 * time.Second},
    }
}

func (c *SwarmClient) RunInference(req InferenceRequest) (*InferenceResponse, error) {
    jsonData, err := json.Marshal(req)
    if err != nil {
        return nil, fmt.Errorf("failed to marshal request: %w", err)
    }

    url := c.baseURL + "/api/v1/inference"
    resp, err := c.httpClient.Post(url, "application/json", bytes.NewBuffer(jsonData))
    if err != nil {
        return nil, fmt.Errorf("failed to send request: %w", err)
    }
    defer resp.Body.Close()

    if resp.StatusCode != http.StatusOK {
        body, _ := io.ReadAll(resp.Body)
        return nil, fmt.Errorf("inference failed with status %d: %s", resp.StatusCode, string(body))
    }

    var result InferenceResponse
    if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
        return nil, fmt.Errorf("failed to decode response: %w", err)
    }

    return &result, nil
}

func main() {
    client := NewSwarmClient("http://localhost:8080")

    request := InferenceRequest{
        Model: "llama-2-7b-chat",
        Input: "Explain machine learning briefly",
        Parameters: map[string]interface{}{
            "temperature": 0.7,
            "max_tokens":  100,
        },
    }

    result, err := client.RunInference(request)
    if err != nil {
        fmt.Printf("Error: %v\n", err)
        return
    }

    fmt.Printf("Output: %s\n", result.Output)
    fmt.Printf("Confidence: %.2f\n", result.Confidence)
}
```

## Advanced Examples

### Streaming Inference

For real-time streaming responses:

```python
import websocket
import json
import threading

class StreamingSwarmClient:
    def __init__(self, ws_url: str = "ws://localhost:8080/ws/inference"):
        self.ws_url = ws_url
        self.ws = None

    def on_message(self, ws, message):
        data = json.loads(message)
        if data.get('type') == 'partial':
            print(f"Partial: {data['content']}", end='', flush=True)
        elif data.get('type') == 'complete':
            print(f"\nComplete! Confidence: {data.get('confidence', 'N/A')}")

    def on_error(self, ws, error):
        print(f"WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        print("WebSocket connection closed")

    def on_open(self, ws):
        # Send inference request
        request = {
            "model": "llama-2-7b-chat",
            "input": "Write a short story about a robot learning to paint",
            "stream": True,
            "parameters": {
                "temperature": 0.8,
                "max_tokens": 500
            }
        }
        ws.send(json.dumps(request))

    def run_streaming_inference(self):
        websocket.enableTrace(True)
        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        self.ws.on_open = self.on_open
        self.ws.run_forever()

# Usage
client = StreamingSwarmClient()
client.run_streaming_inference()
```

### Batch Processing

For processing multiple requests efficiently:

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

class BatchSwarmClient(SwarmClient):
    def __init__(self, coordinator_url: str = "http://localhost:8080", max_workers: int = 10):
        super().__init__(coordinator_url)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def run_batch_inference(self, requests):
        """
        Run multiple inference requests in parallel.

        Args:
            requests: List of dictionaries with 'model', 'input', and optional 'parameters'

        Returns:
            List of results in the same order as requests
        """
        futures = []
        for req in requests:
            future = self.executor.submit(
                self.run_inference,
                req['model'],
                req['input'],
                **req.get('parameters', {})
            )
            futures.append(future)

        results = []
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append({"error": str(e)})

        return results

# Usage
batch_client = BatchSwarmClient(max_workers=5)

requests = [
    {"model": "llama-2-7b", "input": "Summarize quantum physics"},
    {"model": "gpt-2-medium", "input": "Write a poem about the ocean"},
    {"model": "llama-2-7b", "input": "Explain blockchain technology"},
    {"model": "gpt-2-medium", "input": "Create a recipe for chocolate cake"},
]

start_time = time.time()
results = batch_client.run_batch_inference(requests)
end_time = time.time()

print(f"Batch processing completed in {end_time - start_time:.2f} seconds")
for i, result in enumerate(results):
    print(f"Request {i+1}: {result.get('output', result.get('error', 'Unknown error'))[:100]}...")
```

### Model Fine-tuning Example

```python
import requests
import json

class SwarmFineTuningClient(SwarmClient):
    def start_fine_tuning(self, model: str, dataset_path: str, parameters: dict = None) -> str:
        """
        Start a fine-tuning job on the swarm cluster.

        Args:
            model: Base model to fine-tune
            dataset_path: Path to training dataset
            parameters: Fine-tuning parameters

        Returns:
            Job ID for tracking progress
        """
        payload = {
            "model": model,
            "dataset": dataset_path,
            "parameters": parameters or {}
        }

        response = self.session.post(
            f"{self.coordinator_url}/api/v1/fine-tune",
            json=payload
        )
        response.raise_for_status()

        return response.json()["job_id"]

    def get_fine_tuning_status(self, job_id: str) -> dict:
        """Get the status of a fine-tuning job."""
        response = self.session.get(f"{self.coordinator_url}/api/v1/fine-tune/{job_id}")
        response.raise_for_status()
        return response.json()

# Usage
ft_client = SwarmFineTuningClient()

# Start fine-tuning
job_id = ft_client.start_fine_tuning(
    model="llama-2-7b",
    dataset_path="/path/to/training/data.jsonl",
    parameters={
        "learning_rate": 2e-5,
        "epochs": 3,
        "batch_size": 4
    }
)

print(f"Fine-tuning job started: {job_id}")

# Monitor progress
while True:
    status = ft_client.get_fine_tuning_status(job_id)
    print(f"Status: {status['status']}, Progress: {status.get('progress', 0)}%")

    if status['status'] in ['completed', 'failed']:
        break

    time.sleep(30)
```

## Docker Compose Setup

For testing the examples locally, use this `docker-compose.yml`:

```yaml
version: '3.8'
services:
  swarm-coordinator:
    image: swarm-inference:latest
    ports:
      - "8080:8080"
    environment:
      - SWARM_COORDINATOR_ADDR=0.0.0.0:8080
    volumes:
      - ./models:/models

  swarm-worker-1:
    image: swarm-inference:latest
    depends_on:
      - swarm-coordinator
    environment:
      - SWARM_COORDINATOR_ADDR=swarm-coordinator:8080
      - SWARM_NODE_ID=worker-1
    volumes:
      - ./models:/models

  swarm-worker-2:
    image: swarm-inference:latest
    depends_on:
      - swarm-coordinator
    environment:
      - SWARM_COORDINATOR_ADDR=swarm-coordinator:8080
      - SWARM_NODE_ID=worker-2
    volumes:
      - ./models:/models
```

## Running the Examples

1. Start the swarm cluster:
   ```bash
   docker-compose up -d
   ```

2. Install dependencies for Python examples:
   ```bash
   pip install requests websocket-client
   ```

3. Run the basic example:
   ```bash
   python examples/basic_client.py
   ```

4. For Node.js examples:
   ```bash
   npm install axios
   node examples/js_client.js
   ```

## API Reference

### REST Endpoints

- `POST /api/v1/inference` - Run inference
- `GET /api/v1/status` - Get cluster status
- `POST /api/v1/fine-tune` - Start fine-tuning job
- `GET /api/v1/fine-tune/{job_id}` - Get fine-tuning status
- `GET /api/v1/models` - List available models
- `GET /api/v1/health` - Health check

### WebSocket Endpoints

- `ws://host:port/ws/inference` - Streaming inference

### Parameters

Common inference parameters:
- `temperature` (float, 0.0-1.0): Controls randomness
- `max_tokens` (int): Maximum output length
- `top_p` (float, 0.0-1.0): Nucleus sampling
- `top_k` (int): Top-k sampling
- `repetition_penalty` (float): Repetition penalty
- `stream` (bool): Enable streaming responses

## Troubleshooting

### Connection Issues

- Ensure the coordinator is running and accessible
- Check firewall settings and port availability
- Verify TLS certificates if using HTTPS

### Performance Issues

- Monitor cluster resource utilization
- Adjust batch sizes for better GPU utilization
- Consider model sharding for large models

### Error Handling

- Implement retry logic with exponential backoff
- Handle rate limiting gracefully
- Log errors for debugging

## Contributing

To add new examples:

1. Create a new file in the appropriate language subdirectory
2. Include clear documentation and comments
3. Add the example to this README
4. Test the example with the local Docker setup