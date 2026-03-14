#!/usr/bin/env python3
"""
Basic Swarm Inference Client Example

This example demonstrates how to connect to a Swarm Inference cluster
and run inference requests using the REST API.
"""

import requests
import json
import time
from typing import Dict, Any, Optional

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

    def list_models(self) -> Dict[str, Any]:
        """List available models in the cluster."""
        response = self.session.get(f"{self.coordinator_url}/api/v1/models")
        response.raise_for_status()
        return response.json()

def main():
    # Initialize client
    client = SwarmClient("http://localhost:8080")

    # Check cluster status
    try:
        status = client.get_cluster_status()
        print("✓ Connected to Swarm Inference cluster")
        print(f"  Nodes: {status.get('nodes', 'N/A')}")
        print(f"  Status: {status.get('status', 'N/A')}")
    except requests.exceptions.RequestException as e:
        print(f"✗ Failed to connect to cluster: {e}")
        print("Make sure the swarm cluster is running on http://localhost:8080")
        return

    # List available models
    try:
        models = client.list_models()
        print(f"\nAvailable models: {list(models.get('models', {}).keys())}")
    except requests.exceptions.RequestException:
        print("Could not retrieve model list")

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

    print("\n" + "="*60)
    print("Running inference examples...")
    print("="*60)

    for i, example in enumerate(examples, 1):
        print(f"\n--- Example {i} ---")
        print(f"Model: {example['model']}")
        print(f"Input: {example['input'][:60]}{'...' if len(example['input']) > 60 else ''}")

        try:
            start_time = time.time()
            result = client.run_inference(**example)
            end_time = time.time()

            print(".2f"            print(f"Output: {result.get('output', 'N/A')[:200]}{'...' if len(result.get('output', '')) > 200 else ''}")

            if 'confidence' in result:
                print(f"Confidence: {result['confidence']:.2f}")

        except requests.exceptions.RequestException as e:
            print(f"✗ Inference failed: {e}")
        except KeyError as e:
            print(f"✗ Unexpected response format: {e}")

    print("\n" + "="*60)
    print("Example completed!")
    print("="*60)

if __name__ == "__main__":
    main()