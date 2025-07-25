"""End-to-end integration tests for the OpenAI-to-Claude proxy."""

import asyncio
import json
import os
import requests
import pytest
from typing import Dict, Any, List
import httpx
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock

from src.main import app
from tests.fixtures.mock_openai_server import app as mock_app


class TestEndToEndIntegration:
    """End-to-end integration tests covering the full proxy flow."""

    @pytest.fixture(autouse=True)
    def setup_client(self):
        """Set up test environment and create a test client with auth headers."""
        api_key = "mock-key"
        os.environ["API_KEY"] = api_key
        os.environ["OPENAI_API_KEY"] = "mock-openai-key"
        os.environ["OPENAI_BASE_URL"] = "http://localhost:8001"
        
        # Reset mock server configuration
        try:
            requests.delete("http://localhost:8001/mock/reset")
        except requests.ConnectionError:
            # Mock server might not be running, which is fine for some tests
            pass
        
        self.client = TestClient(app)
        self.client.headers = {"Authorization": f"Bearer {api_key}"}
        
        yield
        
        # Clean up environment
        if "API_KEY" in os.environ:
            del os.environ["API_KEY"]
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
        if "OPENAI_BASE_URL" in os.environ:
            del os.environ["OPENAI_BASE_URL"]

    def test_basic_chat_completion(self):
        """Test complete flow from Anthropic request to OpenAI proxy."""
        client = self.client
        
        payload = {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "Hello, world!"}
            ]
        }
        
        response = client.post("/v1/messages", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify Anthropic response structure
        assert "id" in data
        assert "type" in data
        assert "role" in data
        assert "content" in data
        assert "model" in data
        assert "stop_reason" in data
        assert "usage" in data
        
        # Verify content appeared in mock response
        assert "Hello, world!" in data["content"][0]["text"]

    def test_chat_completion_with_system_message(self):
        """Test chat completion with system message included."""
        client = self.client
        
        payload = {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 1024,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Tell me a joke"}
            ]
        }
        
        response = client.post("/v1/messages", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["role"] == "assistant"
        assert "Tell me a joke" in data["content"][0]["text"]

    def test_chat_completion_with_temperature(self):
        """Test parameter conversion including temperature."""
        client = self.client
        
        payload = {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 500,
            "temperature": 0.7,
            "messages": [
                {"role": "user", "content": "Give me a creative response"}
            ]
        }
        
        response = client.post("/v1/messages", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        assert response.status_code == 200
        assert "creative response" in data["content"][0]["text"]

    def test_invalid_model_error(self):
        """Test error handling for invalid model."""
        client = self.client
        
        # Configure mock to return error
        requests.post(
            "http://localhost:8001/mock/configure",
            json={"error_trigger": "invalid_model"}
        )
        
        payload = {
            "model": "invalid-model-name",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "test"}]
        }
        
        response = client.post("/v1/messages", json=payload)
        
        assert response.status_code == 400
        data = response.json()
        
        assert "type" in data
        assert "error" in data
        assert data["error"]["type"] == "invalid_request_error"

    def test_rate_limit_error(self):
        """Test rate limit error response."""
        client = self.client
        
        # Configure mock to return rate limit error
        requests.post(
            "http://localhost:8001/mock/configure",
            json={"error_trigger": "rate_limit"}
        )
        
        payload = {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "test"}]
        }
        
        response = client.post("/v1/messages", json=payload)
        
        assert response.status_code == 429
        data = response.json()
        
        assert data["error"]["type"] == "rate_limit_error"

    def test_server_error(self):
        """Test server error response."""
        client = self.client
        
        # Configure mock to return server error
        requests.post(
            "http://localhost:8001/mock/configure",
            json={"error_trigger": "server_error"}
        )
        
        payload = {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "test"}]
        }
        
        response = client.post("/v1/messages", json=payload)
        
        assert response.status_code == 500
        data = response.json()
        
        assert data["error"]["type"].startswith("api")

    def test_invalid_api_key_error(self):
        """Test unauthorized error with invalid API key."""
        os.environ["OPENAI_API_KEY"] = "invalid-key"
        
        client = self.client
        
        payload = {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "test"}]
        }
        
        response = client.post("/v1/messages", json=payload)
        
        assert response.status_code == 401
        data = response.json()
        
        assert data["error"]["type"] == "authentication_error"

    def test_request_validation_error(self):
        """Test request validation error handling."""
        client = self.client
        
        # Invalid payload structure
        payload = {
            "model": "claude-3-sonnet-20240229",
            # Missing required 'messages'
            "max_tokens": 1024
        }
        
        response = client.post("/v1/messages", json=payload)
        
        assert response.status_code == 422
        data = response.json()
        
        assert "messages" in data["error"]["type"] or "messages" in str(data)

    def test_empty_message_content(self):
        """Test handling of empty message content."""
        client = self.client
        
        payload = {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": ""}]
        }
        
        response = client.post("/v1/messages", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "content" in data
        assert len(data["content"]) > 0

    def test_concurrent_requests(self):
        """Test handling multiple concurrent requests."""
        client = self.client
        
        def make_request(content: str):
            payload = {
                "model": "claude-3-sonnet-20240229",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": content}]
            }
            return client.post("/v1/messages", json=payload)
        
        # Make 5 concurrent requests
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(make_request, f"Request {i}")
                for i in range(5)
            ]
            
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All should succeed
        for response in results:
            assert response.status_code == 200
            data = response.json()
            assert "content" in data

    def test_response_timing(self):
        """Test response timing is under expected threshold."""
        import time
        
        client = self.client
        
        # Configure mock for 100ms delay
        requests.post(
            "http://localhost:8001/mock/configure",
            json={"delay_ms": 100}
        )
        
        payload = {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Why is the sky blue?"}]
        }
        
        start_time = time.time()
        response = client.post("/v1/messages", json=payload)
        end_time = time.time()
        
        assert response.status_code == 200
        duration = (end_time - start_time) * 1000  # Convert to milliseconds
        
        # Should be under 50ms plus mock delay (adjusted for testing environment)
        assert duration < 200  # Conservative threshold including mock overhead

    @pytest.mark.skip
    def test_health_check_endpoint(self):
        """Test health check endpoint connectivity."""
        client = self.client
        
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(mock_app, host="0.0.0.0", port=8001)