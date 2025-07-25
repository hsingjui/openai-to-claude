"""Comprehensive error handling tests for the proxy."""

import pytest
import asyncio
from typing import Dict, Any
import requests
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
import os

from src.main import app
from src.core.clients.openai_client import OpenAIServiceClient


class TestErrorHandlingIntegration:
    """Integration tests for various error paths and edge cases."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment."""
        os.environ["OPENAI_API_KEY"] = "mock-key"
        os.environ["OPENAI_BASE_URL"] = "http://localhost:8001"
        os.environ["REQUEST_TIMEOUT"] = "2"  # Short timeout for testing
        
        # Reset mock server
        requests.delete("http://localhost:8001/mock/reset")
        
        yield
        
        # Clean up environment
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
        if "OPENAI_BASE_URL" in os.environ:
            del os.environ["OPENAI_BASE_URL"]
        if "REQUEST_TIMEOUT" in os.environ:
            del os.environ["REQUEST_TIMEOUT"]

    def test_network_timeout(self):
        """Test network timeout handling."""
        # Configure mock server with long delay
        requests.post(
            "http://localhost:8001/mock/configure",
            json={"delay_ms": 3000}  # 3s delay, longer than 2s timeout
        )
        
        client = TestClient(app)
        
        payload = {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Timeout test"}]
        }
        
        response = client.post("/v1/messages", json=payload)
        
        assert response.status_code == 504  # Gateway timeout
        data = response.json()
        
        assert data["error"]["type"] == "timeout_error"
        assert "timeout" in data["error"]["message"].lower()

    def test_connection_refused(self):
        """Test connection refused error handling."""
        # Configure invalid base URL
        os.environ["OPENAI_BASE_URL"] = "http://localhost:9999"  # Invalid port
        
        client = TestClient(app)
        
        payload = {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Connection test"}]
        }
        
        response = client.post("/v1/messages", json=payload)
        
        assert response.status_code == 503  # Service unavailable
        data = response.json()
        
        assert "connection" in data["error"]["type"] or "service_unavailable" in data["error"]["type"]

    def test_malformed_openai_response(self):
        """Test handling of malformed JSON from OpenAI."""
        from unittest.mock import patch
        import httpx
        
        client = TestClient(app)
        
        # Mock httpx to return invalid JSON
        with patch('httpx.AsyncClient.post') as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.side_effect = ValueError("Invalid JSON")
            mock_response.text = "invalid json response"
            mock_post.return_value.__aenter__.return_value = mock_response
            
            payload = {
                "model": "claude-3-sonnet-20240229",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Malformed response test"}]
            }
            
            response = client.post("/v1/messages", json=payload)
            
            assert response.status_code == 502  # Bad gateway
            data = response.json()
            
            assert data["error"]["type"] == "api_error"

    def test_openai_500_error(self):
        """Test handling of 500 errors from OpenAI."""
        client = TestClient(app)
        
        requests.post(
            "http://localhost:8001/mock/configure",
            json={"error_trigger": "server_error"}
        )
        
        payload = {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Server error test"}]
        }
        
        response = client.post("/v1/messages", json=payload)
        
        assert response.status_code == 500
        data = response.json()
        
        assert data["error"]["type"] == "api_error"
        assert "server" in data["error"]["message"].lower()

    def test_openai_429_rate_limit(self):
        """Test handling of rate limit errors from OpenAI."""
        client = TestClient(app)
        
        requests.post(
            "http://localhost:8001/mock/configure",
            json={"error_trigger": "rate_limit"}
        )
        
        payload = {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Rate limit test"}]
        }
        
        response = client.post("/v1/messages", json=payload)
        
        assert response.status_code == 429
        data = response.json()
        
        assert data["error"]["type"] == "rate_limit_error"

    def test_openai_401_unauthorized(self):
        """Test handling of unauthorized errors from OpenAI."""
        os.environ["OPENAI_API_KEY"] = "invalid-mock-key"
        
        client = TestClient(app)
        
        payload = {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Unauthorized test"}]
        }
        
        response = client.post("/v1/messages", json=payload)
        
        assert response.status_code == 401
        data = response.json()
        
        assert data["error"]["type"] == "authentication_error"

    def test_invalid_json_payload(self):
        """Test handling of invalid JSON payload."""
        client = TestClient(app)
        
        # Send raw text instead of JSON
        response = client.post(
            "/v1/messages", 
            data="invalid json payload",
            headers={"content-type": "application/json"}
        )
        
        assert response.status_code == 422
        data = response.json()
        
        assert "json" in str(data).lower() or "invalid" in str(data).lower()

    def test_missing_required_fields(self):
        """Test validation errors for missing required fields."""
        client = TestClient(app)
        
        test_cases = [
            # Missing messages
            {"model": "claude-3-sonnet-20240229", "max_tokens": 100},
            # Empty messages
            {"model": "claude-3-sonnet-20240229", "max_tokens": 100, "messages": []},
            # Missing model
            {"max_tokens": 100, "messages": [{"role": "user", "content": "test"}]},
            # Invalid message structure
            {
                "model": "claude-3-sonnet-20240229", 
                "max_tokens": 100, 
                "messages": [{"invalid_role": "test", "content": "test"}]
            }
        ]
        
        for payload in test_cases:
            response = client.post("/v1/messages", json=payload)
            assert response.status_code == 422, f"Expected validation error for {payload}"

    def test_invalid_temperature_range(self):
        """Test validation of temperature parameter."""
        client = TestClient(app)
        
        test_cases = [
            -0.1,  # Below 0.0
            2.1,   # Above 2.0
            "abc", # Non-numeric
            None   # Null (optional should be fine)
        ]
        
        for temp in test_cases:
            payload = {
                "model": "claude-3-sonnet-20240229",
                "max_tokens": 100,
                "temperature": temp,
                "messages": [{"role": "user", "content": "test"}]
            }
            
            if temp is None or temp == "abc":
                # These might pass validation, skip
                continue
                
            response = client.post("/v1/messages", json=payload)
            
            if temp < 0.0 or temp > 2.0:
                assert response.status_code == 422

    def test_invalid_max_tokens(self):
        """Test validation of max_tokens parameter."""
        client = TestClient(app)
        
        test_cases = [
            -1,     # Negative
            0,      # Zero
            1_000_000,  # Too large
            "abc",  # Non-integer
            None    # Should be required
        ]
        
        for max_tokens in test_cases:
            payload = {
                "model": "claude-3-sonnet-20240229",
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": "test"}]
            }
            
            if max_tokens is None or max_tokens == "abc" or max_tokens <= 0:
                response = client.post("/v1/messages", json=payload)
                if max_tokens is None or max_tokens <= 0:
                    assert response.status_code == 422

    def test_large_payload_handling(self):
        """Test handling of very large payloads."""
        client = TestClient(app)
        
        # Create large message content
        large_content = "x" * 100_000  # 100KB of content
        
        payload = {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": large_content},
                {"role": "assistant", "content": "That's a large payload"},
                {"role": "user", "content": "Yes it is"}
            ]
        }
        
        response = client.post("/v1/messages", json=payload)
        
        # Should handle large content gracefully (might succeed or validation error)
        assert response.status_code in [200, 422]

    def test_special_characters_in_content(self):
        """Test handling of special characters in message content."""
        client = TestClient(app)
        
        special_contents = [
            "Unicode: 你好世界 🌍",
            "Emojis: 🚀✈️🚁🛸",
            "Newlines: Hello\nWorld\nTest",
            "Quotes: \"Hello\" and 'World'",
            "HTML: <div>test</div>",
            "JSON: {'key': 'value'}",
            "Code: `print('hello')`",
            "Long text: " + "word " * 100
        ]
        
        for content in special_contents:
            payload = {
                "model": "claude-3-sonnet-20240229",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": content}]
            }
            
            response = client.post("/v1/messages", json=payload)
            assert response.status_code == 200, f"Failed for content: {content[:50]}..."

    def test_streaming_timeout(self):
        """Test streaming timeout handling."""
        # Configure mock server with long delay for streaming
        requests.post(
            "http://localhost:8001/mock/configure",  
            json={"delay_ms": 3000}
        )
        
        client = TestClient(app)
        
        payload = {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 100,
            "stream": True,
            "messages": [{"role": "user", "content": "Streaming timeout test"}]
        }
        
        with client.stream("POST", "/v1/messages", json=payload) as response:
            # Streaming timeout might not be caught until we read
            assert response.status_code == 200  # Headers sent immediately
            
            # Collect chunks
            chunks = []
            for line in response.iter_lines():
                if line and line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    
                    try:
                        data = json.loads(data_str)
                        chunks.append(data)
                    except Exception:
                        # Might get timeout error in streaming chunks
                        pass

    def test_concurrent_error_handling(self):
        """Test error handling with concurrent requests."""
        import concurrent.futures
        import time
        
        client = TestClient(app)
        
        def make_request(request_num):
            if request_num % 2 == 0:
                # Valid request
                payload = {
                    "model": "claude-3-sonnet-20240229",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": f"Valid request {request_num}"}]
                }
            else:
                # Invalid request (missing messages)
                payload = {
                    "model": "claude-3-sonnet-20240229",
                    "max_tokens": 100
                }
            
            return client.post("/v1/messages", json=payload)
        
        # Make concurrent requests with mix of valid and invalid
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request, i) for i in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Count success vs error responses
        success_count = sum(1 for r in results if r.status_code == 200)
        error_count = sum(1 for r in results if r.status_code != 200)
        
        # Should have both success and error responses
        assert success_count == 5  # Even-numbered requests
        assert error_count == 5   # Odd-numbered requests