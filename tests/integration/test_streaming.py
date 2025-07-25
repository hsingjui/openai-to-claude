"""Streaming response tests for end-to-end testing."""

import pytest
import requests
from fastapi.testclient import TestClient
import os

from src.main import app


class TestStreamingIntegration:
    """Tests for streaming response handling in the proxy."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment."""
        os.environ["OPENAI_API_KEY"] = "mock-key"
        os.environ["OPENAI_BASE_URL"] = "http://localhost:8001"
        
        # Reset mock server
        requests.delete("http://localhost:8001/mock/reset")
        
        yield
        
        # Clean up environment
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
        if "OPENAI_BASE_URL" in os.environ:
            del os.environ["OPENAI_BASE_URL"]

    def test_streaming_chat_completion(self):
        """Test complete streaming flow from Anthropic to OpenAI proxy."""
        client = TestClient(app)
        
        payload = {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 1024,
            "stream": True,
            "messages": [
                {"role": "user", "content": "Write a short poem about Python"}
            ]
        }
        
        with client.stream("POST", "/v1/messages", json=payload) as response:
            assert response.status_code == 200
            assert response.headers["content-type"] == "text/plain; charset=utf-8"
            
            chunks = []
            for line in response.iter_lines():
                if line and line.startswith("data: "):
                    data_str = line[6:]  # Remove "data: " prefix
                    if data_str.strip() == "[DONE]":
                        break
                    
                    try:
                        data = json.loads(data_str)
                        chunks.append(data)
                    except json.JSONDecodeError:
                        continue
            
            # Verify we received some chunks
            assert len(chunks) > 0
            
            # Check structure of first chunk
            first_chunk = chunks[0]
            assert "type" in first_chunk
            assert first_chunk["type"] == "content_block_start"
            assert "content_block" in first_chunk
            
            # Check structure of content chunks
            content_chunks = [c for c in chunks if c.get("type") == "content_block_delta"]
            assert len(content_chunks) > 0
            
            # Check final chunk
            final_chunks = [c for c in chunks if c.get("type") == "message_stop"]
            assert len(final_chunks) == 1

    def test_streaming_with_system_message(self):
        """Test streaming with system message included."""
        client = TestClient(app)
        
        payload = {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 512,
            "stream": True,
            "temperature": 0.8,
            "messages": [
                {"role": "system", "content": "You are a Python programming assistant."},
                {"role": "user", "content": "Explain decorators"}
            ]
        }
        
        with client.stream("POST", "/v1/messages", json=payload) as response:
            assert response.status_code == 200
            
            chunks = []
            for line in response.iter_lines():
                if line and line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    
                    try:
                        data = json.loads(data_str)
                        chunks.append(data)
                    except json.JSONDecodeError:
                        continue
            
            # Should still receive chunks even with system message
            assert len(chunks) > 2
            
            # Verify we got the message start
            start_chunks = [c for c in chunks if c.get("type") == "message_start"]
            assert len(start_chunks) == 1
            
            # Verify we got content blocks
            content_start = [c for c in chunks if c.get("type") == "content_block_start"]
            assert len(content_start) == 1

    def test_streaming_empty_content_handling(self):
        """Test streaming with empty or minimal content."""
        client = TestClient(app)
        
        payload = {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 10,
            "stream": True,
            "messages": [
                {"role": "user", "content": ""}
            ]
        }
        
        with client.stream("POST", "/v1/messages", json=payload) as response:
            assert response.status_code == 200
            
            chunks = []
            for line in response.iter_lines():
                if line and line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    
                    try:
                        data = json.loads(data_str)
                        chunks.append(data)
                    except json.JSONDecodeError:
                        continue
            
            # Should still receive proper streaming structure
            assert len(chunks) >= 3  # message_start, content_block_start, content_block_delta, message_stop

    def test_streaming_usage_information(self):
        """Test that usage information is included in streaming response."""
        client = TestClient(app)
        
        payload = {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 100,
            "stream": True,
            "messages": [
                {"role": "user", "content": "Count to 5"}
            ]
        }
        
        with client.stream("POST", "/v1/messages", json=payload) as response:
            assert response.status_code == 200
            
            # Collect all chunks to ensure complete flow
            chunks = []
            for line in response.iter_lines():
                if line and line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    
                    try:
                        data = json.loads(data_str)
                        chunks.append(data)
                    except json.JSONDecodeError:
                        continue
            
            # Look for message_delta with usage info
            usage_chunks = [c for c in chunks if c.get("type") == "message_delta"]
            assert len(usage_chunks) >= 1
            
            final_chunk = usage_chunks[0]
            assert "usage" in final_chunk
            usage = final_chunk["usage"]
            assert "input_tokens" in usage
            assert "output_tokens" in usage

    def test_streaming_response_timing(self):
        """Test streaming response timing performance."""
        import time
        
        client = TestClient(app)
        
        payload = {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 50,
            "stream": True,
            "messages": [
                {"role": "user", "content": "Say 'hello' three times"}
            ]
        }
        
        start_time = time.time()
        
        with client.stream("POST", "/v1/messages", json=payload) as response:
            assert response.status_code == 200
            
            chunks = []
            for line in response.iter_lines():
                if line and line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    
                    try:
                        data = json.loads(data_str)
                        chunks.append(data)
                    except json.JSONDecodeError:
                        continue
        
        end_time = time.time()
        duration = (end_time - start_time) * 1000
        
        # Ensure reasonable performance for streaming
        assert duration < 1000  # Should complete within 1 second for test
        assert len(chunks) > 3  # Should have multiple streaming chunks

    def test_streaming_with_stream_false_comparison(self):
        """Test that streaming works differently from non-streaming."""
        client = TestClient(app)
        
        request_data = {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 50,
            "messages": [
                {"role": "user", "content": "Reply with a single word: Python"}
            ]
        }
        
        # Non-streaming version
        non_stream_payload = request_data.copy()
        non_stream_payload["stream"] = False
        sync_response = client.post("/v1/messages", json=non_stream_payload)
        assert sync_response.status_code == 200
        
        sync_data = sync_response.json()
        assert "content" in sync_data
        
        # Streaming version
        stream_payload = request_data.copy()
        stream_payload["stream"] = True
        
        with client.stream("POST", "/v1/messages", json=stream_payload) as response:
            assert response.status_code == 200
            
            # Should be server-sent events format
            assert response.headers["content-type"] == "text/plain; charset=utf-8"
            
            # Collect streaming content
            collected_content = ""
            chunks = []
            
            for line in response.iter_lines():
                if line and line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    
                    try:
                        data = json.loads(data_str)
                        chunks.append(data)
                        
                        # Extract text content from delta chunks
                        if data.get("type") == "content_block_delta":
                            delta = data.get("delta", {})
                            if "text" in delta:
                                collected_content += delta["text"]
                    except json.JSONDecodeError:
                        continue
            
            # Both should return similar content, streaming breaks it into chunks
            assert len(collected_content) > 0
            assert "Python" in collected_content or "python" in collected_content.lower()