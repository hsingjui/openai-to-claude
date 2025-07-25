"""
_integration tests for /v1/messages endpoint

ٛKՌ�t*�B-͔A�pnlb�OpenAI API�
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock
import json

from src.main import app
from src.models.anthropic import AnthropicRequest
from src.models.openai import OpenAIRequest, OpenAIResponse


class TestMessagesEndpoint:
    """K� /v1/messages ﹟�"""

    @pytest.fixture
    def client(self):
        """�Kբ7�"""
        return TestClient(app)

    @pytest.fixture
    def valid_anthropic_request(self):
        """	H�Anthropic�B:�"""
        return {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, how are you?"
                }
            ]
        }

    @pytest.fixture
    def mock_openai_response(self):
        """Mock�OpenAI͔"""
        return {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-3.5-turbo",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "I'm doing well! How can I help you today?"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 15,
                "total_tokens": 25
            }
        }

    def test_endpoint_exists(self, client):
        """���X("""
        response = client.post("/v1/messages", json={"model": "test"})
        assert response.status_code != 404

    def test_invalid_request_format(self, client):
        """K��H�B<"""
        response = client.post("/v1/messages", json={"invalid": "format"})
        assert response.status_code == 422

    def test_missing_required_fields(self, client):
        """K�:�kW�"""
        response = client.post("/v1/messages", json={"model": "test"})
        assert response.status_code == 422

    def test_empty_messages(self, client):
        """K�z�oh"""
        request = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [],
            "max_tokens": 100
        }
        response = client.post("/v1/messages", json=request)
        assert response.status_code == 422

    @patch('src.handlers.messages.MessagesHandler.process_message')
    @pytest.mark.asyncio
    async def test_successful_non_streaming_request(
        self, mock_process, client, valid_anthropic_request, mock_openai_response
    ):
        """K���^A�B"""
        from src.models.anthropic import AnthropicMessageResponse
        
        # ���Anthropic͔
        expected_response = AnthropicMessageResponse(
            id="msg_123",
            type="message",
            role="assistant",
            content=[{"type": "text", "text": "I'm doing well! How can I help you today?"}],
            model="claude-3-5-sonnet-20241022",
            stop_reason="end_turn",
            usage={"input_tokens": 10, "output_tokens": 15}
        )
        
        mock_process.return_value = expected_response
        
        response = client.post("/v1/messages", json=valid_anthropic_request)
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert response_data["type"] == "message"
        assert response_data["role"] == "assistant"
        assert len(response_data["content"]) > 0
        assert response_data["content"][0]["type"] == "text"
        assert response_data["model"] == "claude-3-5-sonnet-20241022"
        assert "usage" in response_data

    @patch('src.handlers.messages.MessagesHandler.process_stream_message')
    @pytest.mark.asyncio
    async def test_successful_streaming_request(
        self, mock_stream, client, valid_anthropic_request, mock_openai_response
    ):
        """K���A�B"""
        valid_anthropic_request["stream"] = True
        
        # !�A͔
        async def mock_generator():
            yield 'event: content_block_start\ndata: {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}\n\n'
            yield 'event: content_block_delta\ndata: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}\n\n'
            yield 'event: message_stop\ndata: {"type": "message_stop"}\n\n'
        
        mock_stream.return_value = mock_generator()
        
        response = client.post("/v1/messages", json=valid_anthropic_request)
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
        
        lines = response.content.decode('utf-8').split('\\n')
        assert any('content_block_start' in line for line in lines)

    def test_system_message_support(self, client):
        """K��߈o/"""
        request = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello"}],
            "system": [{
                "type": "text",
                "text": "You are a helpful assistant"
            }],
            "max_tokens": 100
        }
        
        response = client.post("/v1/messages", json=request)
        assert response.status_code != 400

    def test_tool_definition_support(self, client):
        """K��w�I/"""
        request = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello"}],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get weather information",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        },
                        "required": ["location"]
                    }
                }
            ],
            "max_tokens": 100
        }
        
        response = client.post("/v1/messages", json=request)
        assert response.status_code != 400

    def test_temperature_parameter(self, client):
        """K�temperature�p"""
        request = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.5,
            "max_tokens": 100
        }
        
        response = client.post("/v1/messages", json=request)
        # ��ǌ��E��1converter
        assert response.status_code != 422

    def test_max_tokens_parameter(self, client):
        """K�max_tokens�p"""
        request = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 1000
        }
        
        response = client.post("/v1/messages", json=request)
        assert response.status_code != 422

    def test_top_p_parameter(self, client):
        """K�top_p�p"""
        request = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello"}],
            "top_p": 0.9,
            "max_tokens": 100
        }
        
        response = client.post("/v1/messages", json=request)
        assert response.status_code != 422

    def test_stop_sequences_parameter(self, client):
        """K�stop_sequences�p"""
        request = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello"}],
            "stop_sequences": ["\n\n"],
            "max_tokens": 100
        }
        
        response = client.post("/v1/messages", json=request)
        assert response.status_code != 422

    def test_metadata_parameter(self, client):
        """K�metadata�p"""
        request = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello"}],
            "metadata": {"trace_id": "123456"},
            "max_tokens": 100
        }
        
        response = client.post("/v1/messages", json=request)
        assert response.status_code != 422

    def test_image_content_support(self, client):
        """K��υ�/"""
        request = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": "base64_data_here"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 100
        }
        
        response = client.post("/v1/messages", json=request)
        assert response.status_code != 422

    @patch.dict('src.models.config.Config', {'openai': {'api_key': 'test-key'}})
    def test_environment_variables(self, client):
        """Kկ���Mn"""
        # ��Mn���
        response = client.post("/v1/messages", json={
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100
        })
        
        # Config import
        assert response.status_code != 500

    def test_error_response_format(self, client):
        """K��͔<&Anthropic�"""
        response = client.post("/v1/messages", json={"invalid": "format"})
        
        assert response.status_code == 422
        error_data = response.json()
        
        # ���<+�W�
        assert "type" in error_data
        assert "error" in error_data
        assert "type" in error_data["error"]
        assert "message" in error_data["error"]

    def test_all_required_fields_present(self, client):
        """K��kW��t'"""
        required_fields = [
            "model",
            "messages", 
            "max_tokens"
        ]
        
        for field in required_fields:
            base_request = {
                "model": "claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 100
            }
            
            # �d�kW�
            del base_request[field]
            
            response = client.post("/v1/messages", json=base_request)
            assert response.status_code == 422