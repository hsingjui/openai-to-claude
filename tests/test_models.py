"""测试数据模型是否正确工作"""

import pytest
from typing import List, Dict, Any
from src.models.anthropic import (
    AnthropicRequest, AnthropicMessage, AnthropicMessageResponse,
    AnthropicToolDefinition, AnthropicSystemMessage
)
from src.models.openai import (
    OpenAIRequest, OpenAIMessage, OpenAIResponse, OpenAITool
)
from src.models.errors import (
    get_error_response, is_client_error, is_server_error,
    ValidationErrorItem, ValidationErrorResponse
)


class TestAnthropicModels:
    """测试Anthropic数据模型"""

    def test_anthropic_message_creation(self):
        """测试基础消息创建"""
        message = AnthropicMessage(
            role="user",
            content="Hello, world!"
        )
        assert message.role == "user"
        assert message.content == "Hello, world!"

    def test_anthropic_message_with_content_blocks(self):
        """测试带内容块的消息"""
        message = AnthropicMessage(
            role="user",
            content=[
                {
                    "type": "text",
                    "text": "Hello"
                }
            ]
        )
        assert len(message.content) == 1
        assert message.content[0].type == "text"
        assert message.content[0].text == "Hello"

    def test_anthropic_request_creation(self):
        """测试Anthropic请求创建"""
        request = AnthropicRequest(
            model="claude-3-5-sonnet-20241022",
            messages=[
                AnthropicMessage(role="user", content="Hello")
            ],
            max_tokens=1000,
            temperature=0.7
        )
        assert request.model == "claude-3-5-sonnet-20241022"
        assert len(request.messages) == 1
        assert request.max_tokens == 1000
        assert request.temperature == 0.7

    def test_anthropic_request_with_tools(self):
        """测试带工具的Anthropic请求"""
        tool = AnthropicToolDefinition(
            name="get_weather",
            description="Get weather information",
            input_schema={
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        )
        
        request = AnthropicRequest(
            model="claude-3-5-sonnet-20241022",
            messages=[AnthropicMessage(role="user", content="What's the weather?")],
            max_tokens=1000,
            tools=[tool]
        )
        
        assert len(request.tools) == 1
        assert request.tools[0].name == "get_weather"

    def test_anthropic_request_with_system(self):
        """测试带系统消息的Anthropic请求"""
        request = AnthropicRequest(
            model="claude-3-5-sonnet-20241022",
            messages=[AnthropicMessage(role="user", content="Hello")],
            max_tokens=1000,
            system="You are a helpful assistant."
        )
        
        assert request.system == "You are a helpful assistant."

    def test_anthropic_system_message_list(self):
        """测试系统消息列表"""
        system_messages = [
            AnthropicSystemMessage(type="text", text="You are helpful."),
            AnthropicSystemMessage(type="text", text="Be concise.")
        ]
        
        request = AnthropicRequest(
            model="claude-3-5-sonnet-20241022",
            messages=[AnthropicMessage(role="user", content="Hi")],
            max_tokens=100,
            system=system_messages
        )
        
        assert len(request.system) == 2
        assert request.system[0].text == "You are helpful."


class TestOpenAIModels:
    """测试OpenAI数据模型"""

    def test_openai_message_creation(self):
        """测试基础消息创建"""
        message = OpenAIMessage(
            role="user",
            content="Hello, world!"
        )
        assert message.role == "user"
        assert message.content == "Hello, world!"

    def test_openai_message_with_content_parts(self):
        """测试带内容部分的消息"""
        message = OpenAIMessage(
            role="user",
            content=[
                {"type": "text", "text": "Hello"},
                {"type": "image_url", "image_url": {"url": "http://example.com/image.jpg"}}
            ]
        )
        assert len(message.content) == 2
        assert message.content[0].type == "text"
        assert message.content[1].type == "image_url"

    def test_openai_request_creation(self):
        """测试OpenAI请求创建"""
        request = OpenAIRequest(
            model="gpt-4o",
            messages=[
                OpenAIMessage(role="user", content="Hello")
            ],
            max_tokens=1000,
            temperature=0.7
        )
        assert request.model == "gpt-4o"
        assert len(request.messages) == 1
        assert request.max_tokens == 1000
        assert request.temperature == 0.7

    def test_openai_request_with_tools(self):
        """测试带工具的OpenAI请求"""
        from src.models.openai import OpenAIToolFunction
        
        tool_function = OpenAIToolFunction(
            name="get_weather",
            description="Get weather information",
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        )
        
        tool = OpenAITool(
            type="function",
            function=tool_function
        )
        
        request = OpenAIRequest(
            model="gpt-4o",
            messages=[OpenAIMessage(role="user", content="What's the weather?")],
            tools=[tool]
        )
        
        assert len(request.tools) == 1
        assert request.tools[0].function.name == "get_weather"

    def test_openai_request_system_message(self):
        """测试带系统消息的OpenAI请求"""
        request = OpenAIRequest(
            model="gpt-4o",
            messages=[
                OpenAIMessage(role="system", content="You are a helpful assistant."),
                OpenAIMessage(role="user", content="Hello")
            ]
        )
        
        assert len(request.messages) == 2
        assert request.messages[0].role == "system"
        assert request.messages[1].role == "user"

    def test_openai_request_validation(self):
        """测试OpenAI请求验证"""
        # 测试无效温度
        with pytest.raises(Exception):  # Pydantic会抛出ValidationError
            OpenAIRequest(
                model="gpt-4o",
                messages=[OpenAIMessage(role="user", content="Hello")],
                temperature=5.0  # 超出范围
            )
        
        # 测试无效top_p
        with pytest.raises(Exception):
            OpenAIRequest(
                model="gpt-4o",
                messages=[OpenAIMessage(role="user", content="Hello")],
                top_p=2.0  # 超出范围
            )


class TestErrorModels:
    """测试错误响应模型"""

    def test_get_error_response_client_errors(self):
        """测试客户端错误响应"""
        # 400 Bad Request
        response = get_error_response(400, "Invalid request format")
        assert response.error.code == "bad_request"
        assert response.error.message == "Invalid request format"

        # 401 Unauthorized
        response = get_error_response(401)
        assert response.error.code == "unauthorized"

        # 404 Not Found
        response = get_error_response(404)
        assert response.error.code == "not_found"

    def test_get_error_response_server_errors(self):
        """测试服务器错误响应"""
        # 500 Server Error
        response = get_error_response(500, "Internal processing error")
        assert response.error.code == "internal_server_error"

        # 503 Service Unavailable
        response = get_error_response(503, details={"retry_after": 60})
        assert response.error.code == "service_unavailable"

    def test_is_client_error(self):
        """测试客户端错误判断"""
        assert is_client_error(400) == True
        assert is_client_error(404) == True
        assert is_client_error(429) == True
        assert is_client_error(500) == False
        assert is_client_error(200) == False

    def test_is_server_error(self):
        """测试服务器错误判断"""
        assert is_server_error(500) == True
        assert is_server_error(503) == True
        assert is_server_error(400) == False
        assert is_server_error(429) == False

    def test_validation_error_items(self):
        """测试验证错误项"""
        error_item = ValidationErrorItem(
            loc=["body", "messages", "0", "content"],
            msg="Field required",
            type="value_error.missing"
        )
        
        assert error_item.loc == ["body", "messages", "0", "content"]
        assert error_item.msg == "Field required"
        assert error_item.type == "value_error.missing"