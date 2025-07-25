"""
集成测试：验证模型映射的完整功能
"""
import json
from src.models.anthropic import AnthropicRequest, AnthropicMessage
from src.models.openai import OpenAIRequest
from src.core.converters.request_converter import AnthropicToOpenAIConverter


class TestModelMappingIntegration:
    """集成测试：验证从Anthropic到OpenAI的完整转换"""

    def test_convert_with_thinking_model(self):
        """使用thinking模型的完整转换测试"""
        anthropic_request = AnthropicRequest(
            model="claude-3-5-sonnet-20241022",
            messages=[AnthropicMessage(role="user", content="hello")],
            max_tokens=100,
            thinking=True
        )
        
        openai_request = AnthropicToOpenAIConverter.convert_anthropic_to_openai(anthropic_request)
        
        assert isinstance(openai_request, OpenAIRequest)
        assert openai_request.model == "claude-3-7-sonnet-thinking"
        assert len(openai_request.messages) == 1
        assert openai_request.messages[0].content == "hello"
        assert openai_request.max_tokens == 100

    def test_convert_with_sonnet_model(self):
        """使用sonnet模型的完整转换测试"""
        anthropic_request = AnthropicRequest(
            model="claude-3-5-sonnet-20241022",
            messages=[AnthropicMessage(role="user", content="hello")],
            max_tokens=100,
            thinking=None
        )
        
        openai_request = AnthropicToOpenAIConverter.convert_anthropic_to_openai(anthropic_request)
        
        assert isinstance(openai_request, OpenAIRequest)
        assert openai_request.model == "claude-3-5-sonnet"

    def test_convert_with_haiku_model(self):
        """使用haiku模型的完整转换测试"""
        anthropic_request = AnthropicRequest(
            model="claude-3-5-haiku",
            messages=[AnthropicMessage(role="user", content="hello")],
            max_tokens=100,
            thinking=None
        )
        
        openai_request = AnthropicToOpenAIConverter.convert_anthropic_to_openai(anthropic_request)
        
        assert isinstance(openai_request, OpenAIRequest)
        assert openai_request.model == "claude-3-5-haiku"

    def test_convert_with_system_support(self):
        """带系统提示的完整转换测试"""
        anthropic_request = AnthropicRequest(
            model="claude-3-5-sonnet",
            messages=[AnthropicMessage(role="user", content="hello")],
            max_tokens=100,
            system="你是一个有用的助手",
            thinking=True
        )
        
        openai_request = AnthropicToOpenAIConverter.convert_anthropic_to_openai(anthropic_request)
        
        assert openai_request.model == "claude-3-7-sonnet-thinking"
        assert openai_request.messages[0].role == "system"
        assert openai_request.messages[0].content == "你是一个有用的助手"

    def test_convert_with_tools(self):
        """带工具定义的完整转换测试"""
        tools = [
            {
                "name": "get_weather",
                "description": "获取天气信息",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["location"]
                }
            }
        ]
        
        anthropic_request = AnthropicRequest(
            model="claude-3-5-haiku",
            messages=[AnthropicMessage(role="user", content="天气如何")],
            max_tokens=100,
            tools=tools,
            thinking=None
        )
        
        openai_request = AnthropicToOpenAIConverter.convert_anthropic_to_openai(anthropic_request)
        
        assert openai_request.model == "claude-3-5-haiku"
        assert openai_request.tools is not None
        assert len(openai_request.tools) == 1
        assert openai_request.tools[0].function.name == "get_weather"

    def test_model_passthrough(self):
        """测试带有逗号的模型名不被转换"""
        anthropic_request = AnthropicRequest(
            model="claude-custom,variant",
            messages=[AnthropicMessage(role="user", content="hello")],
            max_tokens=100
        )
        
        # thinking字段存在，但因为模型名有逗号，应该保留原样
        openai_request = AnthropicToOpenAIConverter.convert_anthropic_to_openai(anthropic_request)
        
        assert openai_request.model == "claude-custom,variant"

    def test_serialization_roundtrip(self):
        """测试序列化和反序列化"""
        original_request = {
            "model": "claude-3-5-sonnet",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 100,
            "thinking": True
        }
        
        anthropic_request = AnthropicRequest(**original_request)
        openai_request = AnthropicToOpenAIConverter.convert_anthropic_to_openai(anthropic_request)
        
        # 确保可以序列化为JSON
        openai_dict = openai_request.dict()
        assert openai_dict["model"] == "claude-3-7-sonnet-thinking"
        assert openai_dict["messages"][0]["role"] == "user"
        assert openai_dict["messages"][0]["content"] == "hello"