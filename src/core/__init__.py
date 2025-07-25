"""
核心功能模块

提供代理服务的核心功能，包括：
- OpenAI 客户端封装
- 请求/响应格式转换器
- 数据模型定义

子模块:
- clients: OpenAI API 客户端实现
- converters: Anthropic ↔ OpenAI 格式转换器
"""

# 导入核心客户端
from .clients import OpenAIServiceClient

# 导入转换器
from .converters import (
    AnthropicToOpenAIConverter,
    OpenAIToAnthropicConverter,
)

__all__ = [
    # 客户端
    "OpenAIServiceClient",
    # 转换器
    "AnthropicToOpenAIConverter",
    "OpenAIToAnthropicConverter",
]
