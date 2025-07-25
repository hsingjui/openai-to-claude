"""
OpenAI-Claude Code Proxy

一个高性能的 Anthropic Claude API 到 OpenAI API 格式的代理服务。
提供完整的请求/响应转换、流式处理、错误处理和配置管理功能。

主要功能:
- Anthropic Claude API 到 OpenAI API 格式的双向转换
- 支持流式和非流式响应
- 完整的工具调用支持
- 请求ID追踪和日志记录
- 配置文件热重载
- 性能监控和错误处理

使用示例:
    from src import create_app

    app = create_app()
"""

__version__ = "1.0.0"
__author__ = "OpenAI-Claude Code Proxy Team"
__description__ = "Anthropic Claude API to OpenAI API format proxy service"

# 导出主要的公共API
from .common import configure_logging, request_logger
from .config import get_config, reload_config
from .main import app

__all__ = [
    "app",
    "get_config",
    "reload_config",
    "configure_logging",
    "request_logger",
    "__version__",
    "__author__",
    "__description__",
]
