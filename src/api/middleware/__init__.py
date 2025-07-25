"""
中间件模块

提供FastAPI应用的各种中间件实现。

主要功能:
- API密钥认证中间件
- 请求计时中间件
- 请求ID追踪
- 中间件配置和设置

使用示例:
    from src.api.middleware import setup_middlewares

    setup_middlewares(app)
"""

# 导入中间件实现
from .auth import APIKeyMiddleware
from .timing import RequestTimingMiddleware, setup_middlewares

__all__ = [
    "APIKeyMiddleware",
    "RequestTimingMiddleware",
    "setup_middlewares",
]
