"""
API模块

提供FastAPI应用的路由、处理器和中间件。

主要功能:
- API路由定义
- 请求处理器
- 中间件集成
- 健康检查端点

子模块:
- handlers: API请求处理器
- routes: 路由定义
- middleware: 中间件实现
"""

# 导入路由和处理器
from .handlers import MessagesHandler, messages_endpoint
from .handlers import router as handlers_router

# 导入中间件
from .middleware import (
    APIKeyMiddleware,
    RequestTimingMiddleware,
    setup_middlewares,
)
from .routes import health_check
from .routes import router as routes_router

__all__ = [
    # 路由
    "routes_router",
    "handlers_router",
    "health_check",
    "health_check_detailed",
    # 处理器
    "MessagesHandler",
    "messages_endpoint",
    # 中间件
    "APIKeyMiddleware",
    "RequestTimingMiddleware",
    "setup_middlewares",
]
