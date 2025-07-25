"""
通用工具模块

提供项目中共享的工具和实用功能。

主要功能:
- 日志配置和管理
- 请求ID生成和追踪
- Token计数功能
- 异常处理工具

使用示例:
    from src.common import configure_logging, request_logger

    configure_logging()
    logger = request_logger.get_logger()
"""

# 导入日志相关功能
from .logging import (
    RequestLogger,
    configure_logging,
    generate_request_id,
    get_logger_with_request_id,
    get_request_id_from_request,
    get_request_id_header_name,
    log_exception,
    request_logger,
    should_enable_request_id,
)

# 导入Token计数功能
from .token_counter import TokenCounter, token_counter

__all__ = [
    # 日志功能
    "configure_logging",
    "RequestLogger",
    "request_logger",
    "generate_request_id",
    "should_enable_request_id",
    "get_request_id_header_name",
    "get_request_id_from_request",
    "log_exception",
    "get_logger_with_request_id",
    # Token计数
    "TokenCounter",
    "token_counter",
]
