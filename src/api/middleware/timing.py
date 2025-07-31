"""请求ID中间件"""

import time
from collections.abc import Callable

from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from starlette.types import ASGIApp

# RequestIDMiddleware 已移除 - 如需request_id功能可重新添加


class RequestTimingMiddleware(BaseHTTPMiddleware):
    """记录请求处理时间的中间件"""

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()

        # 延迟导入
        from src.common.logging import (
            generate_request_id,
            get_logger_with_request_id,
            get_request_id_header_name,
        )

        # 生成请求ID并添加到请求状态中（默认启用）
        request_id = await generate_request_id()
        request.state.request_id = request_id

        # 获取绑定了请求ID的logger
        bound_logger = get_logger_with_request_id(request_id)

        try:
            response = await call_next(request)

            response_time = time.time() - start_time
            response_time_ms = round(response_time * 1000, 2)

            # 使用绑定了请求ID的logger记录响应
            bound_logger.info(
                f"请求完成 - Status: {response.status_code}, Time: {response_time_ms}ms"
            )

            response.headers["X-Process-Time"] = f"{response_time:.3f}s"
            header_name = await get_request_id_header_name()
            response.headers[header_name] = request_id

            return response

        except Exception as exc:
            response_time = time.time() - start_time
            error_content = (
                f'{{"error":"Internal Server Error","request_id":"{request_id}"}}'
            )

            response = Response(
                content=error_content,
                status_code=500,
                media_type="application/json",
            )
            header_name = await get_request_id_header_name()
            response.headers[header_name] = request_id

            # 使用绑定了请求ID的logger记录错误
            # 安全构造错误日志 - 避免格式字符串问题
            safe_url = str(request.url) if hasattr(request, "url") else "unknown"
            safe_method = request.method if hasattr(request, "method") else "unknown"

            bound_logger.error(
                "请求处理错误",
                error_type=type(exc).__name__,
                error_message=str(exc),
                url=safe_url,
                method=safe_method,
                exc_info=True,
            )

            return response


def setup_middlewares(app: FastAPI) -> None:
    """设置所有中间件"""
    # 只保留请求计时中间件
    app.add_middleware(RequestTimingMiddleware)
