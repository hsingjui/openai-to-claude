from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from src.models.errors import get_error_response


class APIKeyMiddleware(BaseHTTPMiddleware):
    """中间件：验证API密钥"""

    def __init__(self, app, api_key: str):
        super().__init__(app)
        self.api_key = api_key

    async def dispatch(self, request: Request, call_next):
        # 检查是否为需要密钥验证的路径
        if not self._requires_auth(request.url.path):
            # 跳过认证，直接处理请求
            response = await call_next(request)
            return response
            
        # 从请求头中获取API密钥
        token = request.headers.get("x-api-key")

        if token != self.api_key:
            error_response = await get_error_response(401, message="API密钥无效")

            # 直接返回401响应，而不是抛出异常
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=401,
                content=error_response.dict()
            )

        response = await call_next(request)
        return response
    
    def _requires_auth(self, path: str) -> bool:
        """检查路径是否需要API密钥验证"""
        # 只有 /v1/messages 需要密钥验证
        auth_required_paths = ["/v1/messages"]
        return path in auth_required_paths
