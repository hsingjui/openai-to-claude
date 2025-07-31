"""OpenAI API client for making asynchronous requests to OpenAI endpoints."""

import json
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from loguru import logger

from src.models.errors import StandardErrorResponse, get_error_response
from src.models.openai import OpenAIRequest, OpenAIStreamResponse


class OpenAIClientError(Exception):
    """Base exception for OpenAI client errors."""

    def __init__(self, error_response: StandardErrorResponse):
        self.error_response = error_response
        super().__init__(str(error_response))


class OpenAIServiceClient:
    """Async OpenAI API client with connection pooling and retry logic."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
    ):
        """Initialize OpenAI client with connection pool.

        Args:
            api_key: OpenAI API密钥
            base_url: OpenAI API基础URL
            timeout: 请求超时时间(秒)
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        self.client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Connection": "keep-alive",
            },
            # 确保自动解压缩响应
            follow_redirects=True,
            timeout=timeout,
        )

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.aclose()

    async def aclose(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def send_request(
        self,
        request: OpenAIRequest,
        endpoint: str = "/chat/completions",
        request_id: str = None,
    ) -> dict[str, Any]:
        """Send synchronous request to OpenAI API.

        Args:
            request: OpenAI request object
            endpoint: API endpoint path
            request_id: 请求ID用于日志追踪

        Returns:
            OpenAI API响应

        Raises:
            OpenAIClientError: 当API返回错误时
        """
        # 获取绑定了请求ID的logger
        from src.common.logging import get_logger_with_request_id

        bound_logger = get_logger_with_request_id(request_id)

        url = f"{self.base_url}{endpoint}"
        request_data = request.model_dump(exclude_none=True)

        # 记录请求详情
        bound_logger.info(
            f"发送OpenAI请求 - URL: {url}, Model: {request_data.get('model', 'unknown')}, Messages: {len(request_data.get('messages', []))}"
        )

        try:
            response = await self.client.post(
                url,
                json=request_data,
            )
            response.raise_for_status()
            # 记录响应状态
            content_type = response.headers.get("content-type", "unknown")
            bound_logger.info(
                f"收到OpenAI响应 - Status: {response.status_code}, Content-Type: {content_type}, Size: {len(response.content)} bytes"
            )

            # 使用 response.text 让 httpx 自动处理编码和解压缩
            try:
                text = response.text
                result = json.loads(text)

                # 记录响应内容（如果启用详细日志）
                bound_logger.debug(
                    f"OpenAI响应内容 - ID: {result.get('id', 'unknown')}, Model: {result.get('model', 'unknown')}, Usage: {result.get('usage', {})}"
                )

            except json.JSONDecodeError as e:
                # 记录详细的JSON解析错误信息
                response_preview = (
                    response.text[:500] if response.text else "Empty response"
                )
                content_type = response.headers.get("content-type", "unknown")
                bound_logger.exception(
                    f"OpenAI JSON解析失败 - Status: {response.status_code}, Content-Type: {content_type}, "
                    f"Error: {str(e)}, Response Preview: {response_preview}"
                )
                # 抛出包含更多上下文信息的异常
                raise json.JSONDecodeError(
                    f"Failed to parse OpenAI response (Status: {response.status_code}): {str(e)}",
                    response.text,
                    e.pos,
                )
            return result
        except httpx.HTTPStatusError as e:
            # 安全读取响应内容（非流式模式）
            response_body = ""
            try:
                response_body = e.response.text
            except httpx.ResponseNotRead:
                # 如果响应未被读取，直接获取错误信息
                response_body = str(e)

            bound_logger.error(
                f"OpenAI API返回错误 - Endpoint: {endpoint}, Status: {e.response.status_code}, Response: {response_body[:200]}"
            )

            raise OpenAIClientError(
                get_error_response(
                    status_code=e.response.status_code,
                    message=response_body,
                    details={"type": "http_error"},
                )
            )

        except httpx.TimeoutException as e:
            bound_logger.error(
                f"OpenAI API request timeout - Endpoint: {endpoint}, Timeout: {self.timeout}s"
            )
            raise OpenAIClientError(
                get_error_response(
                    status_code=504,
                    message=str(e),
                    details={"type": "timeout_error", "original_error": str(e)},
                )
            )

        except httpx.ConnectError as e:
            bound_logger.error(
                f"OpenAI API connection error - Endpoint: {endpoint}, Error: {str(e)}"
            )
            raise OpenAIClientError(
                get_error_response(
                    status_code=502,
                    message=str(e),
                    details={"type": "connection_error", "original_error": str(e)},
                )
            )

    async def send_streaming_request(
        self,
        request: OpenAIRequest,
        endpoint: str = "/chat/completions",
        request_id: str = None,
    ) -> AsyncGenerator[str, None]:
        """Send streaming request to OpenAI API.

        Args:
            request: OpenAI request object
            endpoint: API endpoint path
            request_id: 请求ID用于日志追踪

        Yields:
            原始的Server-Sent Events数据行

        Raises:
            OpenAIClientError: 当API返回错误时
        """
        # 获取绑定了请求ID的logger
        from src.common.logging import get_logger_with_request_id

        bound_logger = get_logger_with_request_id(request_id)

        url = f"{self.base_url}{endpoint}"

        # Ensure streaming is enabled
        request_dict = request.model_dump(exclude_none=True)
        request_dict["stream"] = True

        # 记录流式请求详情
        bound_logger.info(
            f"发送OpenAI流式请求 - URL: {url}, Model: {request_dict.get('model', 'unknown')}, Messages: {len(request_dict.get('messages', []))}, Stream: True"
        )
        
        try:
            async with self.client.stream(
                "POST",
                url,
                json=request_dict,
            ) as response:
                response.raise_for_status()

                # 记录流式响应开始
                content_type = response.headers.get("content-type", "unknown")
                bound_logger.info(
                    f"开始接收OpenAI流式响应 - Status: {response.status_code}, Content-Type: {content_type}"
                )

                buffer = ""

                async for chunk_bytes in response.aiter_bytes(chunk_size=1024):
                    chunk_text = chunk_bytes.decode("utf-8", errors="ignore")
                    buffer += chunk_text

                    # 处理完整的行
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()

                        # 直接转发非空行
                        if line:
                            # logger.debug(f"Forwarding line: {line}")
                            yield line
                            # 检查是否结束
                            if line == "data: [DONE]":
                                return

                # 处理最后可能剩余的数据
                if buffer.strip():
                    yield buffer.strip()

        except httpx.HTTPStatusError as e:
            error_body = ""
            try:
                # 尝试读取完整的错误响应体
                error_body = await e.response.aread()
                error_body = error_body.decode("utf-8", errors="ignore")
            except Exception as read_error:
                error_body = f"无法读取错误响应: {str(read_error)}"

            # 记录完整错误信息，但在日志中截断过长内容
            error_summary = (
                error_body[:500] + "..." if len(error_body) > 500 else error_body
            )
            bound_logger.error(
                f"OpenAI API 错误 - Status: {e.response.status_code}, URL: {url}"
            )
            bound_logger.error(f"Error Response: {error_summary}")
            raise OpenAIClientError(
                get_error_response(
                    status_code=e.response.status_code,
                    message=f"HTTP {e.response.status_code} error",
                    details={"type": "http_error"},
                )
            )

        except httpx.TimeoutException as e:
            bound_logger.error(f"OpenAI API 超时 - Error: {str(e)}")
            raise OpenAIClientError(
                get_error_response(
                    status_code=504,
                    message="Request timeout",
                    details={"type": "timeout_error"},
                )
            )

        except httpx.ConnectError as e:
            bound_logger.error(f"OpenAI API 连接错误 - Error: {str(e)}")
            raise OpenAIClientError(
                get_error_response(
                    status_code=502,
                    message="Connection error",
                    details={"type": "connection_error"},
                )
            )

    async def _parse_streaming_chunk(
        self, chunk_data: str, tool_calls_state: dict
    ) -> OpenAIStreamResponse | None:
        """解析流式响应chunk，优雅处理不完整的JSON数据。

        Args:
            chunk_data: JSON字符串的响应块
            tool_calls_state: 预留参数（未使用）

        Returns:
            解析后的响应对象，如果数据不完整则返回None
        """
        import json

        try:
            # 尝试解析JSON数据
            raw_data = json.loads(chunk_data)

            result = OpenAIStreamResponse.model_validate(raw_data)
            return result

        except json.JSONDecodeError as e:
            # JSON解析失败，通常是因为数据被分割，静默跳过
            logger.debug(
                f"Skipping incomplete JSON chunk - Error: {str(e)}, Data: {chunk_data[:100]}"
            )
            return None
        except Exception as e:
            # Pydantic验证失败，可能是tool_calls的增量数据不完整
            logger.debug(
                f"Skipping chunk due to validation error - Error: {str(e)}, Data: {chunk_data[:100]}"
            )
            return None

    async def health_check(self) -> dict[str, bool]:
        """Check OpenAI API availability.

        Returns:
            健康检查结果
        """
        try:
            url = f"{self.base_url}/models"
            response = await self.client.get(url)

            return {
                "openai_service": response.status_code == 200,
                "api_accessible": True,
                "last_check": True,
            }

        except Exception as e:
            logger.exception(f"OpenAI health check failed - Error: {str(e)}")
            return {
                "openai_service": False,
                "api_accessible": False,
                "last_check": True,
            }
