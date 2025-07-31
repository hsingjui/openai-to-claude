"""
Anthropic /v1/messages 端点处理程序

实现Anthropic native messages API与OpenAI API的转换和代理
"""

import asyncio
import json
from collections.abc import AsyncGenerator

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from src.core.clients.openai_client import OpenAIServiceClient
from src.core.converters.request_converter import (
    AnthropicToOpenAIConverter,
)
from src.core.converters.response_converter import OpenAIToAnthropicConverter
from src.models.anthropic import (
    AnthropicMessageResponse,
    AnthropicRequest,
)
from src.models.errors import get_error_response

router = APIRouter(prefix="/v1", tags=["messages"])


class MessagesHandler:
    """处理Anthropic /v1/messages 端点请求"""

    def __init__(self, config):
        self.request_converter = AnthropicToOpenAIConverter()
        self.response_converter = OpenAIToAnthropicConverter()
        self.config = config
        self._config = None
        self.client = OpenAIServiceClient(
            api_key=config.openai.api_key,
            base_url=config.openai.base_url,
        )

    @classmethod
    async def create(cls, config=None):
        """异步工厂方法创建 MessagesHandler 实例"""
        if config is None:
            from src.config.settings import get_config

            config = await get_config()

        instance = cls.__new__(cls)
        instance.request_converter = AnthropicToOpenAIConverter()
        instance.response_converter = OpenAIToAnthropicConverter()
        instance.config = config
        instance._config = config
        instance.client = OpenAIServiceClient(
            api_key=config.openai.api_key,
            base_url=config.openai.base_url,
        )
        return instance

    async def process_message(
        self, request: AnthropicRequest, request_id: str = None
    ) -> AnthropicMessageResponse:
        """处理非流式消息请求"""
        # 获取绑定了请求ID的logger
        from src.common.logging import get_logger_with_request_id

        bound_logger = get_logger_with_request_id(request_id)

        try:
            bound_logger.debug("处理非流式请求")
            # 验证请求
            # await validate_anthropic_request(request, request_id)
            # 将 Anthropic 请求转换为 OpenAI 格式（异步）
            openai_request = await self.request_converter.convert_anthropic_to_openai(
                request, request_id
            )

            # 发送到 OpenAI
            openai_response = await self.client.send_request(
                openai_request, request_id=request_id
            )
            bound_logger.debug(
                f"OpenAI 响应: {json.dumps(openai_response, ensure_ascii=False)}"
            )

            # 将 OpenAI 响应转回 Anthropic 格式
            anthropic_response = await self.response_converter.convert_response(
                openai_response, request.model, request_id
            )
            # 安全地提取响应文本
            response_text = "empty"
            if (
                anthropic_response.content
                and len(anthropic_response.content) > 0
                and hasattr(anthropic_response.content[0], "text")
                and anthropic_response.content[0].text
            ):
                response_text = anthropic_response.content[0].text
            bound_logger.info(
                f"Anthropic 响应生成完成 - Text: {response_text[:100]}..., Usage: {anthropic_response.usage}"
            )

            return anthropic_response

        except ValidationError as e:
            bound_logger.warning(f"Validation error - Errors: {e.errors()}")
            error_response = get_error_response(
                422, details={"validation_errors": e.errors(), "request_id": request_id}
            )
            raise HTTPException(status_code=422, detail=error_response.model_dump())

        except json.JSONDecodeError as e:
            # 专门处理JSON解析错误，这通常发生在OpenAI响应解析时
            bound_logger.exception(
                f"JSON解析错误 - Error: {str(e)}, Position: {e.pos if hasattr(e, 'pos') else 'unknown'}"
            )
            error_response = get_error_response(
                502,
                message="上游服务返回无效JSON格式",
                details={"json_error": str(e), "request_id": request_id},
            )
            raise HTTPException(status_code=502, detail=error_response.model_dump())
        except HTTPException as e:
            bound_logger.exception(
                f"处理非流式消息请求错误 - Type: {type(e).__name__}, Error: {str(e)}"
            )
            error_response = get_error_response(
                e.status_code, message=str(e.detail), details={"request_id": request_id}
            )
            raise HTTPException(
                status_code=e.status_code,
                detail=error_response.model_dump(exclude_none=True),
            )

        except Exception as e:
            bound_logger.exception(
                f"处理非流式消息请求错误 - Type: {type(e).__name__}, Error: {str(e)}"
            )
            error_response = get_error_response(
                500, message=str(e), details={"request_id": request_id}
            )
            raise HTTPException(
                status_code=500, detail=error_response.model_dump(exclude_none=True)
            )

    async def process_stream_message(
        self, request: AnthropicRequest, request_id: str = None
    ) -> AsyncGenerator[str, None]:
        """处理流式消息请求，使用新的流式转换器"""
        if not request.stream:
            raise ValueError("流式响应参数必须为true")

        # 获取绑定了请求ID的logger
        from src.common.logging import get_logger_with_request_id

        bound_logger = get_logger_with_request_id(request_id)

        try:
            # await validate_anthropic_request(request, request_id)
            openai_request = await self.request_converter.convert_anthropic_to_openai(
                request, request_id
            )

            # 创建 OpenAI 流式数据源
            async def openai_stream_generator():
                bound_logger.info("开始OpenAI流式生成")
                chunk_count = 0
                async for chunk in self.client.send_streaming_request(
                    openai_request, request_id=request_id
                ):
                    # 跳过被解析器过滤掉的不完整chunk（通常是tool_calls片段）
                    if chunk is not None:
                        chunk_count += 1
                        # 将 OpenAI 响应对象转换为字符串格式
                        bound_logger.debug(f"OpenAI event: {chunk}")
                        yield f"{chunk}\n\n"
                bound_logger.debug(f"OpenAI流式生成完成，总共{chunk_count}个chunk")

            # 使用新的流式转换器
            bound_logger.info("开始流式转换")
            async for (
                anthropic_event
            ) in self.response_converter.convert_openai_stream_to_anthropic_stream(
                openai_stream_generator(), model=request.model, request_id=request_id
            ):
                bound_logger.debug(f"Anthropic event: {anthropic_event}")
                yield anthropic_event
            bound_logger.info("流式转换完成")

        except (ValidationError, ValueError) as e:
            error_detail = e.errors() if hasattr(e, "errors") else str(e)
            bound_logger.warning(f"流式请求验证失败 - Errors: {error_detail}")
            error_response = get_error_response(422, message=str(error_detail))
            # 在错误响应中添加请求ID
            error_data = error_response.model_dump()
            if request_id:
                error_data["request_id"] = request_id
            yield f"event: error\ndata: {json.dumps(error_data, ensure_ascii=False)}\n\n"

        except json.JSONDecodeError as e:
            # 专门处理流式模式下的JSON解析错误
            bound_logger.exception(
                f"流式模式JSON解析错误 - Error: {str(e)}, Position: {e.pos if hasattr(e, 'pos') else 'unknown'}"
            )
            error_response = get_error_response(
                502,
                message="流式响应中发现无效JSON格式",
                details={"json_error": str(e), "request_id": request_id},
            )
            error_data = error_response.model_dump()
            if request_id:
                error_data["request_id"] = request_id
            yield f"event: error\ndata: {json.dumps(error_data, ensure_ascii=False)}\n\n"

        except Exception as e:
            bound_logger.exception(
                f"流式请求处理错误 - Type: {type(e).__name__}, Error: {str(e)}"
            )
            error_response = get_error_response(500, message=str(e))
            # 在错误响应中添加请求ID
            error_data = error_response.model_dump()
            if request_id:
                error_data["request_id"] = request_id
            yield f"event: error\ndata: {json.dumps(error_data, ensure_ascii=False)}\n\n"


@router.post("/messages")
async def messages_endpoint(request: Request, background_tasks: BackgroundTasks):
    """
    Anthropic /v1/messages 端点

    这个端点实现了Anthropic原生messages API的主要功能：
    - 接受Anthropic格式的请求
    - 转换为OpenAI格式发送到后端
    - 返回Anthropic格式的响应
    """
    # 从应用状态获取消息处理器（已由main.py在启动时初始化）
    handler: MessagesHandler = request.app.state.messages_handler

    # 获取请求ID（由中间件生成，如果启用的话）
    from src.common.logging import (
        get_logger_with_request_id,
        get_request_id_from_request,
    )

    request_id = get_request_id_from_request(request)
    bound_logger = get_logger_with_request_id(request_id)

    # 记录请求
    client_ip = request.client.host if request.client else "unknown"
    bound_logger.info(
        f"收到Anthropic请求 - Method: {request.method}, URL: {str(request.url)}, IP: {client_ip}"
    )

    try:
        # 解析请求体
        body = await request.json()
        # 记录请求
        log_body = body.copy()
        log_body["tools"] = []
        bound_logger.debug(
            f"Anthropic请求体 - Model: {body.get('model', 'unknown')}, Messages: {len(body.get('messages', []))}, Stream: {body.get('stream', False)}\n{json.dumps(log_body, ensure_ascii=False, indent=2)}"
        )

        anthropic_request = AnthropicRequest(**body)

        # 记录清理后的请求信息（移除敏感信息）
        # safe_body = sanitize_for_logging(body)
        # logger.debug("请求已清理", request_body=safe_body)

        # 根据请求类型处理响应
        if anthropic_request.stream:
            # 流式响应 - 优化配置确保真正的流式效果
            async def stream_wrapper():
                """包装器确保流式响应的立即传输"""
                try:
                    async for chunk in handler.process_stream_message(
                        anthropic_request, request_id=request_id
                    ):
                        # 立即传输每个chunk，不缓冲
                        # chunk已经是完整的SSE格式字符串，编码后返回
                        yield chunk.encode("utf-8")
                        # 强制刷新缓冲区（在某些环境中有效）
                        await asyncio.sleep(0)
                except Exception as e:
                    # 如果流式处理出错，记录完整错误并发送错误事件
                    bound_logger.exception(f"流式处理出错 - Error: {str(e)}")
                    error_data = {"error": str(e)}
                    if request_id:
                        error_data["request_id"] = request_id
                    error_event = f"event: error\ndata: {json.dumps(error_data)}\n\n"
                    yield error_event.encode("utf-8")

            return StreamingResponse(
                stream_wrapper(),
                media_type="text/event-stream; charset=utf-8",
                headers={
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",  # 禁用nginx缓冲
                    "X-Content-Type-Options": "nosniff",
                    "Transfer-Encoding": "chunked",
                    "Access-Control-Allow-Origin": "*",  # CORS支持
                    "Access-Control-Allow-Headers": "*",
                    "Access-Control-Allow-Methods": "*",
                    "X-Proxy-Buffering": "no",  # 禁用代理缓冲
                    "Buffering": "no",  # 禁用缓冲
                },
            )
        else:
            # 非流式响应
            response = await handler.process_message(
                anthropic_request, request_id=request_id
            )
            json_response = JSONResponse(content=response.model_dump(exclude_none=True))
            if request_id:
                json_response.headers["X-Request-ID"] = request_id
            return json_response

    except ValidationError as e:
        bound_logger.warning(f"请求验证失败 - Errors: {e.errors()}")
        error_response = get_error_response(
            422, details={"validation_errors": e.errors()}
        )
        error_detail = error_response.model_dump()
        error_detail["request_id"] = request_id
        raise HTTPException(status_code=422, detail=error_detail)

    except json.JSONDecodeError as e:
        bound_logger.warning(f"请求中的JSON格式错误 - Error: {str(e)}")
        error_response = get_error_response(400, message="无效的JSON格式")
        error_detail = error_response.model_dump()
        error_detail["request_id"] = request_id
        raise HTTPException(status_code=400, detail=error_detail)

    except Exception as e:
        # 检查是否为HTTPException，避免重复记录已处理的错误
        if isinstance(e, HTTPException):
            # HTTPException已经在内层处理过，直接重新抛出
            raise e

        bound_logger.exception(
            f"在messages端点发生意外错误 - Type: {type(e).__name__}, Error: {str(e)}"
        )
        error_response = get_error_response(500, message=str(e))
        error_detail = error_response.model_dump()
        error_detail["request_id"] = request_id
        raise HTTPException(status_code=500, detail=error_detail)
