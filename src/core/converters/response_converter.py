"""
OpenAI-to-Anthropic 响应转换器

实现将OpenAI格式的响应转换为Anthropic格式的功能
"""

import json
import time
import traceback
from collections.abc import AsyncIterator
from typing import Any

from src.common.token_cache import get_cached_tokens

from .stream_converters import (
    StreamState,
    _log_stream_completion_details,
    format_event,
    process_finish_event,
    process_regular_content,
    process_thinking_content,
    process_tool_calls,
    safe_json_parse,
)


from src.models.anthropic import (
    AnthropicContentBlock,
    AnthropicContentTypes,
    AnthropicMessageResponse,
    AnthropicMessageTypes,
    AnthropicRoles,
    AnthropicStreamEventTypes,
    AnthropicStreamMessage,
    AnthropicStreamMessageStartMessage,
    AnthropicUsage,
)
from src.models.openai import (
    OpenAIChoice,
    OpenAIMessage,
)


class OpenAIToAnthropicConverter:
    """OpenAI响应到Anthropic格式的转换器"""

    @staticmethod
    async def convert_response(
        openai_response: dict[str, Any],
        original_model: str = None,
        request_id: str = None,
    ) -> AnthropicMessageResponse:
        """
        将OpenAI非流式响应转换为Anthropic格式

        Args:
            openai_response: OpenAI响应字典
            original_model: 原始请求的Anthropic模型
            request_id: 请求ID，用于获取缓存的token数量

        Returns:
            AnthropicMessageResponse: 转换后的Anthropic格式响应
        """
        choices = openai_response.get("choices")
        if not choices:
            raise ValueError("OpenAI响应没有有效的choices")

        # 使用第一个choice作为主要响应
        first_choice_data = choices[0]
        message_data = first_choice_data.get("message", {})
        choice = OpenAIChoice(
            message=(
                OpenAIMessage(
                    role=message_data.get("role"),
                    content=message_data.get("content", ""),
                    tool_calls=message_data.get("tool_calls"),
                )
                if message_data
                else None
            ),
            finish_reason=first_choice_data.get("finish_reason"),
            index=first_choice_data.get("index", 0),
        )

        # 提取内容块
        content_blocks = (
            OpenAIToAnthropicConverter._extract_content_blocks_with_reasoning(
                choice, first_choice_data
            )
        )

        # 转换使用统计
        usage_data = openai_response.get("usage", {})
        usage = OpenAIToAnthropicConverter._convert_usage(
            usage_data, request_id, content_blocks
        )

        # 确定模型ID
        model = original_model if original_model else openai_response.get("model")

        mapping = {
            "stop": "end_turn",
            "length": "max_tokens",
            "content_filter": "content_filter",
            "tool_calls": "tool_use",
            "function_call": "tool_use",
        }

        # 映射完成原因
        stop_reason = mapping.get(choice.finish_reason, "end_turn")

        return AnthropicMessageResponse(
            id=openai_response.get("id", ""),
            type=AnthropicMessageTypes.MESSAGE,
            role=AnthropicRoles.ASSISTANT,
            content=content_blocks,
            model=model,
            stop_reason=stop_reason,
            stop_sequence=None,
            usage=usage,
        )

    @staticmethod
    def _extract_content_blocks_with_reasoning(
        choice, first_choice_data
    ) -> list[AnthropicContentBlock]:
        """
        从OpenAI choice中提取内容块，包括推理内容

        Args:
            choice: OpenAI选择对象
            first_choice_data: OpenAI choice的原始数据

        Returns:
            List[AnthropicContentBlock]: 内容块列表
        """
        if not choice.message:
            return []

        content_blocks = []
        message_data = first_choice_data.get("message", {})

        # 处理推理内容 - 作为独立的thinking类型内容块
        reasoning_content = message_data.get("reasoning_content")
        if (
            reasoning_content
            and isinstance(reasoning_content, str)
            and reasoning_content.strip()
        ):
            content_blocks.append(
                AnthropicContentBlock(
                    type=AnthropicContentTypes.THINKING,
                    thinking=reasoning_content.strip(),
                    signature=f"{int(time.time()*1000)}",
                )
            )

        # 处理普通内容 - 作为独立的text类型内容块
        content_str = message_data.get("content", "")
        if content_str and isinstance(content_str, str) and content_str.strip():
            # 检查content中是否包含<think>标签
            if "<think>" in content_str and "</think>" in content_str:
                # 分离思考内容和普通内容
                import re

                think_pattern = r"<think>(.*?)</think>"
                think_matches = re.findall(think_pattern, content_str, re.DOTALL)

                # 如果还没有添加thinking块且找到了思考内容，添加thinking块
                if think_matches and not any(
                    block.type == AnthropicContentTypes.THINKING
                    for block in content_blocks
                ):
                    thinking_content = think_matches[0].strip()
                    if thinking_content:
                        content_blocks.append(
                            AnthropicContentBlock(
                                type=AnthropicContentTypes.THINKING,
                                thinking=thinking_content,
                                signature=f"{int(time.time()*1000)}",
                            )
                        )

                # 移除<think>标签，保留普通内容
                clean_content = re.sub(
                    think_pattern, "", content_str, flags=re.DOTALL
                ).strip()
                if clean_content:
                    content_blocks.append(
                        AnthropicContentBlock(
                            type=AnthropicContentTypes.TEXT, text=clean_content
                        )
                    )
            else:
                # 没有思考标签，直接作为普通内容
                content_blocks.append(
                    AnthropicContentBlock(
                        type=AnthropicContentTypes.TEXT, text=content_str.strip()
                    )
                )

        # 处理工具调用
        if choice.message.tool_calls:
            from src.models.openai import OpenAIToolCall

            for tool_call_data in choice.message.tool_calls:
                tool_call = OpenAIToolCall.model_validate(tool_call_data)
                if hasattr(tool_call, "function") and tool_call.function:
                    content_blocks.append(
                        AnthropicContentBlock(
                            type=AnthropicContentTypes.TOOL_USE,
                            id=tool_call.id,
                            name=tool_call.function.name,
                            input=(
                                safe_json_parse(tool_call.function.arguments)
                                if tool_call.function.arguments
                                else {}
                            ),
                        )
                    )

        # 如果没有任何内容，返回空的text块
        if not content_blocks:
            content_blocks = [
                AnthropicContentBlock(type=AnthropicContentTypes.TEXT, text="")
            ]

        return content_blocks

    @staticmethod
    def _convert_usage(
        usage_data: dict[str, Any], request_id: str = None, content_blocks: list = None
    ) -> AnthropicUsage:
        """
        将OpenAI使用统计转换为Anthropic格式，支持缓存fallback

        Args:
            usage_data: OpenAI使用统计数据
            request_id: 请求ID，用于获取缓存的token数量
            content_blocks: 内容块列表，用于计算输出token数量

        Returns:
            AnthropicUsage: Anthropic格式的使用统计
        """
        prompt_tokens = usage_data.get("prompt_tokens", 0) if usage_data else 0
        completion_tokens = usage_data.get("completion_tokens", 0) if usage_data else 0

        # 如果OpenAI没有返回prompt_tokens，使用缓存的值
        if not prompt_tokens and request_id:
            from src.common.token_cache import get_cached_tokens

            cached_tokens = get_cached_tokens(request_id)
            if cached_tokens:
                prompt_tokens = cached_tokens

        # 如果OpenAI没有返回completion_tokens，使用我们的计算方法
        if not completion_tokens and content_blocks:
            from src.common.token_counter import token_counter

            # 使用同步版本，保持简单性（KISS原则）
            completion_tokens = token_counter.count_response_tokens(content_blocks)

        return AnthropicUsage(
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
        )

    @staticmethod
    async def convert_openai_stream_to_anthropic_stream(
        openai_stream: AsyncIterator[str],
        model: str = "unknown",
        request_id: str = None,
    ) -> AsyncIterator[str]:
        """将 OpenAI 流式响应转换为 Anthropic 流式响应格式

        Args:
            openai_stream: OpenAI 流式数据源
            model: 模型名称
            request_id: 请求ID用于日志追踪

        Yields:
            str: Anthropic 格式的流式事件字符串
        """
        # 获取绑定了请求ID的logger
        from src.common.logging import get_logger_with_request_id

        bound_logger = get_logger_with_request_id(request_id)

        state = StreamState()

        try:
            async for chunk in openai_stream:
                if state.has_finished:
                    break

                state.buffer += chunk
                lines = state.buffer.split("\n")
                state.buffer = lines.pop() if lines else ""

                for line in lines:
                    if state.has_finished:
                        break

                    if not line.startswith("data: "):
                        continue

                    data = line[6:]
                    if data == "[DONE]":
                        continue
                    try:
                        chunk_data = json.loads(data)
                        state.total_chunks += 1
                        # 处理错误
                        if "error" in chunk_data:
                            error_event = {
                                "type": "error",
                                "message": {
                                    "type": "api_error",
                                    "message": json.dumps(chunk_data["error"]),
                                },
                            }
                            yield format_event("error", error_event)
                            continue

                        # 发送 message_start 事件
                        if not state.has_started and not state.has_finished:
                            # 获取input token缓存
                            input_tokens = 0
                            cached_tokens = get_cached_tokens(request_id)
                            if cached_tokens:
                                input_tokens = cached_tokens
                            state.has_started = True
                            message_start = AnthropicStreamMessage(
                                message=AnthropicStreamMessageStartMessage(
                                    id=state.message_id,
                                    model=model,
                                    usage=AnthropicUsage(input_tokens=input_tokens),
                                ),
                            )
                            yield format_event(
                                AnthropicStreamEventTypes.MESSAGE_START,
                                message_start.model_dump(exclude=["delta", "usage"]),
                            )

                        choices = chunk_data.get("choices", [])
                        if not choices:
                            continue

                        choice = choices[0]
                        delta = choice.get("delta", None)
                        if delta is None:
                            continue

                        content = delta.get("content", None)
                        reasoning_content = delta.get("reasoning_content", None)
                        tool_calls = delta.get("tool_calls", None)

                        # 检查是否有任何内容需要处理
                        has_content = content is not None and content != ""
                        has_reasoning = (
                            reasoning_content is not None and reasoning_content != ""
                        )
                        has_tool_calls = tool_calls is not None and len(tool_calls) > 0

                        if not has_content and not has_reasoning and not has_tool_calls:
                            if choice.get("finish_reason") is None:
                                continue

                        # 处理思考内容
                        events = process_thinking_content(delta, state)
                        if events:
                            for event in events:
                                yield event

                        # 处理普通文本内容
                        events = process_regular_content(delta, state)
                        if events:
                            for event in events:
                                yield event
                            continue

                        # 处理工具调用
                        if has_tool_calls:
                            events = process_tool_calls(delta, state)
                            if events:
                                for event in events:
                                    yield event
                                continue

                        # 处理完成事件
                        finish_reason = choice.get("finish_reason")
                        if finish_reason:
                            finish_events = process_finish_event(
                                chunk_data, state, request_id
                            )
                            for event in finish_events:
                                yield event

                            # 在所有事件生成完成后记录详细日志（遵循KISS原则）
                            _log_stream_completion_details(
                                state,
                                request_id,
                                model,
                            )

                    except json.JSONDecodeError as parse_error:
                        bound_logger.error(
                            f"Parse error - Error: {str(parse_error.args[0])}, Data: {data[:100]}",
                            exc_info=True,
                        )
                    except Exception as e:
                        bound_logger.error(
                            f"Unexpected error processing chunk - Error: {str(e)}",
                            exc_info=True,
                        )
                        traceback.print_exc()

        except Exception as error:
            bound_logger.error(
                f"Stream conversion error - Error: {str(error)}", exc_info=True
            )
            error_event = {
                "type": "error",
                "message": {"type": "api_error", "message": str(error)},
            }
            yield format_event("error", error_event)
