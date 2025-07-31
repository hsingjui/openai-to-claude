"""
OpenAI-to-Anthropic 响应转换器

实现将OpenAI格式的响应转换为Anthropic格式的功能
"""

import json
import time
import traceback
from collections.abc import AsyncIterator
from typing import Any

from ...common.token_cache import get_cached_tokens
from loguru import logger

from src.models.anthropic import (
    AnthropicContentBlock,
    AnthropicContentTypes,
    AnthropicMessageResponse,
    AnthropicMessageTypes,
    AnthropicPing,
    AnthropicRoles,
    AnthropicStreamContentBlock,
    AnthropicStreamContentBlockStart,
    AnthropicStreamContentBlockStop,
    AnthropicStreamEventTypes,
    AnthropicStreamMessage,
    AnthropicStreamMessageStartMessage,
    AnthropicUsage,
    ContentBlock,
    Delta,
    MessageDelta,
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
        model = OpenAIToAnthropicConverter.extract_model_name(
            original_model, openai_response.get("model")
        )

        # 映射完成原因
        stop_reason = OpenAIToAnthropicConverter._map_finish_reason(
            choice.finish_reason
        )

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
    def extract_model_name(original_model: str, response_model: str) -> str:
        """
        提取模型名称，优先使用原始请求的模型名

        Args:
            original_model: 原始请求的模型名
            response_model: 响应中的模型名

        Returns:
            str: 最终使用的模型名
        """
        return original_model if original_model else response_model

    @staticmethod
    def _map_finish_reason(finish_reason: str) -> str:
        """
        将OpenAI完成原因映射到Anthropic格式

        Args:
            finish_reason: OpenAI的完成原因

        Returns:
            str: Anthropic格式的停止原因
        """
        mapping = {
            "stop": "end_turn",
            "length": "max_tokens",
            "content_filter": "content_filter",
            "tool_calls": "tool_use",
            "function_call": "tool_use",
        }
        return mapping.get(finish_reason, "end_turn")

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
                            continue

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
                            break

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


class StreamState:
    """流状态管理类"""

    def __init__(self):
        self.message_id = f"msg_{int(time.time() * 1000)}"
        # 响应开始
        self.has_started = False
        # 文本内容开始
        self.content_started = False
        # 文本内容是否已开始（用于工具调用索引计算）
        self.has_text_content_started = False
        # 响应结束
        self.has_finished = False
        # 思考内容开始
        self.thinking_started = False
        # 思考内容结束
        self.thinking_finish = False
        # 内容块索引
        self.content_index = 0
        self.buffer = ""
        # 思考内容模式 None 无 1 <think> 2 reasoning_content
        self.thinking_mode = None

        # 计数器
        self.total_chunks = 0
        # 工具调用块计数器
        self.tool_call_chunks = 0

        # 工具调用管理
        self.tool_calls: dict[int, dict[str, Any]] = {}
        self.tool_call_index_to_content_block_index: dict[int, int] = {}

        # 新增：累积所有输出内容用于token计算
        self.accumulated_content: list[str] = []


def check_thinking_content(delta: dict[str, Any], state: StreamState) -> bool:
    """检查是否为思考内容"""
    if not delta or not isinstance(delta, dict):
        return False
    if state.thinking_mode is not None:
        return True
    # 检查是否开始思考模式
    content = delta.get("content") or ""
    if not isinstance(content, str):
        content = str(content) if content is not None else ""
    # 检查是否为<think>或<thinking>
    if "<think>" in content or "<thinking>" in content:
        state.thinking_mode = 1
        return True
    # 检查是否为reasoning_content
    if delta.get("reasoning_content"):
        state.thinking_mode = 2
        return True
    return False


def check_regular_content(delta: dict[str, Any], state: StreamState) -> bool:
    """检查是否为普通文本内容"""
    if state.thinking_mode is not None:
        return False

    if "content" in delta and delta["content"]:
        return True
    return False


def format_event(event_type: str, data: dict[str, Any]) -> str:
    """格式化事件为 SSE 格式"""
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def process_regular_content(delta: dict[str, Any], state: StreamState) -> list[str]:
    """处理普通文本内容"""
    events = []

    # 判断是否为普通文本内容
    if not check_regular_content(delta, state):
        return events

    if not state.content_started:
        state.content_started = True
        state.has_text_content_started = True
        content_block_start = AnthropicStreamContentBlockStart(
            index=state.content_index,
            content_block=ContentBlock(
                text="",
            ),
        )
        events.append(
            format_event(
                AnthropicStreamEventTypes.CONTENT_BLOCK_START,
                content_block_start.model_dump(exclude_none=True),
            )
        )
        # ping 事件
        events.append(
            format_event(AnthropicStreamEventTypes.PING, AnthropicPing().model_dump())
        )

    # 累积内容用于token计算
    content = delta.get("content", "")
    if content:
        state.accumulated_content.append(content)

    anthropic_chunk = AnthropicStreamContentBlock(
        index=state.content_index,
        delta=Delta(
            type=AnthropicContentTypes.TEXT_DELTA,
            text=delta["content"],
        ),
    )
    events.append(
        format_event(
            AnthropicStreamEventTypes.CONTENT_BLOCK_DELTA,
            anthropic_chunk.model_dump(exclude_none=True),
        )
    )
    return events


def process_thinking_content(delta: dict[str, Any], state: StreamState) -> list[str]:
    """处理思考内容"""
    events = []

    is_thinking = check_thinking_content(delta, state)

    if not state.thinking_started and is_thinking:
        state.thinking_started = True
        content_block_start = AnthropicStreamContentBlockStart(
            index=state.content_index,
            content_block=ContentBlock(
                type=AnthropicContentTypes.THINKING,
                thinking="",
            ),
        )
        events.append(
            format_event(
                AnthropicStreamEventTypes.CONTENT_BLOCK_START,
                content_block_start.model_dump(exclude_none=True),
            )
        )
        events.append(
            format_event(AnthropicStreamEventTypes.PING, AnthropicPing().model_dump())
        )

    # print(f"is_thinking: {is_thinking}")
    # 提取思考内容
    thinking_content = None
    if state.thinking_mode is not None:
        if state.thinking_mode == 1:
            content = delta.get("content")
            if "</think>" in content or "</thinking>" in content:
                state.thinking_mode = None
            thinking_content = content.replace("<think>", "").replace("</think>", "")
        elif state.thinking_mode == 2:
            thinking_content = delta.get("reasoning_content")

    if thinking_content is not None and thinking_content != "":
        # 累积思考内容用于token计算
        state.accumulated_content.append(thinking_content)
        # 处理普通思考内容
        thinking_chunk = AnthropicStreamContentBlock(
            index=state.content_index,
            delta=Delta(
                type=AnthropicContentTypes.THINKING_DELTA,
                thinking=thinking_content,
            ),
        )
        events.append(
            format_event(
                AnthropicStreamEventTypes.CONTENT_BLOCK_DELTA,
                thinking_chunk.model_dump(exclude_none=True),
            )
        )

    if thinking_content is None and not state.thinking_finish:
        # 结束思考
        state.thinking_mode = None
        state.thinking_finish = True
        # signature_delta
        signature_delta = AnthropicStreamContentBlock(
            index=state.content_index,
            delta=Delta(
                type=AnthropicContentTypes.SIGNATURE_DELTA,
                signature=f"{int(time.time()*1000)}",
            ),
        )
        events.append(
            format_event(
                AnthropicStreamEventTypes.CONTENT_BLOCK_DELTA,
                signature_delta.model_dump(exclude_none=True),
            )
        )
        # content_block_stop
        content_block_stop = AnthropicStreamContentBlockStop(
            index=state.content_index,
        )
        events.append(
            format_event(
                AnthropicStreamEventTypes.CONTENT_BLOCK_STOP,
                content_block_stop.model_dump(exclude_none=True),
            )
        )
        state.content_index += 1
        return events

    return events


def process_tool_calls(delta: dict[str, Any], state: StreamState) -> list[str]:
    """处理工具调用"""
    events = []
    state.tool_call_chunks += 1
    processed_indices: set[int] = set()

    for tool_call in delta["tool_calls"]:
        tool_call_index = tool_call.get("index", 0)
        if tool_call_index in processed_indices:
            continue
        processed_indices.add(tool_call_index)

        # 处理新的工具调用
        if tool_call_index not in state.tool_call_index_to_content_block_index:
            # 计算新的内容块索引
            new_content_block_index = (
                len(state.tool_call_index_to_content_block_index) + 1
                if state.has_text_content_started
                else len(state.tool_call_index_to_content_block_index)
            )

            # 如果不是第一个内容块，先结束上一个
            if new_content_block_index != 0:
                content_block_stop = AnthropicStreamContentBlockStop(
                    index=state.content_index,
                )
                events.append(
                    format_event(
                        AnthropicStreamEventTypes.CONTENT_BLOCK_STOP,
                        content_block_stop.model_dump(exclude_none=True),
                    )
                )
                state.content_index += 1

            # 记录映射关系
            state.tool_call_index_to_content_block_index[tool_call_index] = (
                new_content_block_index
            )

            # 生成工具调用信息
            tool_call_id = (
                tool_call.get("id")
                or f"call_{int(time.time() * 1000)}_{tool_call_index}"
            )
            tool_call_name = (
                tool_call.get("function", {}).get("name") or f"tool_{tool_call_index}"
            )

            # 累积工具名称用于token计算
            if tool_call_name and not tool_call_name.startswith("tool_"):
                state.accumulated_content.append(tool_call_name)

            # 创建内容块开始事件
            content_block_start = AnthropicStreamContentBlockStart(
                index=state.content_index,
                content_block=ContentBlock(
                    type=AnthropicContentTypes.TOOL_USE,
                    id=tool_call_id,
                    name=tool_call_name,
                    input={},
                ),
            )
            events.append(
                format_event(
                    AnthropicStreamEventTypes.CONTENT_BLOCK_START,
                    content_block_start.model_dump(exclude_none=True),
                )
            )
            events.append(
                format_event(
                    AnthropicStreamEventTypes.PING, AnthropicPing().model_dump()
                )
            )

            # 保存工具调用信息
            state.tool_calls[tool_call_index] = {
                "id": tool_call_id,
                "name": tool_call_name,
                "arguments": "",
                "content_block_index": new_content_block_index,
            }

        # 更新已存在的工具调用信息
        elif (
            tool_call.get("id")
            and tool_call.get("function", {}).get("name")
            and tool_call_index in state.tool_calls
        ):

            existing_tool_call = state.tool_calls[tool_call_index]
            was_temporary = existing_tool_call["id"].startswith(
                "call_"
            ) and existing_tool_call["name"].startswith("tool_")

            if was_temporary:
                existing_tool_call["id"] = tool_call["id"]
                existing_tool_call["name"] = tool_call["function"]["name"]

        # 处理工具调用参数
        function_args = tool_call.get("function", {}).get("arguments")
        if function_args and not state.has_finished:
            # 累积工具调用参数用于token计算
            state.accumulated_content.append(function_args)

            if tool_call_index in state.tool_calls:
                state.tool_calls[tool_call_index]["arguments"] += function_args

            try:
                anthropic_chunk = AnthropicStreamContentBlock(
                    index=state.content_index,
                    delta=Delta(
                        type=AnthropicContentTypes.INPUT_JSON_DELTA,
                        partial_json=function_args,
                    ),
                )
                events.append(
                    format_event(
                        AnthropicStreamEventTypes.CONTENT_BLOCK_DELTA,
                        anthropic_chunk.model_dump(exclude_none=True),
                    )
                )
            except Exception as e:
                logger.warning(
                    f"Failed to process tool call arguments - Error: {str(e)}",
                    exc_info=True,
                )
                # 尝试修复参数格式
                try:
                    fixed_args = (
                        function_args.replace("\x00-\x1f\x7f-\x9f", "")
                        .replace("\\", "\\\\")
                        .replace('"', '\\"')
                    )
                    fixed_chunk = AnthropicStreamContentBlock(
                        index=state.content_index,
                        delta=Delta(
                            type=AnthropicContentTypes.INPUT_JSON_DELTA,
                            partial_json=fixed_args,
                        ),
                    )
                    events.append(
                        format_event(
                            AnthropicStreamEventTypes.CONTENT_BLOCK_DELTA,
                            fixed_chunk.model_dump(exclude_none=True),
                        )
                    )
                except Exception as fix_error:
                    logger.error(
                        f"Failed to fix tool call arguments - Error: {str(fix_error)}",
                        exc_info=True,
                    )

    return events


def process_finish_event(
    chunk_data: dict[str, Any],
    state: StreamState,
    request_id: str = None,
) -> list[str]:
    """处理完成事件"""
    events = []
    state.has_finished = True

    # 结束最后一个内容块
    # if state.content_started or state.tool_call_chunks > 0:
    content_block_stop = AnthropicStreamContentBlockStop(
        index=state.content_index,
    )
    events.append(
        format_event(
            AnthropicStreamEventTypes.CONTENT_BLOCK_STOP,
            content_block_stop.model_dump(exclude_none=True),
        )
    )

    # 映射停止原因
    stop_reason_mapping = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "stop_sequence",
    }

    choice = chunk_data.get("choices", [{}])[0]
    finish_reason = choice.get("finish_reason")
    anthropic_stop_reason = stop_reason_mapping.get(finish_reason, "end_turn")

    # 发送 message_delta 事件
    usage_data = choice.get("usage", None)
    if usage_data is None:
        usage_data = chunk_data.get("usage", {})
    # 计算输出token数量
    input_tokens = usage_data.get("prompt_tokens", 0)
    if input_tokens is None or input_tokens == 0:
        cached_tokens = get_cached_tokens(request_id, True)
        if cached_tokens:
            input_tokens = cached_tokens
    completion_tokens = usage_data.get("completion_tokens", 0)
    # 如果OpenAI没有返回completion_tokens，使用我们的计算方法
    if not completion_tokens and state.accumulated_content:
        from src.common.token_counter import token_counter

        # 将累积的内容转换为内容块格式，复用现有计算逻辑
        mock_content_blocks = []
        combined_text = "".join(state.accumulated_content)
        if combined_text:
            # 创建模拟内容块（与现有TokenCounter.count_response_tokens兼容）
            mock_content_blocks.append(
                type("ContentBlock", (), {"text": combined_text})()
            )

        completion_tokens = token_counter.count_response_tokens(mock_content_blocks)

    message_delta = AnthropicStreamMessage(
        type=AnthropicStreamEventTypes.MESSAGE_DELTA,
        delta=MessageDelta(
            stop_reason=anthropic_stop_reason,
            stop_sequence=None,
        ),
        usage=AnthropicUsage(
            input_tokens=input_tokens,
            output_tokens=completion_tokens,  # 使用计算得到的值
        ),
    )
    events.append(
        format_event(
            AnthropicStreamEventTypes.MESSAGE_DELTA,
            message_delta.model_dump(exclude_none=True),
        )
    )

    # 发送 message_stop 事件
    message_stop = AnthropicStreamMessage(
        type=AnthropicStreamEventTypes.MESSAGE_STOP,
    )
    events.append(
        format_event(
            AnthropicStreamEventTypes.MESSAGE_STOP,
            message_stop.model_dump(exclude_none=True),
        )
    )

    return events


def safe_json_parse(json_str: str) -> dict[str, Any]:
    """
    安全地解析JSON字符串，处理单引号等格式问题

    Args:
        json_str: 待解析的JSON字符串

    Returns:
        解析后的字典对象，解析失败时返回空字典
    """
    if not json_str:
        return {}

    try:
        # 首先尝试标准JSON解析
        return json.loads(json_str)
    except json.JSONDecodeError:
        try:
            # 尝试处理单引号问题：将单引号替换为双引号
            # 这是一个简单的修复，适用于大多数情况
            corrected_json = json_str.replace("'", '"')
            return json.loads(corrected_json)
        except json.JSONDecodeError as e:
            logger.warning(
                f"JSON解析失败，使用空字典 - Error: {e}, Content: {json_str[:100]}..."
            )
            return {}
