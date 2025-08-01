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

        self.usage = None
        self.anthropic_stop_reason = None


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
    reasoning_content = delta.get("reasoning_content")
    if reasoning_content is not None and reasoning_content != "":
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

    if thinking_content == "":
        thinking_content = None

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
    state.anthropic_stop_reason = anthropic_stop_reason

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

    nnthropic_usage = AnthropicUsage(
        input_tokens=input_tokens,
        output_tokens=completion_tokens,  # 使用计算得到的值
    )
    state.usage = nnthropic_usage
    message_delta = AnthropicStreamMessage(
        type=AnthropicStreamEventTypes.MESSAGE_DELTA,
        delta=MessageDelta(
            stop_reason=anthropic_stop_reason,
            stop_sequence=None,
        ),
        usage=nnthropic_usage,
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


def _log_stream_completion_details(
    state: StreamState,
    request_id: str = None,
    model: str = "claude-sonnet-4-20250514",
) -> None:
    """
    流式响应完成时的详细日志记录（输出完整JSON格式）

    Args:
        state: 流状态对象，包含累积的内容信息
        stop_reason: 停止原因
        input_tokens: 输入token数量
        output_tokens: 输出token数量
        request_id: 请求ID，用于绑定日志
    """
    from src.common.logging import get_logger_with_request_id

    bound_logger = get_logger_with_request_id(request_id)

    try:
        usage: AnthropicUsage = state.usage
        stop_reason = state.anthropic_stop_reason
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens
        # 构建完整的Anthropic响应JSON
        response_json = _build_complete_anthropic_response(
            state, stop_reason, input_tokens, output_tokens, model
        )

        # 输出完整的JSON格式日志
        formatted_json = json.dumps(response_json, ensure_ascii=False, indent=4)
        bound_logger.info(f"流式响应生成完成: {formatted_json}")

    except Exception as e:
        # 记录日志失败不应影响正常流程
        bound_logger.warning(f"流式响应日志记录失败 - Error: {str(e)}")


def _build_complete_anthropic_response(
    state: StreamState,
    stop_reason: str,
    input_tokens: int,
    output_tokens: int,
    model: str = "claude-sonnet-4-20250514",
) -> dict[str, Any]:
    """
    构建完整的Anthropic响应JSON格式

    Args:
        state: 流状态对象
        stop_reason: 停止原因
        input_tokens: 输入token数量
        output_tokens: 输出token数量
        request_id: 请求ID
        model: 模型名称

    Returns:
        dict: 完整的Anthropic响应JSON
    """
    # 构建content数组
    content_blocks = []

    # 1. 处理思考内容
    if state.thinking_started:
        thinking_text = ""
        # 从accumulated_content中提取思考相关内容
        for content in state.accumulated_content:
            if content and isinstance(content, str):
                # 检查是否包含思考内容的标识
                if any(
                    marker in content
                    for marker in [
                        "<think>",
                        "</think>",
                        "Let me think",
                        "I need to think",
                    ]
                ):
                    thinking_text += content

        if thinking_text.strip():
            # 清理思考内容
            clean_thinking = (
                thinking_text.replace("<think>", "").replace("</think>", "").strip()
            )
            if clean_thinking:
                content_blocks.append(
                    {
                        "type": "thinking",
                        "thinking": clean_thinking,
                        "signature": str(int(time.time() * 1000)),
                    }
                )

    # 2. 处理普通文本内容
    if state.content_started:
        # 提取非思考、非工具的文本内容
        text_content = ""
        for content in state.accumulated_content:
            if content and isinstance(content, str):
                # 过滤掉思考内容和工具相关内容
                if not any(
                    marker in content
                    for marker in [
                        "<think>",
                        "</think>",
                        "Let me think",
                        "I need to think",
                    ]
                ):
                    # 检查是否为工具名称或JSON参数
                    if not (
                        content.strip().startswith("{")
                        and content.strip().endswith("}")
                    ):
                        # 检查是否为常见工具名称
                        tool_names = ["search", "calculate", "web_search", "tool_"]
                        if not any(tool in content.lower() for tool in tool_names):
                            text_content += content

        if text_content.strip():
            content_blocks.append({"type": "text", "text": text_content.strip()})

    # 3. 处理工具调用
    if state.tool_calls:
        for tool_index, tool_info in state.tool_calls.items():
            tool_name = tool_info.get("name", "unknown")
            tool_id = tool_info.get(
                "id", f"call_{int(time.time() * 1000)}_{tool_index}"
            )
            tool_args = tool_info.get("arguments", "{}")

            # 解析工具参数
            try:
                tool_input = json.loads(tool_args) if tool_args else {}
            except (json.JSONDecodeError, TypeError):
                tool_input = {"arguments": tool_args}

            content_blocks.append(
                {
                    "type": "tool_use",
                    "id": tool_id,
                    "name": tool_name,
                    "input": tool_input,
                }
            )

    # 如果没有任何内容，添加空文本块
    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    # 构建完整响应
    response = {
        "id": state.message_id,
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": model,
        "stop_reason": stop_reason,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "service_tier": "standard",
        },
    }

    return response
