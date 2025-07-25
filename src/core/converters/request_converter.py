"""
OpenAI-to-Claude请求转换器

该模块提供将Anthropic格式请求转换为OpenAI格式的功能。
"""

import json
from typing import Any

from loguru import logger

from src.common.token_counter import TokenCounter
from src.models.anthropic import (
    AnthropicMessage,
    AnthropicRequest,
    AnthropicSystemMessage,
    AnthropicToolDefinition,
)
from src.models.openai import (
    OpenAIMessage,
    OpenAIRequest,
    OpenAITool,
    OpenAIToolFunction,
)

# 全局缓存配置对象


class AnthropicToOpenAIConverter:
    """将Anthropic请求转换为OpenAI格式"""

    @staticmethod
    async def get_target_model(anthropic_request: AnthropicRequest) -> str:
        """
        Args:
            anthropic_request: Anthropic特定请求对象

        Returns:
            选定的目标模型ID
        """
        original_model = anthropic_request.model
        # 如果模型包含逗号，直接返回原模型（保留复杂性）
        if original_model and "," in original_model:
            return original_model

        # 使用全局缓存的配置对象（同步获取，因为转换器不是异步的）
        from src.config.settings import get_config

        config = await get_config()
        if not config.models.default:
            return original_model

        resolved_model = config.models.default

        if "haiku" in original_model:
            resolved_model = config.models.small
        elif "sonnet" in original_model:
            resolved_model = config.models.default

        # 如果有tools定义，使用tool模型
        # if anthropic_request.tools and len(anthropic_request.tools) > 0:
        #     resolved_model = config.models.tool

        # 如果thinking为enabled，使用think模型
        if (
            anthropic_request.thinking is not None
            and anthropic_request.thinking["type"] == "enabled"
        ):
            resolved_model = config.models.think

        # 计算token数量
        token_counter = TokenCounter()
        total_tokens = await token_counter.count_tokens(
            anthropic_request.messages,
            anthropic_request.system,
            anthropic_request.tools,
        )
        if total_tokens > 1000 * 100:
            resolved_model = config.models.longContext

        return resolved_model

    @staticmethod
    async def convert_anthropic_to_openai(
        anthropic_request: AnthropicRequest,
        request_id: str = None,
    ) -> OpenAIRequest:
        """
        将Anthropic请求转换为OpenAI格式请求（异步版本）

        Args:
            anthropic_request: Anthropic格式的请求
            request_id: 请求ID用于日志追踪

        Returns:
            转换后的OpenAI格式请求
        """
        # 获取绑定了请求ID的logger
        from src.common.logging import get_logger_with_request_id

        bound_logger = get_logger_with_request_id(request_id)

        # 动态选择目标模型
        target_model = await AnthropicToOpenAIConverter.get_target_model(
            anthropic_request
        )

        bound_logger.debug(
            "将Anthropic请求转换为OpenAI格式",
            extra={
                "source_model": anthropic_request.model,
                "target_model": target_model,
                "message_count": len(anthropic_request.messages),
                "has_tools": anthropic_request.tools is not None,
                "has_system": anthropic_request.system is not None,
                "thinking": anthropic_request.thinking,
            },
        )

        # 转换消息列表
        messages = AnthropicToOpenAIConverter._convert_messages(anthropic_request)

        # 转换工具定义
        tools = AnthropicToOpenAIConverter._convert_tools(anthropic_request.tools)

        # 获取配置中的参数覆盖设置（异步获取）
        from src.config.settings import get_config

        config = await get_config()
        overrides = config.parameter_overrides

        # 应用参数覆盖逻辑（配置覆盖客户端请求参数）
        final_max_tokens = (
            overrides.max_tokens
            if overrides.max_tokens is not None
            else anthropic_request.max_tokens
        )
        final_temperature = (
            overrides.temperature
            if overrides.temperature is not None
            else anthropic_request.temperature
        )
        final_top_p = (
            overrides.top_p if overrides.top_p is not None else anthropic_request.top_p
        )
        final_top_k = (
            overrides.top_k if overrides.top_k is not None else anthropic_request.top_k
        )

        # 记录参数覆盖情况
        overridden_params = []
        if overrides.max_tokens is not None:
            overridden_params.append(
                f"max_tokens: {anthropic_request.max_tokens} -> {final_max_tokens}"
            )
        if overrides.temperature is not None:
            overridden_params.append(
                f"temperature: {anthropic_request.temperature} -> {final_temperature}"
            )
        if overrides.top_p is not None:
            overridden_params.append(
                f"top_p: {anthropic_request.top_p} -> {final_top_p}"
            )
        if overrides.top_k is not None:
            overridden_params.append(
                f"top_k: {anthropic_request.top_k} -> {final_top_k}"
            )

        if overridden_params:
            bound_logger.debug(f"应用参数覆盖: {', '.join(overridden_params)}")

        # 构建OpenAI请求
        openai_request = OpenAIRequest(
            model=target_model,
            messages=messages,
            max_tokens=final_max_tokens,
            temperature=final_temperature,
            top_p=final_top_p,
            top_k=final_top_k,
            stream=anthropic_request.stream,
            stop=anthropic_request.stop_sequences,
            tools=tools,
            tool_choice=AnthropicToOpenAIConverter._convert_tool_choice(
                anthropic_request.tool_choice
            ),
            # frequency_penalty=None,  # Anthropic没有直接对应的参数
            # presence_penalty=None,  # Anthropic没有直接对应的参数
            # logprobs=False,  # Anthropic默认不返回logprobs
            # n=1,  # Anthropic默认只生成一个响应
        )

        bound_logger.info(
            f"模型转换完成 - Anthropic: {anthropic_request.model} -> OpenAI: {openai_request.model}"
        )
        bound_logger.debug(
            f"OpenAI 请求体: {openai_request.model_dump_json(exclude_none=True)}"
        )
        return openai_request

    @staticmethod
    def _convert_messages(
        anthropic_request: AnthropicRequest,
    ) -> list[OpenAIMessage]:
        """
        将Anthropic消息列表转换为OpenAI消息格式

        Args:
            anthropic_request: Anthropic请求

        Returns:
            OpenAI格式的消息列表
        """
        messages = []

        # 处理system消息
        if anthropic_request.system:
            system_messages = AnthropicToOpenAIConverter._convert_system_message(
                anthropic_request.system
            )
            messages.extend(system_messages)

        # 转换用户和助手消息
        for anthropic_msg in anthropic_request.messages:
            converted_messages = AnthropicToOpenAIConverter._convert_single_message(
                anthropic_msg
            )
            # _convert_single_message现在可能返回多个消息（当包含tool_result时）
            if isinstance(converted_messages, list):
                messages.extend(converted_messages)
            else:
                messages.append(converted_messages)

        # 过滤不完整的tool_calls序列
        filtered_messages = AnthropicToOpenAIConverter._filter_incomplete_tool_calls(
            messages
        )

        return filtered_messages

    @staticmethod
    def _convert_system_message(
        system: str | list[AnthropicSystemMessage],
    ) -> list[OpenAIMessage]:
        """
        将Anthropic system字段转换为OpenAI system消息

        Args:
            system: Anthropic的system字段

        Returns:
            OpenAI格式的system消息列表
        """
        system_messages = []

        if isinstance(system, str):
            # 字符串格式的system提示
            system_messages.append(OpenAIMessage(role="system", content=system))
        elif isinstance(system, list):
            # 列表格式的system提示
            for system_msg in system:
                system_messages.append(
                    OpenAIMessage(role="system", content=system_msg.text)
                )

        return system_messages

    @staticmethod
    def _convert_single_message(
        anthropic_msg: AnthropicMessage,
    ) -> OpenAIMessage | list[OpenAIMessage]:
        """
        转换单个Anthropic消息为OpenAI格式

        Args:
            anthropic_msg: 单个Anthropic消息

        Returns:
            OpenAI格式的消息
        """
        if not anthropic_msg.content:
            raise ValueError("Anthropic消息内容不能为空")

        # 处理内容转换
        if isinstance(anthropic_msg.content, str):
            # 纯文本内容
            return OpenAIMessage(role=anthropic_msg.role, content=anthropic_msg.content)
        elif isinstance(anthropic_msg.content, list):
            # 复杂内容（包括工具调用等）
            content_parts = []
            tool_calls = []
            tool_results = []

            for content_block in anthropic_msg.content:
                if isinstance(content_block, dict):
                    # 处理字典格式的内容块
                    if content_block.get("type") == "tool_use":
                        # 将tool_use转换为OpenAI的tool_calls格式
                        tool_call = {
                            "id": content_block.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": content_block.get("name", ""),
                                "arguments": json.dumps(
                                    content_block.get("input", {}), ensure_ascii=False
                                ),
                            },
                        }
                        tool_calls.append(tool_call)
                    elif content_block.get("type") == "tool_result":
                        # 收集tool_result，稍后转换为独立的tool消息
                        tool_result_content = content_block.get("content", "")
                        if isinstance(tool_result_content, list):
                            tool_result_content = json.dumps(
                                tool_result_content, ensure_ascii=False
                            )
                        tool_results.append(
                            {
                                "tool_call_id": content_block.get("tool_use_id", ""),
                                "content": tool_result_content,
                            }
                        )
                    elif content_block.get("type") in ["text", "image_url"]:
                        # 只保留OpenAI支持的内容类型
                        content_parts.append(content_block)
                elif hasattr(content_block, "type"):
                    # 处理Pydantic模型对象
                    if content_block.type == "tool_use":
                        # 将tool_use转换为OpenAI的tool_calls格式
                        tool_call = {
                            "id": getattr(content_block, "id", ""),
                            "type": "function",
                            "function": {
                                "name": getattr(content_block, "name", ""),
                                "arguments": json.dumps(
                                    getattr(content_block, "input", {}),
                                    ensure_ascii=False,
                                ),
                            },
                        }
                        tool_calls.append(tool_call)
                    elif content_block.type == "tool_result":
                        # 收集tool_result，稍后转换为独立的tool消息
                        tool_result_content = getattr(content_block, "content", "")
                        if isinstance(tool_result_content, list):
                            tool_result_content = json.dumps(
                                tool_result_content, ensure_ascii=False
                            )
                        tool_results.append(
                            {
                                "tool_call_id": getattr(
                                    content_block, "tool_use_id", ""
                                ),
                                "content": tool_result_content,
                            }
                        )
                    elif content_block.type in ["text", "image_url"]:
                        # 只保留OpenAI支持的内容类型
                        content_parts.append(content_block.model_dump())
                else:
                    # 简单文本内容
                    content_parts.append({"type": "text", "text": str(content_block)})

            # 如果有tool_result，需要返回多个消息
            if tool_results:
                messages = []

                # 首先创建主消息（如果有非tool_result内容）
                if content_parts or tool_calls:
                    content = None
                    if content_parts:
                        # 如果只有一个文本内容，简化为字符串
                        if (
                            len(content_parts) == 1
                            and content_parts[0]
                            and isinstance(content_parts[0], dict)
                            and content_parts[0].get("type") == "text"
                        ):
                            content = content_parts[0]["text"]
                        else:
                            content = content_parts

                    # 创建主消息
                    main_msg = OpenAIMessage(role=anthropic_msg.role, content=content)
                    if tool_calls:
                        main_msg.tool_calls = tool_calls
                    messages.append(main_msg)

                # 然后为每个tool_result创建独立的tool消息
                for tool_result in tool_results:
                    tool_msg = OpenAIMessage(
                        role="tool",
                        content=tool_result["content"],
                        tool_call_id=tool_result["tool_call_id"],
                    )
                    messages.append(tool_msg)

                return messages
            else:
                # 没有tool_result，返回单个消息
                content = None
                if content_parts:
                    # 如果只有一个文本内容，简化为字符串
                    if (
                        len(content_parts) == 1
                        and content_parts[0]
                        and isinstance(content_parts[0], dict)
                        and content_parts[0].get("type") == "text"
                    ):
                        content = content_parts[0]["text"]
                    else:
                        content = content_parts

                # 创建OpenAI消息
                openai_msg = OpenAIMessage(role=anthropic_msg.role, content=content)

                # 如果有工具调用，添加到消息中
                if tool_calls:
                    openai_msg.tool_calls = tool_calls

                return openai_msg
        else:
            # 默认处理
            return OpenAIMessage(
                role=anthropic_msg.role, content=str(anthropic_msg.content)
            )

    @staticmethod
    def _convert_tools(
        anthropic_tools: list[AnthropicToolDefinition] | None,
    ) -> list[OpenAITool] | None:
        """
        将Anthropic工具定义转换为OpenAI工具格式

        Args:
            anthropic_tools: Anthropic工具定义列表

        Returns:
            OpenAI格式的工具列表或None
        """
        if not anthropic_tools:
            return None

        openai_tools = []
        for tool in anthropic_tools:
            openai_tool = OpenAITool(
                type="function",
                function=OpenAIToolFunction(
                    name=tool.name,
                    description=tool.description,
                    parameters=tool.input_schema,
                ),
            )
            openai_tools.append(openai_tool)

        return openai_tools if openai_tools else None

    @staticmethod
    def _convert_tool_choice(
        anthropic_tool_choice: str | dict[str, Any] | None,
    ) -> str | dict[str, Any] | None:
        """
        转换Anthropic的tool_choice到OpenAI格式

        Args:
            anthropic_tool_choice: Anthropic的tool_choice配置

        Returns:
            OpenAI格式的tool_choice或None
        """
        if anthropic_tool_choice is None:
            return None

        if isinstance(anthropic_tool_choice, str):
            # 直接映射字符串值
            if anthropic_tool_choice == "any":
                return "required"  # Anthropic的"any"对应OpenAI的"required"
            elif anthropic_tool_choice == "auto":
                return "auto"
            else:
                return anthropic_tool_choice

        elif isinstance(anthropic_tool_choice, dict):
            # 处理复杂配置
            if anthropic_tool_choice.get("type") == "tool":
                tool_name = anthropic_tool_choice.get("name", "")
                if tool_name:
                    return {"type": "function", "function": {"name": tool_name}}

        # 默认返回原始值
        return anthropic_tool_choice

    @staticmethod
    def _filter_incomplete_tool_calls(
        messages: list[OpenAIMessage],
    ) -> list[OpenAIMessage]:
        """过滤不完整的tool_calls序列

        OpenAI要求每个带有tool_calls的assistant消息后面必须跟对应的tool消息。
        此方法会移除没有对应tool消息的assistant消息中的tool_calls序列。
        同时也会移除没有对应assistant消息的独立tool消息。

        Args:
            messages: 原始消息列表

        Returns:
            过滤后的消息列表
        """
        if not messages:
            return messages

        filtered_messages = []
        i = 0

        while i < len(messages):
            current_msg = messages[i]

            # 如果当前消息是assistant且有tool_calls
            if (
                current_msg.role == "assistant"
                and current_msg.tool_calls
                and len(current_msg.tool_calls) > 0
            ):

                # 检查后续消息是否有对应的tool消息
                tool_call_ids = {
                    call.get("id") for call in current_msg.tool_calls if call.get("id")
                }
                found_tool_ids = set()

                # 查找后续的tool消息
                j = i + 1
                while j < len(messages) and messages[j].role == "tool":
                    tool_msg = messages[j]
                    if tool_msg.tool_call_id and tool_msg.tool_call_id in tool_call_ids:
                        found_tool_ids.add(tool_msg.tool_call_id)
                    j += 1

                # 如果所有tool_calls都有对应的tool消息，保留完整序列
                if found_tool_ids == tool_call_ids:
                    # 添加assistant消息
                    filtered_messages.append(current_msg)
                    # 添加对应的tool消息
                    for k in range(i + 1, j):
                        if messages[k].role == "tool":
                            filtered_messages.append(messages[k])
                    i = j  # 跳过已处理的tool消息
                else:
                    # 不完整的tool_calls序列，跳过整个序列
                    logger.debug(
                        f"过滤不完整的tool_calls序列: 期望{len(tool_call_ids)}个tool消息，实际找到{len(found_tool_ids)}个"
                    )
                    i = j  # 跳过整个不完整序列
            # 如果当前消息是独立的tool消息（前面没有对应的assistant消息）
            elif current_msg.role == "tool":
                # 检查前面是否有对应的assistant消息
                has_corresponding_assistant = False
                for k in range(i - 1, -1, -1):
                    prev_msg = messages[k]
                    if prev_msg.role == "assistant" and prev_msg.tool_calls:
                        # 检查tool_call_id是否匹配
                        for call in prev_msg.tool_calls:
                            if call.get("id") == current_msg.tool_call_id:
                                has_corresponding_assistant = True
                                break
                        if has_corresponding_assistant:
                            break
                    # 如果遇到非tool消息且不是assistant，则停止向前查找
                    elif prev_msg.role != "tool":
                        break

                # 只有当有对应的assistant消息时才保留tool消息
                if has_corresponding_assistant:
                    filtered_messages.append(current_msg)
                else:
                    logger.debug(
                        f"过滤没有对应assistant消息的独立tool消息: {current_msg.tool_call_id}"
                    )
                i += 1
            else:
                # 普通消息，直接添加
                filtered_messages.append(current_msg)
                i += 1

        return filtered_messages


async def validate_anthropic_request(
    request: AnthropicRequest, request_id: str = None
) -> None:
    """
    验证Anthropic请求的完整性

    Args:
        request: 要验证的Anthropic请求
        request_id: 请求ID用于日志追踪

    Raises:
        ValueError: 如果请求格式不正确
    """
    # 获取绑定了请求ID的logger
    from src.common.logging import get_logger_with_request_id

    bound_logger = get_logger_with_request_id(request_id)

    if not request.model:
        raise ValueError("模型字段不能为空")

    if not request.messages:
        raise ValueError("消息列表不能为空")

    if request.max_tokens <= 0:
        raise ValueError("max_tokens必须是正整数")

    if request.temperature is not None and not (0.0 <= request.temperature <= 1.0):
        raise ValueError("temperature必须在0.0到1.0之间")

    if request.top_p is not None and not (0.0 <= request.top_p <= 1.0):
        raise ValueError("top_p必须在0.0到1.0之间")

    for msg in request.messages:
        if not msg.role or msg.role not in ["user", "assistant"]:
            raise ValueError(f"消息角色必须是'user'或'assistant'，但得到: {msg.role}")

    bound_logger.debug("Anthropic请求验证通过")
