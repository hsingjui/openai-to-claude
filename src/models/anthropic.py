"""Anthropic API 数据模型定义"""

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field


class AnthropicStreamEventTypes:
    """Anthropic流式响应事件类型常量"""

    # 消息相关事件
    MESSAGE_START = "message_start"
    MESSAGE_DELTA = "message_delta"
    MESSAGE_STOP = "message_stop"

    # 内容块相关事件
    CONTENT_BLOCK_START = "content_block_start"
    CONTENT_BLOCK_DELTA = "content_block_delta"
    CONTENT_BLOCK_STOP = "content_block_stop"

    # 其他事件
    PING = "ping"


class AnthropicContentTypes:
    """Anthropic内容类型常量"""

    # 基础内容类型
    TEXT = "text"
    IMAGE = "image"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    THINKING = "thinking"

    # 增量类型
    TEXT_DELTA = "text_delta"
    INPUT_JSON_DELTA = "input_json_delta"
    THINKING_DELTA = "thinking_delta"
    SIGNATURE_DELTA = "signature_delta"


class AnthropicMessageTypes:
    """Anthropic消息类型常量"""

    MESSAGE = "message"
    ERROR = "error"


class AnthropicRoles:
    """Anthropic角色常量"""

    USER = "user"
    ASSISTANT = "assistant"


class AnthropicMessageContent(BaseModel):
    """Anthropic消息内容项"""

    type: Literal["text", "thinking", "image", "tool_use", "tool_result"] = Field(
        description="内容类型"
    )
    text: str | None = Field(None, description="文本内容（当type为text时）")
    source: dict[str, Any] | None = Field(None, description="当type为image时的源信息")
    id: str | None = Field(None, description="工具调用ID（当type为tool_use时）")
    name: str | None = Field(None, description="工具名称（当type为tool_use时）")
    input: dict[str, Any] | None = Field(
        None, description="工具输入参数（当type为tool_use时）"
    )
    tool_use_id: str | None = Field(
        None, description="工具使用ID（当type为tool_result时）"
    )
    content: str | list[dict[str, Any]] | None = Field(
        None, description="工具结果内容（当type为tool_result时）"
    )
    is_error: bool | None = Field(
        None, description="工具调用是否为错误结果（当type为tool_result时）"
    )


class AnthropicMessage(BaseModel):
    """Anthropic消息格式"""

    role: Literal["user", "assistant"] = Field(description="消息角色")
    content: str | list[AnthropicMessageContent] = Field(description="消息内容")


class AnthropicSystemMessage(BaseModel):
    """Anthropic系统消息"""

    type: Literal["text"] = Field(
        default=AnthropicContentTypes.TEXT, description="系统消息类型，固定为text"
    )
    text: str = Field(description="系统消息文本内容")


class AnthropicToolDefinition(BaseModel):
    """Anthropic工具定义"""

    name: str = Field(description="工具名称")
    description: Optional[str] = Field(None, description="工具描述")
    input_schema: Optional[dict[str, Any]] = Field(
        None, description="JSON Schema格式的输入参数定义"
    )
    type: Optional[str] = Field(None, description="工具名称")
    max_uses: Optional[int] = Field(None, description="工具名称")


class AnthropicRequest(BaseModel):
    """Anthropic API请求模型"""

    model: str = Field(description="使用的模型ID，如claude-3-5-sonnet-20241022")
    messages: list[AnthropicMessage] = Field(description="对话消息列表")
    max_tokens: int = Field(description="最大输出token数量")
    system: str | list[AnthropicSystemMessage] | None = Field(
        None, description="系统提示信息"
    )
    tools: list[AnthropicToolDefinition] | None = Field(
        None, description="可用工具定义"
    )
    tool_choice: str | dict[str, Any] | None = Field(None, description="工具选择配置")
    metadata: dict[str, Any] | None = Field(None, description="可选元数据")
    stop_sequences: list[str] | None = Field(None, description="停止序列")
    stream: bool | None = Field(False, description="是否使用流式响应")
    temperature: float | None = Field(None, ge=0.0, le=1.0, description="采样温度")
    top_p: float | None = Field(None, ge=0.0, le=1.0, description="top-p采样参数")
    top_k: int | None = Field(None, ge=1, le=1000, description="top-k采样参数")
    thinking: bool | dict[str, Any] | None = Field(
        None, description="是否启用推理模型模式或配置对象"
    )


class AnthropicToolUse(BaseModel):
    """工具使用响应"""

    id: str = Field(description="工具调用ID")
    type: str = Field(default=AnthropicContentTypes.TOOL_USE, description="响应类型")
    name: str = Field(description="工具名称")
    input: dict[str, Any] = Field(description="工具输入参数")


class AnthropicTextContent(BaseModel):
    """文本内容响应"""

    type: str = Field(default=AnthropicContentTypes.TEXT, description="内容类型")
    text: str = Field(description="文本内容")


class AnthropicContentBlock(BaseModel):
    """Anthropic内容块"""

    type: Literal["text", "tool_use", "thinking"]
    text: str | None = Field(None, description="文本内容，当type为text时")
    id: str | None = Field(None, description="工具调用ID，当type为tool_use时")
    name: str | None = Field(None, description="工具名称，当type为tool_use时")
    input: dict[str, Any] | None = Field(
        None, description="工具输入，当type为tool_use时"
    )
    thinking: str | None = Field(None, description="思考内容，当type为thinking时")
    signature: str | None = Field(None, description="思考内容签名，当type为thinking时")


class AnthropicUsage(BaseModel):
    """Anthropic使用统计"""

    input_tokens: int = Field(1, description="输入token数量")
    output_tokens: int | None = Field(1, description="输出token数量")
    cache_creation_input_tokens: int | None = Field(
        0, description="缓存创建输入token数量"
    )
    cache_read_input_tokens: int | None = Field(0, description="缓存读取输入token数量")
    service_tier: str | None = Field("standard", description="服务层级")


class MessageDelta(BaseModel):
    """消息增量"""

    stop_reason: str | None = Field(None, description="停止原因")
    stop_sequence: str | None = Field(None, description="停止序列")


class AnthropicStreamMessageStartMessage(BaseModel):
    """Anthropic流式消息开始事件中的消息详情"""

    id: str = Field(description="消息ID")
    type: str = Field(default=AnthropicMessageTypes.MESSAGE, description="消息类型")
    role: Literal["assistant"] = Field(
        default=AnthropicRoles.ASSISTANT, description="消息角色"
    )
    model: str = Field(description="使用的模型ID")
    content: list[Any] = Field([], description="内容块，通常为空")
    stop_reason: str | None = Field(None, description="停止原因")
    stop_sequence: str | None = Field(None, description="停止序列")
    usage: AnthropicUsage = Field(description="使用统计")


class AnthropicStreamMessage(BaseModel):
    """流式消息开始事件"""

    type: str = Field(
        default=AnthropicStreamEventTypes.MESSAGE_START, description="事件类型"
    )
    message: AnthropicStreamMessageStartMessage = Field(None, description="消息详情")
    delta: MessageDelta = Field(None, description="消息增量")
    usage: AnthropicUsage = Field(None, description="使用统计")


class Delta(BaseModel):
    """文本增量"""

    type: str = Field(default=AnthropicContentTypes.TEXT_DELTA, description="增量类型")
    text: str | None = Field(None, description="文本增量内容")
    thinking: str | None = Field(None, description="思考内容")
    signature: str | None = Field(None, description="签名内容")
    partial_json: str | None = Field(None, description="部分JSON字符串")


class InputJsonDelta(BaseModel):
    """输入JSON增量"""

    type: str = Field(
        default=AnthropicContentTypes.INPUT_JSON_DELTA, description="增量类型"
    )


class AnthropicUsageDelta(BaseModel):
    """使用统计增量"""

    output_tokens: int = Field(description="输出token数量增量")


class AnthropicMessageResponse(BaseModel):
    """Anthropic消息响应"""

    id: str = Field(description="响应唯一ID")
    type: str = Field(default=AnthropicMessageTypes.MESSAGE, description="响应类型")
    role: Literal["assistant"] = Field(
        default=AnthropicRoles.ASSISTANT, description="消息角色"
    )
    content: list[AnthropicContentBlock] = Field(description="消息内容块")
    model: str = Field(description="使用的模型ID")
    stop_reason: str | None = Field(None, description="停止原因")
    stop_sequence: str | None = Field(None, description="停止序列")
    usage: AnthropicUsage = Field(description="使用统计")


class AnthropicErrorDetail(BaseModel):
    """Anthropic错误详情"""

    type: str = Field(description="错误类型")
    message: str = Field(description="错误消息")


class AnthropicErrorResponse(BaseModel):
    """Anthropic错误响应模型"""

    type: str = Field(default=AnthropicMessageTypes.ERROR, description="响应类型")
    error: AnthropicErrorDetail = Field(description="错误详情")


class ContentBlock(BaseModel):
    """内容块"""

    type: str = Field(default=AnthropicContentTypes.TEXT, description="内容块类型")
    text: str | None = Field(None, description="文本内容")
    thinking: str | None = Field(None, description="思考内容")
    signature: str | None = Field(None, description="签名内容")
    # tool_use相关字段
    id: str | None = Field(None, description="工具调用ID，当type为tool_use时")
    name: str | None = Field(None, description="工具名称，当type为tool_use时")
    input: dict[str, Any] | None = Field(
        None, description="工具输入，当type为tool_use时"
    )


class AnthropicStreamContentBlockStart(BaseModel):
    """流式内容块开始"""

    type: str = Field(
        default=AnthropicStreamEventTypes.CONTENT_BLOCK_START, description="事件类型"
    )
    index: int = Field(default=0, description="内容块索引")
    content_block: ContentBlock = Field(
        default_factory=lambda: ContentBlock(type=AnthropicContentTypes.TEXT, text="")
    )


class AnthropicStreamContentBlock(BaseModel):
    """流式内容块增量"""

    type: str = Field(
        default=AnthropicStreamEventTypes.CONTENT_BLOCK_DELTA, description="事件类型"
    )
    index: int = Field(0, description="内容块索引")
    delta: Delta = Field(None, description="增量内容")
    content_block: ContentBlock = Field(None, description="内容块")
    usage: Delta | None = Field(None, description="使用统计")


class AnthropicStreamContentBlockStop(BaseModel):
    """流式内容块结束"""

    type: str = Field(
        default=AnthropicStreamEventTypes.CONTENT_BLOCK_STOP, description="事件类型"
    )
    index: int = Field(0, description="内容块索引")


class AnthropicPing(BaseModel):
    """流式ping消息"""

    type: str = Field(default=AnthropicStreamEventTypes.PING, description="事件类型")


AnthropicStreamResponse = Union[
    AnthropicStreamMessage,
    AnthropicStreamContentBlockStart,
    AnthropicStreamContentBlock,
    AnthropicStreamContentBlockStop,
    AnthropicPing,
]
