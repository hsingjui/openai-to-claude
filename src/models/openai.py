"""OpenAI API 数据模型定义"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class OpenAIMessageContent(BaseModel):
    """OpenAI消息内容项"""

    type: Literal["text", "image_url"] = Field(description="内容类型")
    text: str | None = Field(None, description="文本内容")
    image_url: dict[str, str] | None = Field(None, description="图像URL配置")


class OpenAIMessage(BaseModel):
    """OpenAI消息格式"""

    role: Literal["system", "user", "assistant", "tool"] = Field(description="消息角色")
    content: str | list[OpenAIMessageContent] | None = Field(description="消息内容")
    name: str | None = Field(None, description="消息作者名称")
    tool_calls: list[dict[str, Any]] | None = Field(
        None, description="工具调用信息（当role为assistant时）"
    )
    tool_call_id: str | None = Field(
        None, description="工具调用ID（当role为tool时）"
    )
    refusal: str | None = Field(None, description="拒绝服务的详细信息")
    reasoning_content: str | None = Field(None, description="推理内容")
    annotations: list[dict[str, Any]] | None = Field(
        None, description="模型生成的标注"
    )


class OpenAIImageUrl(BaseModel):
    """OpenAI图像URL"""

    url: str = Field(description="图像URL")
    detail: Literal["auto", "low", "high"] | None = Field(
        "auto", description="图像细节级别"
    )


class OpenAIToolCallFunction(BaseModel):
    """工具调用函数"""

    name: str | None = Field(None, description="函数名称")  # 改为可选，因为流式响应中可能缺失
    arguments: str | None = Field(None, description="JSON格式的函数参数")  # 改为可选，支持增量传输


class OpenAIToolCall(BaseModel):
    """工具调用"""

    id: str = Field(description="工具调用ID")
    type: Literal["function"] = Field("function", description="调用类型")
    function: OpenAIToolCallFunction = Field(description="函数详情")


class OpenAIDeltaToolCall(BaseModel):
    """工具调用增量"""

    index: int | None = Field(None, description="工具调用索引")  # 改为可选，因为流式响应中可能缺失
    id: str | None = Field(None, description="工具调用ID")
    type: Literal["function"] | None = Field(None, description="调用类型")
    function: OpenAIToolCallFunction | None = Field(None, description="函数详情增量")


class OpenAIToolFunction(BaseModel):
    """OpenAI工具函数定义"""

    name: str = Field(description="函数名称")
    description: str | None = Field(None, description="函数描述")
    parameters: dict[str, Any] | None = Field(
        None, description="JSON Schema格式的函数参数"
    )


class OpenAITool(BaseModel):
    """OpenAI工具定义"""

    type: Literal["function"] = Field("function", description="工具类型")
    function: OpenAIToolFunction = Field(description="函数定义")


class OpenAIRequest(BaseModel):
    """OpenAI API请求模型"""

    model: str = Field(description="使用的模型ID，如gpt-4o或claude-3-5-sonnet")
    messages: list[OpenAIMessage] = Field(description="对话消息列表")
    max_tokens: int | None = Field(None, description="最大输出token数量")
    max_completion_tokens: int | None = Field(
        None, description="最大完成token数量（新格式）"
    )
    temperature: float | None = Field(None, ge=0.0, le=2.0, description="采样温度")
    top_p: float | None = Field(None, ge=0.0, le=1.0, description="top-p采样参数")
    top_k: int | None = Field(None, ge=0, description="top-k采样参数")
    stream: bool | None = Field(False, description="是否使用流式响应")
    stream_options: dict[str, Any] | None = Field(None, description="流式响应选项")
    stop: str | list[str] | None = Field(None, description="停止序列")
    frequency_penalty: float | None = Field(None, description="频率惩罚")
    presence_penalty: float | None = Field(None, description="存在惩罚")
    logprobs: bool | None = Field(False, description="是否返回log概率")
    top_logprobs: int | None = Field(
        None, description="返回的top log probability数量"
    )
    logit_bias: dict[str, int] | None = Field(None, description="logit偏差")
    n: int | None = Field(None, ge=1, le=128, description="生成的消息数量")
    seed: int | None = Field(None, description="随机种子")
    response_format: dict[str, Any] | None = Field(None, description="响应格式配置")
    tools: list[OpenAITool] | None = Field(None, description="可用工具定义")
    tool_choice: str | dict[str, Any] | None = Field(
        None, description="工具选择配置"
    )
    parallel_tool_calls: bool | None = Field(
        None, description="是否允许并行工具调用"
    )
    user: str | None = Field(None, description="用户信息")
    think: bool | None = Field(None, description="是否启用推理模型模式")


class OpenAIChoiceDelta(BaseModel):
    """流式响应增量内容"""

    role: str | None = Field(None, description="消息角色")
    content: str | None = Field(None, description="内容增量")
    reasoning_content: str | None = Field(None, description="推理内容增量")
    tool_calls: list[OpenAIDeltaToolCall] | None = Field(
        None, description="工具调用增量"
    )


class OpenAIChoice(BaseModel):
    """OpenAI响应选项"""

    index: int = Field(description="选项索引")
    message: OpenAIMessage | None = Field(None, description="完整消息响应")
    delta: OpenAIChoiceDelta | None = Field(None, description="流式增量内容")
    finish_reason: str | None = Field(
        None,
        description="完成原因: stop, length, content_filter, tool_calls, function_call",
    )
    logprobs: dict[str, Any] | None = Field(None, description="log概率信息")


class OpenAIUsage(BaseModel):
    """OpenAI使用统计"""

    prompt_tokens: int = Field(description="提示token数量")
    completion_tokens: int = Field(description="完成token数量")
    total_tokens: int = Field(description="总token数量")
    completion_tokens_details: dict[str, Any] | None = Field(
        None, description="完成token的详细信息"
    )
    prompt_tokens_details: dict[str, Any] | None = Field(
        None, description="提示token的详细信息"
    )


class OpenAICompletionUsage(BaseModel):
    """OpenAI完成使用统计"""

    completion_tokens: int = Field(description="完成token数量")
    prompt_tokens: int = Field(description="提示token数量")
    total_tokens: int = Field(description="总token数量")


class OpenAIResponse(BaseModel):
    """OpenAI API响应模型"""

    id: str = Field(description="响应唯一ID")
    object: Literal["chat.completion"] = Field(description="对象类型")
    created: int = Field(description="创建时间戳")
    model: str = Field(description="使用的模型ID")
    choices: list[OpenAIChoice] = Field(description="响应选项列表")
    usage: OpenAIUsage = Field(description="使用统计")
    system_fingerprint: str | None = Field(None, description="系统指纹")


class OpenAIStreamResponse(BaseModel):
    """OpenAI流式响应模型"""

    id: str = Field(description="响应唯一ID")
    object: Literal["chat.completion.chunk"] = Field(description="对象类型")
    created: int = Field(description="创建时间戳")
    model: str = Field(description="使用的模型ID")
    choices: list[OpenAIChoice] = Field(description="响应选项列表")
    usage: OpenAIUsage | None = Field(
        None, description="使用统计（仅在流式响应最后一块出现）"
    )
    system_fingerprint: str | None = Field(None, description="系统指纹")


class OpenAIErrorDetail(BaseModel):
    """OpenAI错误详情"""

    message: str = Field(description="错误消息")
    type: str = Field(description="错误类型")
    param: str | None = Field(None, description="相关参数")
    code: str | None = Field(None, description="错误代码")


class OpenAIErrorResponse(BaseModel):
    """OpenAI错误响应模型"""

    error: OpenAIErrorDetail = Field(description="错误详情")
