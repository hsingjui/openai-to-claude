"""标准化错误响应模型"""

from typing import Any

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    """错误详细信息"""

    code: str = Field(description="错误代码")
    message: str = Field(description="错误消息")
    param: str | None = Field(None, description="相关参数名称")
    type: str | None = Field(None, description="错误类型")
    details: dict[str, Any] | None = Field(None, description="额外错误详情")
    request_id: str | None = Field(None, description="请求ID用于追踪")


class StandardErrorResponse(BaseModel):
    """标准化错误响应模型"""

    type: str = Field("error", description="响应类型")
    error: ErrorDetail = Field(description="错误详情")


class ValidationErrorItem(BaseModel):
    """验证错误项"""

    loc: list[str] = Field(description="错误位置字段路径")
    msg: str = Field(description="错误消息")
    type: str = Field(description="错误类型")


class ValidationError(BaseModel):
    """验证错误详情"""

    code: str = Field("validation_error", description="错误代码")
    message: str = Field("请求参数验证失败", description="错误消息")
    details: list[ValidationErrorItem] = Field(..., description="验证错误列表")


class ValidationErrorResponse(BaseModel):
    """验证错误响应模型"""

    type: str = Field("validation_error", description="响应类型")
    error: ValidationError = Field(description="验证错误详情")


class UnauthorizedError(BaseModel):
    """未授权错误"""

    code: str = Field("unauthorized", description="错误代码")
    message: str = Field("无效的API密钥或未经授权的访问", description="错误消息")
    type: str = Field("authentication_error", description="错误类型")


class RateLimitError(BaseModel):
    """限流错误"""

    code: str = Field("rate_limit_exceeded", description="错误代码")
    message: str = Field("请求频率超出限制，请稍后重试", description="错误消息")
    type: str = Field("rate_limit_error", description="错误类型")
    retry_after: int | None = Field(None, description="重试等待时间（秒）")


class ServerError(BaseModel):
    """服务器内部错误"""

    code: str = Field("internal_server_error", description="错误代码")
    message: str = Field("服务器内部错误，请稍后重试", description="错误消息")
    type: str = Field("server_error", description="错误类型")
    request_id: str | None = Field(None, description="请求ID用于排查问题")


class TimeoutError(BaseModel):
    """超时错误"""

    code: str = Field("timeout", description="错误代码")
    message: str = Field("请求超时，请稍后重试", description="错误消息")
    type: str = Field("timeout_error", description="错误类型")
    timeout: int | None = Field(None, description="超时时间（秒）")


class NotFoundError(BaseModel):
    """资源未找到错误"""

    code: str = Field("not_found", description="错误代码")
    message: str = Field("请求的资源不存在", description="错误消息")
    type: str = Field("not_found_error", description="错误类型")


class BadRequestError(BaseModel):
    """错误请求"""

    code: str = Field("bad_request", description="错误代码")
    message: str = Field("请求格式错误或参数无效", description="错误消息")
    type: str = Field("invalid_request_error", description="错误类型")


class ServiceUnavailableError(BaseModel):
    """服务不可用"""

    code: str = Field("service_unavailable", description="错误代码")
    message: str = Field("服务暂时不可用，请稍后重试", description="错误消息")
    type: str = Field("server_error", description="错误类型")
    retry_after: int | None = Field(None, description="建议重试时间（秒）")


class ExternalServiceError(BaseModel):
    """外部服务错误"""

    code: str = Field("external_service_error", description="错误代码")
    message: str = Field("外部服务错误，请稍后重试", description="错误消息")
    type: str = Field("api_error", description="错误类型")
    service: str | None = Field(None, description="出错的外部服务名称")
    original_error: dict[str, Any] | None = Field(
        None, description="原始错误信息（生产环境可能会省略）"
    )


# 错误代码映射表
ERROR_CODE_MAPPING = {
    400: BadRequestError,
    401: UnauthorizedError,
    404: NotFoundError,
    422: ValidationError,
    429: RateLimitError,
    500: ServerError,
    502: ExternalServiceError,
    503: ServiceUnavailableError,
    504: TimeoutError,
}


def format_compact_traceback(error: Exception, max_lines: int = 10) -> str:
    """格式化紧凑的错误堆栈信息，只保留项目相关部分"""
    import traceback
    
    error_traceback = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    lines = error_traceback.split('\n')
    
    # 过滤出项目相关的行
    filtered_lines = []
    for line in lines:
        if ('openai-claude-code-proxy/src' in line or 
            line.strip().startswith('File "/Users') or
            any(keyword in line for keyword in ['Error:', 'Exception:', '    ']) or
            line.strip() == ''):
            filtered_lines.append(line)
    
    # 只保留最后max_lines行
    return '\n'.join(filtered_lines[-max_lines:]) if filtered_lines else str(error)

async def get_error_response(
    status_code: int,
    message: str | None = None,
    details: dict[str, Any] | None = None,
) -> StandardErrorResponse:
    """根据HTTP状态码获取对应的错误响应模型，返回Pydantic模型实例"""
    error_class = ERROR_CODE_MAPPING.get(status_code, ServerError)

    # 创建错误详情
    if error_class == ValidationError:
        # 验证错误需要特殊处理
        if details and "validation_errors" in details:
            validation_items = [
                {
                    "loc": error.get("loc", []),
                    "msg": error.get("msg", ""),
                    "type": error.get("type", "value_error"),
                }
                for error in details["validation_errors"]
            ]
            error_detail_data = {
                "code": "validation_error",
                "message": message or "请求参数验证失败",
                "details": validation_items,
            }
        else:
            error_detail_data = {
                "code": "validation_error",
                "message": message or "请求参数验证失败",
                "details": [],
            }
    else:
        # 其他错误类型
        error_detail_data = {
            "code": error_class.model_fields["code"].default,
            "message": message or error_class.model_fields["message"].default,
        }

        if details:
            # Add optional fields if present in details
            for key in [
                "param",
                "type",
                "retry_after",
                "request_id",
                "service",
                "original_error",
            ]:
                if key in details:
                    error_detail_data[key] = details[key]
            # 将其他details作为details字段
            other_details = {
                k: v
                for k, v in details.items()
                if k
                not in [
                    "param",
                    "type",
                    "retry_after",
                    "request_id",
                    "service",
                    "original_error",
                ]
            }
            if other_details:
                error_detail_data["details"] = other_details

    # 创建并返回StandardErrorResponse实例
    return StandardErrorResponse(error=ErrorDetail(**error_detail_data))
