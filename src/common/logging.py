"""Loguru日志配置"""

import sys
import uuid
from pathlib import Path
import traceback

from loguru import logger


def format_exception_truncated(record):
    """格式化异常信息，截取前100个字符"""
    if record["exception"]:
        exc_text = "".join(traceback.format_exception(*record["exception"]))
        if len(exc_text) > 1000:
            return exc_text[:1000] + "..."
        return exc_text
    return ""


def configure_logging(log_config) -> None:
    """配置Loguru日志系统

    Args:
        log_config: 日志配置对象
    """
    # 移除默认的handler
    logger.remove()

    # 使用相对路径而不是绝对路径
    log_path = Path("logs/app.log")
    
    # 确保日志目录存在，并设置正确的权限
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # 设置目录权限为755，确保当前用户可写
    log_path.parent.chmod(0o755)
    
    # 如果日志文件已存在，确保其可写
    if log_path.exists():
        log_path.chmod(0o644)

    # 控制台日志格式（包含请求ID）
    console_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{extra[request_id]}</cyan> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"

    # 文件日志格式（包含请求ID和截取的异常信息）
    def file_format_with_truncated_exception(record):
        exc_info = format_exception_truncated(record)
        return f"{record['time'].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} | {record['level'].name:<8} | {record['extra'].get('request_id', '---')} | {record['name']}:{record['line']} | {record['message']} | {exc_info}"

    # 配置控制台日志
    logger.add(
        sys.stdout,
        format=console_format,
        level=log_config.level,
        colorize=True,
        filter=lambda record: record["extra"].setdefault("request_id", "---"),
    )

    # 配置文件日志（包含截取的异常堆栈）
    logger.add(
        str(log_path),
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {extra[request_id]} | {name}:{line} | {message}",
        level=log_config.level,
        rotation="10 MB",
        retention="1 day",
        encoding="utf-8",
        backtrace=True,  # 启用回溯信息
        diagnose=True,  # 启用诊断信息
        filter=lambda record: record["extra"].setdefault("request_id", "---"),
    )

    # 配置全局异常处理
    def exception_handler(exc_type, exc_value, exc_traceback):
        """全局异常处理器"""
        if issubclass(exc_type, KeyboardInterrupt):
            # 允许KeyboardInterrupt正常退出
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logger.opt(exception=(exc_type, exc_value, exc_traceback)).critical(
            "未捕获的异常"
        )

    # 设置全局异常处理器
    sys.excepthook = exception_handler


class RequestLogger:
    """请求日志处理器"""

    async def log_response(
        self, status_code: int, response_time: float, request_id: str = None
    ):
        """记录响应结束"""
        bound_logger = get_logger_with_request_id(request_id)

        response_time_ms = round(response_time * 1000, 2)
        bound_logger.info(
            f"请求完成 - Status: {status_code}, Time: {response_time_ms}ms"
        )

    async def log_error(
        self, error: Exception, context: dict = None, request_id: str = None
    ):
        """记录错误情况"""
        bound_logger = get_logger_with_request_id(request_id)

        error_type = type(error).__name__
        error_message = str(error)
        context_str = f", Context: {context}" if context else ""

        # 使用loguru的exception方法记录完整的堆栈跟踪
        bound_logger.exception(
            f"请求处理错误 - Type: {error_type}, Message: {error_message}{context_str}"
        )


# 全局logger实例
request_logger = RequestLogger()


async def generate_request_id() -> str:
    """生成唯一的请求ID

    Returns:
        str: 格式为 req_xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx 的请求ID
    """
    return f"req_{uuid.uuid4()}"


async def should_enable_request_id() -> bool:
    """检查是否应该启用请求ID（始终启用）

    Returns:
        bool: 始终返回True，请求ID功能默认启用
    """
    return True


async def get_request_id_header_name() -> str:
    """获取请求ID响应头名称

    Returns:
        str: 固定返回 "X-Request-ID"
    """
    return "X-Request-ID"


def get_request_id_from_request(request) -> str | None:
    """从请求对象中安全地获取请求ID

    Args:
        request: FastAPI Request对象

    Returns:
        str | None: 请求ID，如果不存在则返回None
    """
    try:
        return getattr(request.state, "request_id", None)
    except AttributeError:
        return None


async def log_exception(message: str = "发生异常", **kwargs):
    """记录异常的便捷函数

    使用示例:
        try:
            # 一些可能出错的代码
            pass
        except Exception as e:
            log_exception("处理请求时发生错误", request_id="123", user_id="456")

    Args:
        message: 异常描述信息
        **kwargs: 额外的上下文信息
    """
    kwargs_str = ", ".join([f"{k}: {v}" for k, v in kwargs.items()]) if kwargs else ""
    full_message = f"{message} - {kwargs_str}" if kwargs_str else message
    logger.exception(full_message)


def get_logger_with_request_id(request_id: str = None):
    """获取绑定了请求ID的日志器实例

    Args:
        request_id: 请求ID，如果为None则使用默认值

    Returns:
        绑定了请求ID的logger实例
    """
    if request_id:
        return logger.bind(request_id=request_id)
    else:
        return logger.bind(request_id="---")
