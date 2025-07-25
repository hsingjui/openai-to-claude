from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.handlers import router as messages_router
from src.api.middleware.auth import APIKeyMiddleware
from src.api.middleware.timing import setup_middlewares
from src.api.routes import router as health_router

# 启动时同步加载配置（模块级别，应用启动时执行）
from .common.logging import get_request_id_from_request
from src.config.settings import Config

config = Config.from_file_sync()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    from src.api.handlers import MessagesHandler
    from src.common.logging import configure_logging
    from src.config.settings import get_config_file_path, reload_config
    from src.config.watcher import ConfigWatcher

    host, port = await config.get_server_config()

    # 配置Loguru日志
    configure_logging(config.logging)

    # 创建配置化的消息处理器并缓存到应用中
    app.state.messages_handler = await MessagesHandler.create(config)

    # 配置重载回调函数
    async def on_config_reload():
        """配置重载时的回调函数"""
        try:
            # 重新加载配置
            new_config = await reload_config()

            # 重新配置日志
            configure_logging(new_config.logging)

            # 重新创建消息处理器
            app.state.messages_handler = await MessagesHandler.create(new_config)

            logger.info("配置热重载完成，服务已更新")
        except Exception as e:
            logger.error(f"配置热重载失败: {e}")

    # 启动配置文件监听器
    config_watcher = ConfigWatcher(get_config_file_path())
    config_watcher.add_reload_callback(on_config_reload)
    await config_watcher.start_watching()

    # 缓存监听器到应用状态
    app.state.config_watcher = config_watcher

    logger.info(
        f"启动 OpenAI To Claude 服务器 - Host: {host}, Port: {port}, LogLevel: {config.logging.level}"
    )
    logger.info(f"配置文件监听已启用: {get_config_file_path()}")

    yield

    # 关闭时的清理工作
    logger.info("正在停止配置文件监听...")
    if hasattr(app.state, "config_watcher"):
        app.state.config_watcher.stop_watching()
    logger.info("服务器已停止")


app = FastAPI(
    title="OpenAI To Claude Server",
    version="0.1.0",
    description="A server to convert OpenAI API calls to Claude format.",
    lifespan=lifespan,
)

# 设置CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境建议指定具体域名
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有HTTP方法
    allow_headers=["*"],  # 允许所有请求头
)

# 设置其他中间件
setup_middlewares(app)
app.add_middleware(APIKeyMiddleware, api_key=config.api_key)

app.include_router(health_router)
app.include_router(messages_router)


@app.get("/")
async def root():
    return {"message": "Welcome to the OpenAI To Claude Server"}


from fastapi.responses import JSONResponse
from src.models.errors import get_error_response


# 全局异常处理程序
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常处理，防止Internal Server Error直接返回给客户端"""
    from src.common.logging import (
        get_logger_with_request_id,
        get_request_id_from_request,
    )

    request_id = get_request_id_from_request(request)
    bound_logger = get_logger_with_request_id(request_id)

    bound_logger.exception(
        "捕获未处理的服务器异常",
        error_type=type(exc).__name__,
        error_message=str(exc),
        request_path=str(request.url),
        request_method=request.method,
    )

    # 使用标准错误响应格式返回客户端
    error_response = await get_error_response(500, message="服务器内部错误，请稍后重试")
    return JSONResponse(status_code=500, content=error_response.model_dump())


# Pydantic验证错误处理
@app.exception_handler(422)
async def validation_exception_handler(request, exc):
    """处理Pydantic验证错误"""
    from src.common.logging import get_logger_with_request_id

    request_id = get_request_id_from_request(request)
    bound_logger = get_logger_with_request_id(request_id)

    bound_logger.warning("请求验证失败")
    return JSONResponse(status_code=422, content=exc.detail)


# 404错误处理
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """处理404错误"""
    error_response = await get_error_response(404, message="请求的资源不存在")

    return JSONResponse(status_code=404, content=error_response.model_dump())
