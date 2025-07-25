"""
配置管理模块

提供应用程序配置的加载、验证和热重载功能。

主要功能:
- 配置文件加载和验证
- 配置热重载监听
- 配置模型定义
- 配置实例管理

使用示例:
    from src.config import get_config, reload_config

    config = get_config()
    print(config.server.host)
"""

# 导入配置相关功能
from .settings import (
    Config,
    LoggingConfig,
    ModelConfig,
    OpenAIConfig,
    ParameterOverridesConfig,
    ServerConfig,
    get_config,
    get_config_file_path,
    reload_config,
)
from .watcher import ConfigFileHandler, ConfigWatcher

__all__ = [
    # 配置管理函数
    "get_config",
    "reload_config",
    "get_config_file_path",
    # 配置模型
    "Config",
    "OpenAIConfig",
    "ServerConfig",
    "LoggingConfig",
    "ModelConfig",
    "ParameterOverridesConfig",
    # 配置监听器
    "ConfigWatcher",
    "ConfigFileHandler",
]
