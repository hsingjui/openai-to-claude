import json
import os
from pathlib import Path

import aiofiles
from loguru import logger
from pydantic import BaseModel, Field, field_validator

# 全局配置缓存
_config_instance = None


async def get_config() -> "Config":
    """
    获取全局配置对象（带缓存的单例模式）
    """
    global _config_instance
    if _config_instance is None:
        try:
            _config_instance = await Config.from_file()
        except Exception:
            # 如果配置文件读取失败，创建默认配置
            _config_instance = Config(
                openai={
                    "api_key": "your-openai-api-key-here",
                    "base_url": "https://api.openai.com/v1",
                }
            )
    return _config_instance


async def reload_config(config_path: str | None = None) -> "Config":
    """重新加载全局配置对象

    Args:
        config_path: 配置文件路径，如果为None则使用默认路径

    Returns:
        Config: 重新加载的配置实例

    Raises:
        Exception: 配置加载失败时保持原配置不变
    """
    global _config_instance

    try:
        # 尝试加载新配置
        new_config = await Config.from_file(config_path)
        _config_instance = new_config
        logger.info("配置重载成功")
        return _config_instance
    except Exception as e:
        logger.error(f"配置重载失败，保持原配置: {e}")
        if _config_instance is None:
            # 如果没有原配置，则创建默认配置
            _config_instance = Config(
                openai={
                    "api_key": "your-openai-api-key-here",
                    "base_url": "https://api.openai.com/v1",
                }
            )
        return _config_instance


def get_config_file_path() -> str:
    """获取当前使用的配置文件路径

    Returns:
        str: 配置文件路径
    """
    import os

    return os.getenv("CONFIG_PATH", "config/settings.json")


class OpenAIConfig(BaseModel):
    """OpenAI API 配置"""

    api_key: str = Field(..., description="OpenAI API密钥")
    base_url: str = Field("https://api.openai.com/v1", description="OpenAI API基础URL")


class ServerConfig(BaseModel):
    """服务器配置"""

    host: str = Field("0.0.0.0", description="服务监听主机")
    port: int = Field(8000, gt=0, lt=65536, description="服务监听端口")


class LoggingConfig(BaseModel):
    """日志配置"""

    level: str = Field(
        "INFO", description="日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )

    def __init__(self, **data):
        """初始化时支持环境变量覆盖"""
        # 环境变量覆盖
        if "LOG_LEVEL" in os.environ:
            data["level"] = os.environ["LOG_LEVEL"]

        super().__init__(**data)

    @field_validator("level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """验证日志级别"""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"日志级别必须是以下之一: {', '.join(valid_levels)}")
        return v.upper()


class ModelConfig(BaseModel):
    """模型配置类

    定义不同使用场景下的模型选择
    """

    default: str = Field(
        description="默认通用模型", default="claude-3-5-sonnet-20241022"
    )
    small: str = Field(
        description="轻量级模型，用于简单任务", default="claude-3-5-haiku-20241022"
    )
    tool: str = Field(
        description="工具使用专用模型", default="claude-3-5-sonnet-20241022"
    )
    think: str = Field(
        description="深度思考模型，用于复杂推理任务",
        default="claude-3-7-sonnet-20250219",
    )
    longContext: str = Field(
        description="长上下文处理模型", default="claude-3-7-sonnet-20250219"
    )


class ParameterOverridesConfig(BaseModel):
    """参数覆盖配置类

    允许管理员在配置文件中设置模型参数的覆盖值。
    当设置了这些参数时，会覆盖客户端请求中的相应参数。
    """

    max_tokens: int | None = Field(
        None,
        gt=0,
        description="最大token数覆盖，设置后会覆盖客户端请求中的max_tokens参数",
    )
    temperature: float | None = Field(
        None, ge=0.0, le=2.0, description="温度参数覆盖，控制输出的随机程度"
    )
    top_p: float | None = Field(
        None, ge=0.0, le=1.0, description="top_p采样参数覆盖，控制候选词汇的概率阈值"
    )
    top_k: int | None = Field(
        None, ge=0, description="top_k采样参数覆盖，控制候选词汇的数量"
    )


class Config(BaseModel):
    """应用配置根类

    使用 JSON 配置文件加载配置。
    配置文件优先级：
    1. 命令行指定的配置路径
    2. 环境变量 CONFIG_PATH 指定的路径
    3. ./config/settings.json (默认)
    4. ./config/example.json (示例配置)
    5. 默认值
    """

    # 各模块配置
    openai: OpenAIConfig
    server: ServerConfig = ServerConfig()
    api_key: str = Field(..., description="/v1/messages接口的API密钥")
    logging: LoggingConfig = LoggingConfig()
    models: ModelConfig = ModelConfig()
    parameter_overrides: ParameterOverridesConfig = ParameterOverridesConfig()

    @classmethod
    async def from_file(cls, config_path: str | None = None) -> "Config":
        """
        从 JSON 配置文件加载配置
        Args:
            config_path: JSON配置文件路径，如果为None则使用默认路径

        Returns:
            Config: 配置实例

        Raises:
            FileNotFoundError: 配置文件不存在
            json.JSONDecodeError: JSON格式错误
            ValidationError: 配置数据验证错误
        """
        import os

        if config_path is None:
            # 优先使用环境变量指定的路径
            config_path = os.getenv("CONFIG_PATH", "config/settings.json")

        config_file = Path(config_path)

        if config_file.exists():
            try:
                async with aiofiles.open(config_file, encoding="utf-8") as f:
                    config_data = await f.read()
                    config_data = json.loads(config_data)
            except json.JSONDecodeError as e:
                print(f"❌ 配置文件格式错误: {e}")
                raise
        else:
            print(f"⚠️  配置文件 {config_file.absolute()} 不存在")
            print("📦 使用 config/example.json 作为模板")

            # 尝试使用 example 配置
            example_file = Path("config/example.json")
            if example_file.exists():
                try:
                    async with aiofiles.open(example_file, encoding="utf-8") as f:
                        config_data = await f.read()
                        config_data = json.loads(config_data)
                    # 创建 settings.json 作为实际配置文件
                    async with aiofiles.open(config_file, "w", encoding="utf-8") as f:
                        await f.write(
                            json.dumps(config_data, indent=2, ensure_ascii=False)
                        )
                    print(f"✅ 已从模板创建 {config_file}")

                except (json.JSONDecodeError, OSError) as e:
                    print(f"❌ 无法创建配置文件: {e}")
                    config_data = {}
            else:
                config_data = {}

        # 验证必填的 openai 配置
        if "openai" not in config_data:
            config_data["openai"] = {
                "api_key": "your-openai-api-key-here",
                "base_url": "https://api.openai.com/v1",
            }

        # 确保api_key存在（这是一个必填项）
        if "api_key" not in config_data:
            config_data["api_key"] = "your-proxy-api-key-here"

        return cls(**config_data)

    @classmethod
    def from_file_sync(cls, config_path: str | None = None) -> "Config":
        """
        从 JSON 配置文件加载配置
        Args:
            config_path: JSON配置文件路径，如果为None则使用默认路径

        Returns:
            Config: 配置实例

        Raises:
            FileNotFoundError: 配置文件不存在
            json.JSONDecodeError: JSON格式错误
            ValidationError: 配置数据验证错误
        """
        import os

        if config_path is None:
            # 优先使用环境变量指定的路径
            config_path = os.getenv("CONFIG_PATH", "config/settings.json")

        config_file = Path(config_path)

        if config_file.exists():
            try:
                with open(config_file, encoding="utf-8") as f:
                    config_data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"❌ 配置文件格式错误: {e}")
                raise
        else:
            print(f"⚠️  配置文件 {config_file.absolute()} 不存在")
            print("📦 使用 config/example.json 作为模板")

            # 尝试使用 example 配置
            example_file = Path("config/example.json")
            if example_file.exists():
                try:
                    with open(example_file, encoding="utf-8") as f:
                        config_data = json.load(f)
                    # 创建 settings.json 作为实际配置文件
                    with open(config_file, "w", encoding="utf-8") as f:
                        f.write(json.dumps(config_data, indent=2, ensure_ascii=False))
                    print(f"✅ 已从模板创建 {config_file}")
                except (json.JSONDecodeError, OSError) as e:
                    print(f"❌ 无法创建配置文件: {e}")
                    config_data = {}
            else:
                config_data = {}

        # 验证必填的 openai 配置
        if "openai" not in config_data:
            config_data["openai"] = {
                "api_key": "your-openai-api-key-here",
                "base_url": "https://api.openai.com/v1",
            }

        # 确保api_key存在（这是一个必填项）
        if "api_key" not in config_data:
            config_data["api_key"] = "your-proxy-api-key-here"

        return cls(**config_data)

    async def get_server_config(self) -> tuple[str, int]:
        """获取服务器配置 (host, port)

        Returns:
            tuple[str, int]: (host, port)
        """
        return self.server.host, self.server.port
