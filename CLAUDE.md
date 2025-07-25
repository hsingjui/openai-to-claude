# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个高性能的 RESTful API 代理服务，将 OpenAI API 格式转换为 Anthropic Claude API 兼容格式，允许开发者使用现有的 OpenAI 客户端代码来调用 Anthropic Claude 模型。

主要功能：
- ✅ 请求/响应格式转换
- ✅ 流式响应支持
- ✅ 错误处理映射
- ✅ 配置管理
- ✅ 结构化日志
- ✅ 健康检查端点

## 项目结构

```
openai-to-claude/
├── src/
│   ├── config/          # 配置管理
│   ├── models/          # Pydantic 数据模型
│   ├── converters/      # 数据格式转换器
│   ├── clients/         # HTTP 客户端
│   ├── api/             # API 端点和中间件
│   └── common/          # 公共工具日志、token计数等）
├── tests/               # 测试套件
├── config/              # 配置文件模板
└── pyproject.toml       # 项目依赖和配置
```

## 核心架构

### 数据流向
1. 客户端发送 Anthropic 格式的请求到 `/v1/messages` 端点
2. `MessagesHandler` 接收请求并使用 `AnthropicToOpenAIConverter` 转换为 OpenAI 格式
3. `OpenAIServiceClient` 将请求发送到实际的 OpenAI API 服务
4. 收到 OpenAI 响应后，使用 `OpenAIToAnthropicConverter` 转换回 Anthropic 格式
5. 返回给客户端

### 主要组件
- **src/api/handlers.py**: 核心消息处理逻辑
- **src/core/converters/**: 请求/响应格式转换器
- **src/core/clients/openai_client.py**: OpenAI API 客户端
- **src/models/**: 数据模型定义 (Anthropic 和 OpenAI)
- **src/config/settings.py**: 配置管理
- **src/common/**: 公共工具（日志、token 计数等）

## 开发命令

### 安装依赖
```bash
uv sync
```

### 运行服务
```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### 运行测试
```bash
# 运行所有测试
pytest

# 运行特定测试文
pytest tests/unit/test_models.py

# 运行集成测试
pytest tests/integration

# 带覆盖率报告
pytest --cov=src --cov-report=html
```

### 代码质量检查
```bash
# 运行 linting
ruff check .

# 自动修复代码风格
ruff check . --fix

# 代码格式化
black .

# 类型检查
mypy src
```

## 配置说明

### 环境变量
- `LOG_LEVEL`: 日志级别 (默认: INFO)
- `CONFIG_PATH`: 配置文件路径 (默认: config/settings.json)

### 配置文件 (config/settings.json)
```json
{
  "openai": {
    "api_key": "your-openai-api-key-here",
    "base_url": "https://api.openai.com/v1"
  },
  "server": {
    "host": "0.0.0.0",
    "port": 8000
  },
  "api_key": "your-proxy-api-key-here",
  "logging": {
    "level": "INFO"
  },
  "models": {
    "default": "claude-3-5-sonnet-20241022",
    "small": "claude-3-5-haiku-20241022",
    "tool": "claude-3-5-sonnet-20241022",
    "think": "claude-3-7-sonnet-20250219",
    "longContext": "claude-3-7-sonnet-20250219"
  },
  "parameter_overrides": {
    "max_tokens": null,
    "temperature": null,
    "top_p": null,
    "top_k": null
  }
}
```

## API 端点

- `POST /v1/messages` - Anthropic 消息 API（兼容 OpenAI 聊天 API）
- `GET /health` - 健康检查端点