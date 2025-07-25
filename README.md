# OpenAI-to-Claude API Proxy Service

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

High-performance proxy service that converts OpenAI API to Anthropic API compatible format. Allows developers to seamlessly call OpenAI models using existing Anthropic client code.

[中文版本](README_zh.md)

## 🌟 Core Features

- ✅ **Seamless Compatibility**: Call OpenAI models using standard Anthropic clients
- ✅ **Full Functionality**: Supports text, tool calls, streaming responses, and more
- ✅ **Intelligent Routing**: Automatically selects the most suitable OpenAI model based on request content
- ✅ **Hot Reload**: Automatically reloads configuration file changes without restarting the service
- ✅ **Structured Logging**: Detailed request/response logs for debugging and monitoring
- ✅ **Error Mapping**: Comprehensive error handling and mapping mechanisms

## 🚀 Quick Start

### Requirements

- Python 3.11+
- uv (recommended package manager)

### Install Dependencies

```bash
# Install dependencies using uv (recommended)
uv sync
```

### Configuration

1. Copy the example configuration file:
```bash
cp config/example.json config/settings.json
```

2. Edit `config/settings.json`:
```json
{
  "openai": {
    "api_key": "your-openai-api-key-here",  // Replace with your OpenAI API key
    "base_url": "https://api.openai.com/v1"  // OpenAI API address
  },
  "api_key": "your-proxy-api-key-here",  // API key for the proxy service
  // Other configurations...
}
```

### Start the Service

```bash
# Development mode
uv run main.py --config config/settings.json

# Production mode
uv run main.py
```

### Start with Docker

```bash
# Build and start the service
docker-compose up --build

# Run in background
docker-compose up --build -d

# Stop the service
docker-compose down
```

The service will start at `http://localhost:8000`.

## 🛠️ Usage

### Claude Code Usage

This project can be used with [Claude Code](https://claude.ai/code) for development and testing. To configure Claude Code to work with this proxy service, create a `.claude/settings.json` file with the following configuration:

```json
{
    "env": {
        "ANTHROPIC_API_KEY": "your-api-key",
        "ANTHROPIC_BASE_URL": "http://127.0.0.1:8000",
        "DISABLE_TELEMETRY": "1",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"
    },
    "apiKeyHelper": "echo 'your-api-key'",
    "permissions": {
        "allow": [],
        "deny": []
    }
}
```

Configuration Notes:
- Replace `ANTHROPIC_API_KEY` with your API key, in `config/settings.json`
- Replace `ANTHROPIC_BASE_URL` with the actual URL where this proxy service is running
- The `apiKeyHelper` with your API key, in `config/settings.json`

### Using Anthropic Python Client

```python
from anthropic import Anthropic

# Initialize client pointing to the proxy service
client = Anthropic(
    base_url="http://localhost:8000/v1",
    api_key="your-proxy-api-key-here"  # Use the api_key from the configuration file
)

# Send a message request
response = client.messages.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "Hello, GPT!"}
    ],
    max_tokens=1024
)

print(response.content[0].text)
```

### Streaming Response

```python
# Streaming response
stream = client.messages.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "Tell me a story about AI"}
    ],
    max_tokens=1024,
    stream=True
)

for chunk in stream:
    if chunk.type == "content_block_delta":
        print(chunk.delta.text, end="", flush=True)
```

### Tool Calls

```python
# Tool calls
tools = [
    {
        "name": "get_current_weather",
        "description": "Get the current weather for a specified city",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name"
                }
            },
            "required": ["city"]
        }
    }
]

response = client.messages.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "What's the weather like in Beijing now?"}
    ],
    tools=tools,
    tool_choice={"type": "auto"}
)
```

## 📁 Project Structure

```
openai-to-claude/
├── src/
│   ├── api/             # API endpoints and middleware
│   ├── config/          # Configuration management
│   ├── core/            # Core business logic
│   │   ├── clients/     # HTTP clients
│   │   └── converters/  # Data format converters
│   ├── models/          # Pydantic data models
│   └── common/          # Common utilities (logging, token counting, etc.)
├── config/              # Configuration files
├── tests/               # Test suite
├── CLAUDE.md           # Claude Code project guide
└── pyproject.toml      # Project dependencies and configuration
```

## 🔧 Configuration

### Environment Variables

- `CONFIG_PATH`: Configuration file path (default: `config/settings.json`)
- `LOG_LEVEL`: Log level (default: `INFO`)

### Configuration File (`config/settings.json`)

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

#### Configuration Items

- **openai**: OpenAI API configuration
  - `api_key`: OpenAI API key for accessing OpenAI services
  - `base_url`: OpenAI API base URL, default is `https://api.openai.com/v1`

- **server**: Server configuration
  - `host`: Service listening host address, default is `0.0.0.0` (listen on all network interfaces)
  - `port`: Service listening port, default is `8000`

- **api_key**: API key for the proxy service, used to verify requests to the `/v1/messages` endpoint

- **logging**: Logging configuration
  - `level`: Log level, options are `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`, default is `INFO`

- **models**: Model configuration, defines model selection for different usage scenarios
  - `default`: Default general model for general requests
  - `small`: Lightweight model for simple tasks
  - `think`: Deep thinking model for complex reasoning tasks
  - `longContext`: Long context processing model for handling long text

- **parameter_overrides**: Parameter override configuration, allows administrators to set model parameter override values in the configuration file
  - `max_tokens`: Maximum token count override, when set, will override the max_tokens parameter in client requests
  - `temperature`: Temperature parameter override, controls the randomness of output, range 0.0-2.0
  - `top_p`: top_p sampling parameter override, controls the probability threshold of candidate words, range 0.0-1.0
  - `top_k`: top_k sampling parameter override, controls the number of candidate words, range >=0

## 🧪 Testing

```bash
# Run all tests
pytest

# Run unit tests
pytest tests/unit

# Run integration tests
pytest tests/integration

# Generate coverage report
pytest --cov=src --cov-report=html
```

## 📊 API Endpoints

- `POST /v1/messages` - Anthropic Messages API
- `GET /health` - Health check endpoint
- `GET /` - Welcome page

## 🛡️ Security

- API key authentication
- Request rate limiting (planned)
- Input validation and sanitization
- Structured logging

## 📈 Performance Monitoring

- Request/response time monitoring
- Memory usage tracking
- Error rate statistics

## 🤝 Contributing

Issues and Pull Requests are welcome!

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [claude-code-router](https://github.com/musistudio/claude-code-router) - Very good project, many places in this project have referenced this project
- [FastAPI](https://fastapi.tiangolo.com/) - Modern high-performance web framework
- [Anthropic](https://www.anthropic.com/) - Claude AI models
- [OpenAI](https://openai.com/) - OpenAI API specification

## 🤖 Claude Code Usage

This project can be used with [Claude Code](https://claude.ai/code) for development and testing. To configure Claude Code to work with this proxy service, create a `.claude/settings.json` file with the following configuration:

### Example Configuration

```json
{
    "env": {
        "ANTHROPIC_API_KEY": "sk-chen0v0...",
        "ANTHROPIC_BASE_URL": "http://127.0.0.1:8100",
        "DISABLE_TELEMETRY": "1",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"
    },
    "apiKeyHelper": "echo 'sk-chen0v0...'",
    "permissions": {
        "allow": [],
        "deny": []
    }
}
```

### Configuration Notes

- Replace `ANTHROPIC_API_KEY` with your actual Anthropic API key
- Replace `ANTHROPIC_BASE_URL` with the actual URL where this proxy service is running
- The `apiKeyHelper` field should also be updated with your actual API key