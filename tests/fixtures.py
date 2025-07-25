"""Mock OpenAI server for end-to-end testing."""

from typing import Dict, Any, List, Optional, AsyncGenerator
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field
import time
import asyncio
from contextlib import asynccontextmanager


class MockMessage(BaseModel):
    role: str
    content: str


class MockChoice(BaseModel):
    index: int = 0
    message: MockMessage
    finish_reason: str = "stop"


class MockDelta(BaseModel):
    content: Optional[str] = None
    role: Optional[str] = None


class MockStreamChoice(BaseModel):
    index: int = 0
    delta: MockDelta
    finish_reason: Optional[str] = None


class MockUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class MockCompletionResponse(BaseModel):
    id: str = Field(default="mock-completion-id")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[MockChoice]
    usage: MockUsage


class MockStreamResponse(BaseModel):
    id: str = Field(default="mock-completion-id")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[MockStreamChoice]


class MockChatCompletionRequest(BaseModel):
    model: str
    messages: List[MockMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: bool = False


mock_responses: Dict[str, Dict[str, Any]] = {}
error_trigger: str = ""
delay_ms: int = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage mock server lifecycle."""
    print("Mock OpenAI server starting...")
    yield
    print("Mock OpenAI server shutting down...")


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def add_delay(request, call_next):
    """Add configurable delay to all responses."""
    if delay_ms > 0:
        await asyncio.sleep(delay_ms / 1000)
    response = await call_next(request)
    return response


@app.post("/v1/chat/completions")
async def mock_chat_completions(
    request: MockChatCompletionRequest,
    authorization: str = Header(...),
):
    """Mock the OpenAI chat completions endpoint."""
    
    if not authorization.startswith("Bearer mock-key"):
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    if error_trigger == "invalid_model":
        raise HTTPException(status_code=400, detail="Invalid model")
    
    if error_trigger == "rate_limit":
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    if error_trigger == "server_error":
        raise HTTPException(status_code=500, detail="Internal server error")
    
    model = request.model
    if model == "invalid-model":
        raise HTTPException(status_code=404, detail="Model not found")
    
    last_message = request.messages[-1]
    
    if request.stream:
        return mock_stream_response(model, last_message.content or "")
    
    return mock_non_stream_response(model, last_message.content or "")


def mock_non_stream_response(model: str, content: str) -> MockCompletionResponse:
    """Generate a non-streaming chat completion response."""
    prompt_tokens = len(content.split())
    completion_tokens = 10  # Fixed for testing
    
    return MockCompletionResponse(
        model=model,
        choices=[
            MockChoice(
                message=MockMessage(
                    role="assistant",
                    content=f"Mock response for: {content}"
                ),
                finish_reason="stop"
            )
        ],
        usage=MockUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        )
    )


async def mock_stream_response(model: str, content: str):
    """Generate a streaming chat completion response."""
    words = ["This", "is", "a", "mock", "response", "for:", content[:10], "..."]
    
    # First chunk with role
    yield MockStreamResponse(
        model=model,
        choices=[
            MockStreamChoice(
                delta=MockDelta(role="assistant", content=""),
                finish_reason=None
            )
        ]
    ).model_dump_json() + "\n"
    
    # Stream content chunks
    for word in words:
        chunk = MockStreamResponse(
            model=model,
            choices=[
                MockStreamChoice(
                    delta=MockDelta(content=f"{word}"),
                    finish_reason=None
                )
            ]
        )
        yield chunk.model_dump_json() + "\n"
        await asyncio.sleep(0.05)  # Simulate streaming delay
    
    # Final chunk
    chunk = MockStreamResponse(
        model=model,
        choices=[
            MockStreamChoice(
                delta=MockDelta(content=""),
                finish_reason="stop"
            )
        ]
    )
    yield chunk.model_dump_json() + "\n"


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/mock/config")
async def get_config():
    """Get current mock configuration."""
    return {
        "error_trigger": error_trigger,
        "delay_ms": delay_ms,
        "responses": list(mock_responses.keys())
    }


@app.post("/mock/configure")
async def configure_mock(config: Dict[str, Any]):
    """Configure mock server behavior."""
    global error_trigger, delay_ms, mock_responses
    
    if "error_trigger" in config:
        error_trigger = config["error_trigger"]
    if "delay_ms" in config:
        delay_ms = config["delay_ms"]
    if "responses" in config:
        mock_responses.update(config["responses"])
    
    return {"message": "Mock server configured"}


@app.delete("/mock/reset")
async def reset_mock():
    """Reset mock server to default state."""
    global error_trigger, delay_ms, mock_responses
    error_trigger = ""
    delay_ms = 0
    mock_responses.clear()
    return {"message": "Mock server reset"}