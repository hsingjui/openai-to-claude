from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends

from src.config.settings import Config
from src.core.clients.openai_client import OpenAIServiceClient

router = APIRouter()


async def get_openai_client() -> OpenAIServiceClient:
    """获取OpenAI客户端实例"""
    config = await Config.from_file()
    return OpenAIServiceClient(
        api_key=config.openai.api_key,
        base_url=config.openai.base_url,
    )


@router.get("/health", tags=["health"])
async def health_check(
    client: OpenAIServiceClient = Depends(get_openai_client),
) -> dict[str, Any]:
    """健康检查端点 - 验证OpenAI连通性"""

    health_status = {
        "status": "healthy",
        "service": "openai-to-claude",
        "timestamp": datetime.now().isoformat(),
        "checks": {},
    }

    try:
        # 检查OpenAI服务可用性
        openai_health = await client.health_check()
        health_status["checks"]["openai"] = openai_health

        # 如果任何一个检查失败，状态设为降级
        if not all(openai_health.values()):
            health_status["status"] = "degraded"

    except Exception as e:
        # 如果无法创建客户端或者检查抛出异常
        health_status["status"] = "unhealthy"
        health_status["checks"]["openai"] = {
            "openai_service": False,
            "api_accessible": False,
            "error": str(e),
        }

    return health_status
