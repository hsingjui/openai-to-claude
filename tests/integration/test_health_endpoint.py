"""
Health check endpoint integration tests.
"""
import pytest
from httpx import AsyncClient
from fastapi import FastAPI
from unittest.mock import AsyncMock, patch

from src.main import app


class TestHealthEndpoint:
    """Test cases for the health check endpoint."""

    @pytest.fixture
    def test_app(self) -> FastAPI:
        """Create test FastAPI app."""
        return app

    @pytest.mark.asyncio
    async def test_health_check_success(self, test_app):
        """Test successful health check with OpenAI service available."""
        
        mock_health_result = {
            "openai_service": True,
            "api_accessible": True,
            "last_check": True
        }
        
        with patch(
            "src.clients.openai_client.OpenAIServiceClient.health_check", 
            new_callable=AsyncMock
        ) as mock_health_check:
            mock_health_check.return_value = mock_health_result
            
            async with AsyncClient(app=test_app, base_url="http://test") as ac:
                response = await ac.get("/health")
                
            assert response.status_code == 200
            data = response.json()
            
            assert data["status"] == "healthy"
            assert data["service"] == "openai-to-claude"
            assert "timestamp" in data
            assert data["checks"]["openai"] == mock_health_result

    @pytest.mark.asyncio
    async def test_health_check_degraded(self, test_app):
        """Test health check when OpenAI service is degraded."""
        
        mock_health_result = {
            "openai_service": False,
            "api_accessible": True,
            "last_check": True
        }
        
        with patch(
            "src.clients.openai_client.OpenAIServiceClient.health_check", 
            new_callable=AsyncMock
        ) as mock_health_check:
            mock_health_check.return_value = mock_health_result
            
            async with AsyncClient(app=test_app, base_url="http://test") as ac:
                response = await ac.get("/health")
                
            assert response.status_code == 200
            data = response.json()
            
            assert data["status"] == "degraded"
            assert data["service"] == "openai-to-claude"
            assert data["checks"]["openai"] == mock_health_result

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, test_app):
        """Test health check when OpenAI service is unavailable."""
        
        with patch(
            "src.clients.openai_client.OpenAIServiceClient.health_check", 
            new_callable=AsyncMock
        ) as mock_health_check:
            mock_health_check.side_effect = Exception("Connection failed")
            
            async with AsyncClient(app=test_app, base_url="http://test") as ac:
                response = await ac.get("/health")
                
            assert response.status_code == 200
            data = response.json()
            
            assert data["status"] == "unhealthy"
            assert data["service"] == "openai-to-claude"
            assert data["checks"]["openai"]["openai_service"] is False
            assert data["checks"]["openai"]["api_accessible"] is False
            assert "error" in data["checks"]["openai"]

    @pytest.mark.asyncio
    async def test_health_check_response_structure(self, test_app):
        """Test health check response structure and required fields."""
        
        mock_health_result = {
            "openai_service": True,
            "api_accessible": True,
            "last_check": True
        }
        
        with patch(
            "src.clients.openai_client.OpenAIServiceClient.health_check", 
            new_callable=AsyncMock
        ) as mock_health_check:
            mock_health_check.return_value = mock_health_result
            
            async with AsyncClient(app=test_app, base_url="http://test") as ac:
                response = await ac.get("/health")
                
            assert response.status_code == 200
            assert response.headers["content-type"] == "application/json"
            
            data = response.json()
            
            # Check required top-level fields
            assert "status" in data
            assert "service" in data
            assert "timestamp" in data
            assert "checks" in data
            
            # Check status values
            assert data["status"] in ["healthy", "degraded", "unhealthy"]
            assert data["service"] == "openai-to-claude"
            
            # Check timestamp format
            from datetime import datetime
            datetime.fromisoformat(data["timestamp"].replace('Z', '+00:00'))
            
            # Check checks structure
            assert isinstance(data["checks"], dict)
            assert "openai" in data["checks"]