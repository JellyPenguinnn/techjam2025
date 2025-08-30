import pytest
from httpx import AsyncClient
from httpx import ASGITransport
from backend.app.main import app


@pytest.mark.asyncio
async def test_health():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.get("/health")
        assert res.status_code == 200
        assert res.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_redact_email_and_phone():
    text = "Contact me at alice@example.com or +1-415-555-0123."
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.post("/redact", json={"text": text})
        assert res.status_code == 200
        data = res.json()
        assert "***[EMAIL#1]***" in data["redacted_text"]
        assert "***[PHONE#1]***" in data["redacted_text"]
        assert data["map"]["***[EMAIL#1]***"] == "alice@example.com"
