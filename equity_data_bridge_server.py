"""
Equity Market Data Bridge — real historical price bars and quotes for
Equity Screener's cointegrated-pairs screening and Statistical
Arbitrage Bot's OU fit/re-check (Dynamic Grok Bot Desk)
"""

import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import httpx
from starlette.applications import Starlette
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

ALPACA_API_KEY_ID = os.environ["ALPACA_API_KEY_ID"]
ALPACA_API_SECRET_KEY = os.environ["ALPACA_API_SECRET_KEY"]
PORT = int(os.environ.get("PORT", "10000"))

ALPACA_DATA_BASE = "https://data.alpaca.markets/v2"
ALPACA_HEADERS = {
    "APCA-API-KEY-ID": ALPACA_API_KEY_ID,
    "APCA-API-SECRET-KEY": ALPACA_API_SECRET_KEY,
}

mcp = FastMCP(
    "equity-data-bridge",
    stateless_http=True,
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=False,
    ),
)
mcp.settings.streamable_http_path = "/"


@mcp.tool()
def get_historical_bars(symbol: str, timeframe: str, start: str, end: str) -> dict:
    """
    Real historical OHLCV price bars for one stock symbol, from Alpaca's
    real market data — never invented, never estimated. Cointegration
    testing, hedge-ratio fitting, and OU parameter estimation are the
    calling bot's own job, not done here.

    symbol: ticker, e.g. "AAPL"
    timeframe: one of "1Min", "5Min", "15Min", "1Hour", "1Day"
    start: ISO date or datetime, e.g. "2026-01-01" or "2026-01-01T00:00:00Z"
    end: ISO date or datetime, same format as start
    """
    url = f"{ALPACA_DATA_BASE}/stocks/{symbol}/bars"
    params = {
        "timeframe": timeframe,
        "start": start,
        "end": end,
        "limit": 10000,
        "adjustment": "raw",
        "feed": "iex",
    }
    with httpx.Client(timeout=30) as client:
        resp = client.get(url, headers=ALPACA_HEADERS, params=params)
    if resp.status_code != 200:
        return {"status": "error", "http_status": resp.status_code, "detail": resp.text}
    data = resp.json()
    return {
        "status": "ok",
        "symbol": symbol,
        "timeframe": timeframe,
        "bar_count": len(data.get("bars", [])),
        "bars": data.get("bars", []),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


@mcp.tool()
def get_latest_quote(symbol: str) -> dict:
    """The current real-time bid/ask quote for a symbol, from Alpaca's IEX feed."""
    url = f"{ALPACA_DATA_BASE}/stocks/{symbol}/quotes/latest"
    with httpx.Client(timeout=15) as client:
        resp = client.get(url, headers=ALPACA_HEADERS, params={"feed": "iex"})
    if resp.status_code != 200:
        return {"status": "error", "http_status": resp.status_code, "detail": resp.text}
    return {"status": "ok", "symbol": symbol, **resp.json()}


@asynccontextmanager
async def lifespan(app):
    async with mcp.session_manager.run():
        yield


app = Starlette(routes=[], lifespan=lifespan)
app.mount("/mcp-server", mcp.streamable_http_app())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
