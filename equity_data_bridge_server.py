"""
Equity Market Data Bridge — real historical price bars and quotes for
Equity Screener's cointegrated-pairs screening and Statistical
Arbitrage Bot's OU fit/re-check (Dynamic Grok Bot Desk)
"""

import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import io
import zipfile

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
    real market data — never invented, never estimated. This is raw
    material only — cointegration testing, hedge-ratio fitting, and OU
    parameter estimation are the calling bot's own job, not done here.

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


FAMA_FRENCH_5_DAILY_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)


@mcp.tool()
def get_fama_french_5factors(start_date: str, end_date: str) -> dict:
    """
    Real daily Fama-French five-factor returns (Mkt-RF, SMB, HML, RMW,
    CMA, RF) from Kenneth French's own public Data Library at Dartmouth
    — the canonical academic source, not invented or estimated. Free,
    no account, no API key.

    start_date / end_date: YYYYMMDD strings, e.g. "20260101", "20260901"

    All values are returned as DECIMAL fractions (already divided by
    100) for consistency with how this desk represents returns
    elsewhere — the source file itself publishes them in percent.

    This is a periodically-updated research file, not a live feed — it
    typically lags real time by some days as Kenneth French's team
    updates it. Do not treat it as real-time data.
    """
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        resp = client.get(FAMA_FRENCH_5_DAILY_URL)
    if resp.status_code != 200:
        return {"status": "error", "http_status": resp.status_code, "detail": "could not fetch source zip"}

    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            csv_name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
            raw_text = zf.read(csv_name).decode("utf-8", errors="replace")
    except Exception as e:
        return {"status": "error", "detail": f"could not extract CSV from zip: {e}"}

    lines = raw_text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if "Mkt-RF" in line:
            header_idx = i
            break
    if header_idx is None:
        return {"status": "error", "detail": "could not locate header row containing Mkt-RF in source file"}

    header = [h.strip() for h in lines[header_idx].split(",")]
    factor_names = header[1:]

    all_rows = []
    for line in lines[header_idx + 1:]:
        if not line.strip():
            break  # first blank line ends the daily data table
        parts = line.split(",")
        date_str = parts[0].strip()
        if not (date_str.isdigit() and len(date_str) == 8):
            break  # defensive stop if a non-date row appears
        row = {"date": date_str}
        for col_name, val in zip(factor_names, parts[1:]):
            try:
                row[col_name] = round(float(val) / 100.0, 6)
            except ValueError:
                row[col_name] = None
        all_rows.append(row)

    filtered = [r for r in all_rows if start_date <= r["date"] <= end_date]

    return {
        "status": "ok",
        "source": "Kenneth R. French Data Library (Dartmouth), public, free",
        "source_url": FAMA_FRENCH_5_DAILY_URL,
        "units": "decimal (source file is in percent; already converted)",
        "start_date": start_date,
        "end_date": end_date,
        "row_count": len(filtered),
        "rows": filtered,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "note": "Periodically-updated research file, not real-time. Dates are YYYYMMDD.",
    }


@asynccontextmanager
async def lifespan(app):
    async with mcp.session_manager.run():
        yield


app = Starlette(routes=[], lifespan=lifespan)
app.mount("/mcp-server", mcp.streamable_http_app())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
