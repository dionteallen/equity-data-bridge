"""
Equity Market Data Bridge — real historical price bars, opening auctions,
trades, and quotes for Equity Screener's cointegrated-pairs screening and
Statistical Arbitrage Bot's OU fit/re-check (Dynamic Grok Bot Desk)
"""

import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Annotated, Optional, Tuple

import io
import zipfile

import httpx
from pydantic import Field
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

# Alpaca Basic rejects historical SIP unless `end` is at least 15 minutes old
# (https://docs.alpaca.markets/us/docs/market-data-faq). An end that lands on
# that boundary is still rejected when this host's clock is slightly ahead of
# Alpaca's, so the clamp sits one extra minute back.
SIP_MIN_AGE = timedelta(minutes=15)
SIP_CLOCK_SKEW_GUARD = timedelta(minutes=1)
SIP_RECENCY_GUARD = SIP_MIN_AGE + SIP_CLOCK_SKEW_GUARD
# A YYYY-MM-DD end has no clock time. US equity sessions are dated in
# America/New_York, and that calendar date ends at 04:00 UTC (EDT) or
# 05:00 UTC (EST) the next day. 29 hours from UTC midnight is the later of
# those, so a date-only end is not forwarded while it can still cover the
# most recent 15 minutes.
DATE_ONLY_END_SPAN = timedelta(hours=29)

BarsRequest = Tuple[Optional[str], Optional[str], bool, Optional[dict]]

mcp = FastMCP(
    "equity-data-bridge",
    stateless_http=True,
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=False,
    ),
)
mcp.settings.streamable_http_path = "/"


def _feed_error(detail: str) -> dict:
    return {"status": "error", "detail": detail}


def _normalize_bar_feed(feed) -> Tuple[Optional[str], Optional[dict]]:
    """Return (feed, error). Blank means the historical default, iex."""
    if feed is None:
        return "iex", None
    if not isinstance(feed, str):
        return None, _feed_error(f"feed must be 'iex' or 'sip', got {feed!r}")
    text = feed.strip().lower()
    if text == "":
        return "iex", None
    if text not in ("iex", "sip"):
        return None, _feed_error(f"feed must be 'iex' or 'sip', got {feed!r}")
    return text, None


def _parse_bound(value, as_end: bool) -> Optional[datetime]:
    """Parse YYYY-MM-DD or ISO-8601 as an aware UTC instant.

    A date-only start is UTC midnight at the beginning of that date (the
    earliest instant it can include). A date-only end runs through the
    latest America/New_York midnight that date can reach, so the SIP delay
    check sees the whole session date. Returns None when value is not a
    recognized bound.
    """
    if not isinstance(value, str):
        return None
    text = value.strip()
    if len(text) == 10 and text[4] == "-" and text[7] == "-":
        try:
            day = datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            return None
        if as_end:
            return day + DATE_ONLY_END_SPAN - timedelta(seconds=1)
        return day
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _format_utc(moment: datetime) -> str:
    stamped = moment.astimezone(timezone.utc).replace(microsecond=0)
    return stamped.strftime("%Y-%m-%dT%H:%M:%SZ")


def _prepare_historical_bars_request(feed, start: str, end: str, now: Optional[datetime] = None) -> BarsRequest:
    """Resolve the feed and the end timestamp that will be sent to Alpaca.

    Returns (feed_name, end_to_send, end_adjusted, error). On an error the
    first three values are unused and error is a status dict; the caller
    must not call Alpaca.
    """
    feed_name, error = _normalize_bar_feed(feed)
    if error is not None:
        return None, None, False, error
    if feed_name != "sip":
        return feed_name, end, False, None

    end_instant = _parse_bound(end, as_end=True)
    if end_instant is None:
        return None, None, False, _feed_error(
            "feed 'sip' requires end as YYYY-MM-DD or an ISO-8601 datetime "
            f"so it can be kept at least 15 minutes old; got {end!r}"
        )

    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    else:
        now = now.astimezone(timezone.utc)
    cutoff = (now - SIP_RECENCY_GUARD).replace(microsecond=0)
    if end_instant <= cutoff:
        return feed_name, end, False, None

    start_instant = _parse_bound(start, as_end=False)
    allowed = _format_utc(cutoff)
    if start_instant is not None and start_instant > cutoff:
        return None, None, False, _feed_error(
            "feed 'sip' cannot query the most recent 15 minutes of consolidated "
            f"data. Requested start {start!r} is after the latest allowed end {allowed}"
        )
    return feed_name, allowed, True, None


@mcp.tool()
def get_historical_bars(
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    feed: Annotated[
        str,
        Field(
            description=(
                "Price feed: 'iex' (default, Investors Exchange only) or "
                "'sip' (consolidated tape, all US exchanges). Omit it, or "
                "leave it blank, for iex. Matching is case-insensitive; any "
                "other value is an error. For sip, an end inside the most "
                "recent 15 minutes is moved earlier so Alpaca's Basic plan "
                "accepts the query, and the response then includes "
                "end_requested and end_effective."
            )
        ),
    ] = "iex",
) -> dict:
    """
    Real historical OHLCV price bars for one stock symbol, from Alpaca's
    real market data — never invented, never estimated. This is raw
    material only — cointegration testing, hedge-ratio fitting, and OU
    parameter estimation are the calling bot's own job, not done here.

    symbol: ticker, e.g. "AAPL"
    timeframe: one of "1Min", "5Min", "15Min", "1Hour", "1Day"
    start: ISO date or datetime, e.g. "2026-01-01" or "2026-01-01T00:00:00Z"
    end: ISO date or datetime, same format as start. Naive datetimes are UTC.
    feed: "iex" (default) or "sip", case-insensitive. "iex" is the Investors
    Exchange only; callers that omit feed still receive IEX bars. "sip" is
    the consolidated tape across US exchanges. Any other value returns
    status "error" and does not call Alpaca.

    On Alpaca's Basic plan, historical SIP is rejected unless end is at
    least 15 minutes old (docs.alpaca.markets Market Data FAQ: "subscription
    does not permit querying recent SIP data"). When feed is sip and the
    requested end is newer than that, end is moved back to about 16 minutes
    ago (15 minutes plus a one-minute clock-skew guard) and the response
    includes end_requested and end_effective. A YYYY-MM-DD end is treated as
    lasting through the end of that America/New_York calendar date, so a
    date of "today" is moved back too. If the whole requested window is
    inside those 15 minutes, the call returns an error instead of asking
    Alpaca. Older sip windows are forwarded unchanged.
    """
    feed_name, end_sent, end_adjusted, error = _prepare_historical_bars_request(feed, start, end)
    if error is not None:
        return error

    url = f"{ALPACA_DATA_BASE}/stocks/{symbol}/bars"
    params = {
        "timeframe": timeframe,
        "start": start,
        "end": end_sent,
        "limit": 10000,
        "adjustment": "raw",
        "feed": feed_name,
    }
    with httpx.Client(timeout=30) as client:
        resp = client.get(url, headers=ALPACA_HEADERS, params=params)
    if resp.status_code != 200:
        return {"status": "error", "http_status": resp.status_code, "detail": resp.text}
    data = resp.json()
    result = {
        "status": "ok",
        "symbol": symbol,
        "timeframe": timeframe,
        "feed": feed_name,
    }
    if end_adjusted:
        result["end_requested"] = end
        result["end_effective"] = end_sent
        result["note"] = (
            "end was moved earlier so this SIP query stays outside the most "
            "recent 15 minutes. Alpaca's Basic plan rejects newer SIP history."
        )
    result["bar_count"] = len(data.get("bars", []))
    result["bars"] = data.get("bars", [])
    result["fetched_at"] = datetime.now(timezone.utc).isoformat()
    return result


@mcp.tool()
def get_latest_quote(symbol: str) -> dict:
    """The current real-time bid/ask quote for a symbol, from Alpaca's IEX feed."""
    url = f"{ALPACA_DATA_BASE}/stocks/{symbol}/quotes/latest"
    with httpx.Client(timeout=15) as client:
        resp = client.get(url, headers=ALPACA_HEADERS, params={"feed": "iex"})
    if resp.status_code != 200:
        return {"status": "error", "http_status": resp.status_code, "detail": resp.text}
    return {"status": "ok", "symbol": symbol, **resp.json()}


# Alpaca pages historical auctions and trades with next_page_token. limit
# applies to the whole page, not per symbol, and 10000 is the documented max.
ALPACA_PAGE_LIMIT = 10000
ALPACA_MAX_PAGES = 50
_SIP_CLAMP_NOTE = (
    "end was moved earlier so this SIP query stays outside the most "
    "recent 15 minutes. Alpaca's Basic plan rejects newer SIP history."
)


def _alpaca_http_error(resp) -> dict:
    """Return a non-200 Alpaca response without rewriting its body."""
    return {"status": "error", "http_status": resp.status_code, "detail": resp.text}


def _get_paginated(url: str, params: dict, payload_key: str, max_pages: Optional[int] = None):
    """GET an Alpaca multi-symbol endpoint, following next_page_token.

    Returns (merged, pages, error). Lists for the same symbol are concatenated
    in page order. Rows are not filtered or rewritten. A non-200 response, or
    a 200 body that is not the documented object, is returned as that status
    and body; earlier pages are dropped so the error is not hidden behind a
    partial success.
    """
    if max_pages is None:
        max_pages = ALPACA_MAX_PAGES
    merged: dict = {}
    pages = 0
    token = None
    with httpx.Client(timeout=30) as client:
        while True:
            query = dict(params)
            if token:
                query["page_token"] = token
            resp = client.get(url, headers=ALPACA_HEADERS, params=query)
            pages += 1
            if resp.status_code != 200:
                return None, pages, _alpaca_http_error(resp)
            try:
                data = resp.json()
            except ValueError:
                return None, pages, _alpaca_http_error(resp)
            if not isinstance(data, dict):
                return None, pages, _alpaca_http_error(resp)
            if payload_key in data and data[payload_key] is not None:
                bucket = data[payload_key]
            else:
                bucket = {}
            if not isinstance(bucket, dict):
                return None, pages, _alpaca_http_error(resp)
            for sym, items in bucket.items():
                if not isinstance(items, list):
                    return None, pages, _alpaca_http_error(resp)
                merged.setdefault(sym, []).extend(items)
            token = data.get("next_page_token")
            if not token:
                return merged, pages, None
            if pages >= max_pages:
                return merged, pages, {
                    "status": "error",
                    "detail": (
                        f"stopped after {max_pages} pages of {ALPACA_PAGE_LIMIT}; "
                        "narrow the window"
                    ),
                    "partial": True,
                }


def _finish_historical_result(result: dict, end: str, end_sent: str, end_adjusted: bool, error: Optional[dict]) -> dict:
    """Attach a page-cap notice and the SIP clamp fields, then a fetch time."""
    if error is not None:
        result["status"] = "partial"
        result["detail"] = error["detail"]
    if end_adjusted:
        result["end_requested"] = end
        result["end_effective"] = end_sent
        result["note"] = _SIP_CLAMP_NOTE
    result["fetched_at"] = datetime.now(timezone.utc).isoformat()
    return result


def _normalize_auction_feed(feed) -> Tuple[Optional[str], Optional[dict]]:
    """Auctions accept only sip. Blank means that default."""
    if feed is None:
        return "sip", None
    if not isinstance(feed, str):
        return None, _feed_error(f"auctions support feed 'sip' only, got {feed!r}")
    text = feed.strip().lower()
    if text in ("", "sip"):
        return "sip", None
    return None, _feed_error(f"auctions support feed 'sip' only, got {feed!r}")


def _without_closing_auctions(auctions: dict) -> dict:
    """Copy each day without its closing-auction array.

    Alpaca's day object uses "c" for the closing auction. Each print also
    uses "c", for its condition code. Only the day-level key is removed.
    Opening prints are the same objects Alpaca returned.
    """
    trimmed = {}
    for symbol, days in auctions.items():
        if not isinstance(days, list):
            trimmed[symbol] = days
            continue
        trimmed[symbol] = [
            {key: value for key, value in day.items() if key != "c"} if isinstance(day, dict) else day
            for day in days
        ]
    return trimmed


def _parse_conditions(conditions) -> Tuple[Optional[set], Optional[dict]]:
    """Return (wanted codes, error). An empty wanted set means do not filter."""
    if conditions is None:
        return None, None
    if not isinstance(conditions, str):
        return None, _feed_error(
            "conditions must be a comma-separated string of condition codes, "
            f"for example 'Q,O'; got {conditions!r}"
        )
    wanted = {part.strip() for part in conditions.split(",") if part.strip()}
    return (wanted or None), None


def _trade_conditions(trade) -> set:
    """Condition codes on one trade. Does not copy or modify the trade."""
    if not isinstance(trade, dict):
        return set()
    raw = trade.get("c")
    if isinstance(raw, str):
        code = raw.strip()
        return {code} if code else set()
    if isinstance(raw, (list, tuple)):
        return {code.strip() for code in raw if isinstance(code, str) and code.strip()}
    return set()


def _rows_for_symbol(grouped: dict, symbol: str) -> list:
    """Rows Alpaca keyed by this symbol. Matching is case-insensitive.

    The row objects are returned as Alpaca sent them. A single-symbol query
    can come back under a differently cased key; the fields are not rewritten
    to the caller's spelling.
    """
    if not isinstance(grouped, dict):
        return []
    if symbol in grouped and isinstance(grouped[symbol], list):
        return grouped[symbol]
    if isinstance(symbol, str):
        folded = symbol.casefold()
        for key, rows in grouped.items():
            if isinstance(key, str) and key.casefold() == folded and isinstance(rows, list):
                return rows
    return []


@mcp.tool()
def get_opening_auctions(
    symbols: Annotated[
        str,
        Field(description="Comma-separated tickers, for example \"PCG,INTC\"."),
    ],
    start: Annotated[
        str,
        Field(description="Inclusive start, YYYY-MM-DD or ISO-8601. Forwarded to Alpaca as given."),
    ],
    end: Annotated[
        str,
        Field(
            description=(
                "Inclusive end, YYYY-MM-DD or ISO-8601. On Alpaca's Basic plan, "
                "SIP history must be at least 15 minutes old. A newer end is moved "
                "earlier, and the response then includes end_requested and end_effective."
            )
        ),
    ],
    feed: Annotated[
        str,
        Field(
            description=(
                "Only 'sip' is valid for Alpaca auctions. Omit it, or leave it "
                "blank, for sip. Any other value is an error and does not call Alpaca. "
                "Alpaca's Basic plan rejects SIP from the most recent 15 minutes; "
                "this uses the same clamp as get_historical_bars."
            )
        ),
    ] = "sip",
    include_closing: Annotated[
        bool,
        Field(
            description=(
                "When false (default), omit each day's closing-auction array, the "
                "day-level key \"c\". Opening prints, including their condition "
                "code, are returned unchanged either way."
            )
        ),
    ] = False,
) -> dict:
    """
    Real historical opening auctions from Alpaca
    GET https://data.alpaca.markets/v2/stocks/auctions. Prints are passed
    through as Alpaca sent them — never invented, never estimated, and
    never reduced to a single exchange.

    symbols: comma-separated tickers, e.g. "PCG,INTC"
    start: YYYY-MM-DD or ISO-8601, e.g. "2026-07-21" or "2026-07-21T13:30:00Z"
    end: same format as start. Naive datetimes are UTC.
    feed: "sip" only. Alpaca documents that only sip is valid for auctions.
    Omit it, or leave it blank, for sip. Any other value returns status
    "error" and does not call Alpaca.
    include_closing: false (default) omits each day's closing-auction array,
    the day-level key "c". Opening prints are not rewritten either way.
    Set it true to also return closing auctions.

    Each day is {"d": date, "o": [opening prints], "c": [closing prints]}.
    A print is {t, x, p, s, c}: t timestamp, x exchange code, p price,
    s size, and c the condition. Condition "Q" is the Market Center
    Official Open. Condition "O" is the Market Center Opening Trade. The
    same symbol can print on more than one exchange at different prices, so
    the caller picks the listing exchange's official open (condition Q)
    print. This tool does not choose an exchange. That Q price is the
    exchange's official open and can differ from the open of a SIP bar.

    Pages follow next_page_token, up to 50 pages of 10,000 data points
    (a point is one symbol's day, not one print). Hitting
    that cap returns status "partial" and the rows already retrieved;
    status "ok" means the window was fully read. Any non-200 Alpaca
    response is returned as status "error" with http_status and the
    response body in detail, unchanged. A plan or permission error is
    not retried, rewritten, or replaced with another feed.

    On Alpaca's Basic plan, historical SIP is rejected unless end is at
    least 15 minutes old (docs.alpaca.markets Market Data FAQ:
    "subscription does not permit querying recent SIP data"). This uses
    the same recency clamp as get_historical_bars. When the requested end
    is newer than that, end is moved back to about 16 minutes ago (15
    minutes plus a one-minute clock-skew guard) and the response includes
    end_requested and end_effective. A YYYY-MM-DD end is treated as
    lasting through the end of that America/New_York calendar date. If
    the whole requested window is inside those 15 minutes, the call
    returns an error instead of asking Alpaca. Older windows are
    forwarded unchanged.
    """
    feed_name, feed_error = _normalize_auction_feed(feed)
    if feed_error is not None:
        return feed_error
    feed_name, end_sent, end_adjusted, error = _prepare_historical_bars_request(feed_name, start, end)
    if error is not None:
        return error
    params = {
        "symbols": symbols,
        "start": start,
        "end": end_sent,
        "feed": feed_name,
        "limit": ALPACA_PAGE_LIMIT,
    }
    auctions, pages, error = _get_paginated(f"{ALPACA_DATA_BASE}/stocks/auctions", params, "auctions")
    if error is not None and not error.get("partial"):
        return error
    if not include_closing:
        auctions = _without_closing_auctions(auctions)
    result = {
        "status": "ok",
        "feed": feed_name,
        "symbols": symbols,
        "start": start,
        "end": end_sent,
        "pages": pages,
        "include_closing": bool(include_closing),
        "auctions": auctions,
    }
    return _finish_historical_result(result, end, end_sent, end_adjusted, error)


@mcp.tool()
def get_trades(
    symbol: Annotated[str, Field(description="One ticker, for example \"PCG\".")],
    start: Annotated[
        str,
        Field(description="Inclusive start, YYYY-MM-DD or ISO-8601. Forwarded to Alpaca as given."),
    ],
    end: Annotated[
        str,
        Field(
            description=(
                "Inclusive end, YYYY-MM-DD or ISO-8601. For feed 'sip' on Alpaca's "
                "Basic plan, history must be at least 15 minutes old. A newer end "
                "is moved earlier, and the response then includes end_requested and "
                "end_effective. feed 'iex' is not clamped."
            )
        ),
    ],
    feed: Annotated[
        str,
        Field(
            description=(
                "Price feed: 'sip' (default, consolidated tape) or 'iex' "
                "(Investors Exchange only). Case-insensitive. Any other value "
                "is an error and does not call Alpaca. sip uses the same clamp "
                "as get_historical_bars for the most recent 15 minutes."
            )
        ),
    ] = "sip",
    conditions: Annotated[
        Optional[str],
        Field(
            description=(
                "Optional comma-separated condition codes, for example 'Q,O'. "
                "Applied after download: a trade is kept when its condition list "
                "contains any listed code. Kept trades are not rewritten. "
                "Condition Q is the Market Center Official Open; the caller "
                "picks the listing exchange's Q print. Omit to return every trade."
            )
        ),
    ] = None,
) -> dict:
    """
    Real historical trades for one symbol from Alpaca
    GET https://data.alpaca.markets/v2/stocks/trades. Trade objects are
    passed through as Alpaca sent them — never invented or rewritten.
    Keep the window narrow, for example 13:29:00Z–13:31:00Z around the
    open. A multi-day tape can be enormous.

    symbol: one ticker, e.g. "PCG"
    start: YYYY-MM-DD or ISO-8601. Naive datetimes are UTC.
    end: same format as start.
    feed: "sip" (default, consolidated tape) or "iex" (Investors Exchange
    only). Case-insensitive. Any other value returns status "error" and
    does not call Alpaca.
    conditions: optional comma-separated condition codes, e.g. "Q,O".
    Alpaca has no server-side condition filter, so this is applied after
    download and only drops rows. A trade is kept when its "c" array
    contains any listed code. Kept trades are not rewritten. Omit it to
    return every trade in the window.

    Condition "Q" is the Market Center Official Open and "O" is the
    Market Center Opening Trade. Field "x" is the exchange code. The
    caller picks the listing exchange's official open (condition Q)
    print. This tool does not choose an exchange.

    Pages follow next_page_token, up to 50 pages of 10,000 trades.
    Hitting that cap returns status "partial" and the rows already
    retrieved. Any non-200 Alpaca response is returned as status "error"
    with http_status and the response body in detail, unchanged. A plan
    or permission error is not retried or hidden.

    On Alpaca's Basic plan, historical SIP is rejected unless end is at
    least 15 minutes old (docs.alpaca.markets Market Data FAQ:
    "subscription does not permit querying recent SIP data"). feed "sip"
    uses the same recency clamp as get_historical_bars: a newer end is
    moved back to about 16 minutes ago and the response includes
    end_requested and end_effective. A YYYY-MM-DD end is treated as
    lasting through the end of that America/New_York calendar date. If
    the whole window is inside those 15 minutes, the call returns an
    error instead of asking Alpaca. feed "iex" is not clamped. Older sip
    windows are forwarded unchanged.
    """
    wanted, conditions_error = _parse_conditions(conditions)
    if conditions_error is not None:
        return conditions_error
    feed_name, end_sent, end_adjusted, error = _prepare_historical_bars_request(feed, start, end)
    if error is not None:
        return error
    params = {
        "symbols": symbol,
        "start": start,
        "end": end_sent,
        "feed": feed_name,
        "limit": ALPACA_PAGE_LIMIT,
    }
    grouped, pages, error = _get_paginated(f"{ALPACA_DATA_BASE}/stocks/trades", params, "trades")
    if error is not None and not error.get("partial"):
        return error
    rows = _rows_for_symbol(grouped, symbol)
    total = len(rows)
    if wanted:
        rows = [trade for trade in rows if wanted.intersection(_trade_conditions(trade))]
    result = {
        "status": "ok",
        "symbol": symbol,
        "feed": feed_name,
        "start": start,
        "end": end_sent,
        "pages": pages,
        "trade_count_unfiltered": total,
        "conditions_filter": conditions if wanted else None,
        "trade_count": len(rows),
        "trades": rows,
    }
    return _finish_historical_result(result, end, end_sent, end_adjusted, error)


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
