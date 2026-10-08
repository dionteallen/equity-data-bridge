"""Stdlib checks for historical-bar feed selection. No network, no extra packages."""

import os
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

os.environ.setdefault("ALPACA_API_KEY_ID", "test-key")
os.environ.setdefault("ALPACA_API_SECRET_KEY", "test-secret")

import equity_data_bridge_server as bridge


NOW = datetime(2026, 10, 8, 21, 38, 0, tzinfo=timezone.utc)
CUTOFF = "2026-10-08T21:22:00Z"  # now minus the 16-minute SIP guard


class _FrozenDatetime(datetime):
    """datetime stand-in whose now() stays put. Other constructors still work."""

    @classmethod
    def now(cls, tz=None):
        if tz is None:
            return NOW.replace(tzinfo=None)
        return NOW.astimezone(tz)


def _bars_response(bars=None):
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"bars": bars if bars is not None else [{"t": "2026-07-21T13:30:00Z", "o": 17.33}]}
    client = MagicMock()
    client.get.return_value = response
    manager = MagicMock()
    manager.__enter__.return_value = client
    manager.__exit__.return_value = False
    return manager, client


class PrepareHistoricalBarsTests(unittest.TestCase):
    def test_guard_is_at_least_fifteen_minutes(self):
        self.assertGreaterEqual(bridge.SIP_RECENCY_GUARD, timedelta(minutes=15))
        self.assertEqual(bridge.SIP_RECENCY_GUARD, timedelta(minutes=16))

    def test_omitted_feed_stays_iex_and_does_not_move_end(self):
        feed, end, adjusted, error = bridge._prepare_historical_bars_request(
            "iex", "2026-10-08T21:00:00Z", "2026-10-08T21:40:00Z", now=NOW
        )
        self.assertEqual((feed, end, adjusted, error), ("iex", "2026-10-08T21:40:00Z", False, None))

    def test_blank_and_none_feed_default_to_iex(self):
        for feed in (None, "", "   "):
            resolved, end, adjusted, error = bridge._prepare_historical_bars_request(
                feed, "2026-01-01", "2026-01-02", now=NOW
            )
            self.assertEqual(resolved, "iex", feed)
            self.assertFalse(adjusted)
            self.assertIsNone(error)
            self.assertEqual(end, "2026-01-02")

    def test_feed_match_is_case_insensitive(self):
        feed, end, adjusted, error = bridge._prepare_historical_bars_request(
            "  SIP ", "2026-07-21", "2026-07-21", now=NOW
        )
        self.assertEqual((feed, end, adjusted, error), ("sip", "2026-07-21", False, None))

    def test_invalid_feed_is_an_error_and_names_the_value(self):
        feed, end, adjusted, error = bridge._prepare_historical_bars_request(
            "boats", "2026-07-21", "2026-07-21", now=NOW
        )
        self.assertIsNone(feed)
        self.assertFalse(adjusted)
        self.assertEqual(error["status"], "error")
        self.assertIn("'iex'", error["detail"])
        self.assertIn("'sip'", error["detail"])
        self.assertIn("boats", error["detail"])

    def test_historical_sip_end_is_forwarded_unchanged(self):
        cases = (
            "2026-07-21",
            "2026-07-21T13:30:00Z",
            "2020-01-02T00:00:00-05:00",
            "2026-06-17T13:30:00",
        )
        for end in cases:
            feed, sent, adjusted, error = bridge._prepare_historical_bars_request(
                "sip", "2026-01-01", end, now=NOW
            )
            self.assertEqual((feed, sent, adjusted, error), ("sip", end, False, None), end)

    def test_recent_sip_timestamp_is_clamped_to_the_guard(self):
        feed, sent, adjusted, error = bridge._prepare_historical_bars_request(
            "sip", "2026-10-08T13:30:00Z", "2026-10-08T21:40:00Z", now=NOW
        )
        self.assertEqual((feed, sent, adjusted, error), ("sip", CUTOFF, True, None))

    def test_sip_end_exactly_at_the_cutoff_is_not_rewritten(self):
        feed, sent, adjusted, error = bridge._prepare_historical_bars_request(
            "sip", "2026-10-08T13:30:00Z", CUTOFF, now=NOW
        )
        self.assertEqual((feed, sent, adjusted, error), ("sip", CUTOFF, False, None))

    def test_date_only_end_of_today_is_clamped(self):
        # 2026-10-08 as a session date still covers the New York evening,
        # which at 21:38 UTC is inside the SIP delay.
        feed, sent, adjusted, error = bridge._prepare_historical_bars_request(
            "sip", "2026-10-08", "2026-10-08", now=NOW
        )
        self.assertEqual((feed, sent, adjusted, error), ("sip", CUTOFF, True, None))

    def test_date_only_end_just_after_utc_midnight_is_still_clamped(self):
        # 00:30 UTC is 20:30 ET the previous calendar date. end=<that date>
        # must not be forwarded raw: the New York date has not finished.
        now = datetime(2026, 10, 9, 0, 30, tzinfo=timezone.utc)
        feed, sent, adjusted, error = bridge._prepare_historical_bars_request(
            "sip", "2026-10-08T13:30:00Z", "2026-10-08", now=now
        )
        self.assertEqual(feed, "sip")
        self.assertTrue(adjusted)
        self.assertEqual(sent, "2026-10-09T00:14:00Z")
        self.assertIsNone(error)

    def test_finished_session_date_is_not_clamped_the_next_morning(self):
        # 14:00 UTC is 10:00 ET, so the previous New York date ended hours ago.
        now = datetime(2026, 10, 9, 14, 0, tzinfo=timezone.utc)
        feed, sent, adjusted, error = bridge._prepare_historical_bars_request(
            "sip", "2026-10-08", "2026-10-08", now=now
        )
        self.assertEqual((feed, sent, adjusted, error), ("sip", "2026-10-08", False, None))

    def test_window_entirely_inside_the_delay_does_not_call_through(self):
        feed, sent, adjusted, error = bridge._prepare_historical_bars_request(
            "sip", "2026-10-08T21:30:00Z", "2026-10-08T21:40:00Z", now=NOW
        )
        self.assertIsNone(feed)
        self.assertEqual(error["status"], "error")
        self.assertIn("15 minutes", error["detail"])
        self.assertIn(CUTOFF, error["detail"])
        self.assertIn("2026-10-08T21:30:00Z", error["detail"])

    def test_unparseable_sip_end_is_rejected(self):
        feed, sent, adjusted, error = bridge._prepare_historical_bars_request(
            "sip", "2026-07-21", "next week", now=NOW
        )
        self.assertEqual(error["status"], "error")
        self.assertIn("next week", error["detail"])
        self.assertIn("15 minutes", error["detail"])

    def test_iex_does_not_require_a_parseable_end(self):
        feed, sent, adjusted, error = bridge._prepare_historical_bars_request(
            "iex", "whenever", "next week", now=NOW
        )
        self.assertEqual((feed, sent, adjusted, error), ("iex", "next week", False, None))


class GetHistoricalBarsToolTests(unittest.TestCase):
    def test_schema_advertises_optional_feed_and_leaves_other_tools_alone(self):
        tools = {tool.name: tool for tool in bridge.mcp._tool_manager.list_tools()}
        self.assertEqual(
            sorted(tools),
            [
                "get_fama_french_5factors",
                "get_historical_bars",
                "get_latest_quote",
                "get_opening_auctions",
                "get_trades",
            ],
        )
        bars = tools["get_historical_bars"]
        feed = bars.parameters["properties"]["feed"]
        self.assertNotIn("feed", bars.parameters.get("required", []))
        self.assertEqual(feed.get("default"), "iex")
        self.assertIn("sip", feed.get("description", ""))
        self.assertIn("end_effective", feed.get("description", ""))
        description = bars.description
        self.assertIn("sip", description)
        self.assertIn("iex", description)
        self.assertIn("15 minutes", description)
        self.assertIn("end_effective", description)
        quote = tools["get_latest_quote"]
        self.assertEqual(list(quote.parameters["properties"]), ["symbol"])
        self.assertIn("IEX", quote.description)

    @patch("equity_data_bridge_server.httpx.Client")
    def test_default_call_still_requests_iex(self, client_cls):
        manager, client = _bars_response()
        client_cls.return_value = manager
        result = bridge.get_historical_bars("PCG", "15Min", "2026-07-21T13:30:00Z", "2026-07-21T14:30:00Z")
        params = client.get.call_args.kwargs["params"]
        self.assertEqual(params["feed"], "iex")
        self.assertEqual(params["end"], "2026-07-21T14:30:00Z")
        self.assertEqual(params["adjustment"], "raw")
        self.assertEqual(params["limit"], 10000)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["feed"], "iex")
        self.assertNotIn("end_effective", result)
        self.assertEqual(result["bars"][0]["o"], 17.33)

    @patch("equity_data_bridge_server.httpx.Client")
    def test_sip_reports_the_feed_and_the_effective_end_when_moved(self, client_cls):
        manager, client = _bars_response()
        client_cls.return_value = manager
        with patch("equity_data_bridge_server.datetime", _FrozenDatetime):
            result = bridge.get_historical_bars(
                "INTC",
                "15Min",
                "2026-10-08T13:30:00Z",
                "2026-10-08T21:40:00Z",
                feed="sip",
            )
        params = client.get.call_args.kwargs["params"]
        self.assertEqual(params["feed"], "sip")
        self.assertEqual(params["end"], CUTOFF)
        self.assertEqual(params["start"], "2026-10-08T13:30:00Z")
        self.assertEqual(result["feed"], "sip")
        self.assertEqual(result["end_requested"], "2026-10-08T21:40:00Z")
        self.assertEqual(result["end_effective"], CUTOFF)
        self.assertIn("15 minutes", result["note"])

    @patch("equity_data_bridge_server.httpx.Client")
    def test_invalid_feed_does_not_call_alpaca(self, client_cls):
        result = bridge.get_historical_bars("PCG", "15Min", "2026-07-21", "2026-07-21", feed="nasdaq")
        client_cls.assert_not_called()
        self.assertEqual(result["status"], "error")
        self.assertIn("nasdaq", result["detail"])

    @patch("equity_data_bridge_server.httpx.Client")
    def test_recent_only_sip_window_does_not_call_alpaca(self, client_cls):
        with patch("equity_data_bridge_server.datetime", _FrozenDatetime):
            result = bridge.get_historical_bars(
                "PCG",
                "1Min",
                "2026-10-08T21:30:00Z",
                "2026-10-08T21:38:00Z",
                feed="sip",
            )
        client_cls.assert_not_called()
        self.assertEqual(result["status"], "error")
        self.assertIn(CUTOFF, result["detail"])


if __name__ == "__main__":
    unittest.main()
