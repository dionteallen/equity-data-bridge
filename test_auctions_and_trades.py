"""Stdlib checks for opening auctions and trades. No network, no extra packages."""

import inspect
import json
import os
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

os.environ.setdefault("ALPACA_API_KEY_ID", "test-key")
os.environ.setdefault("ALPACA_API_SECRET_KEY", "test-secret")

import equity_data_bridge_server as bridge


NOW = datetime(2026, 10, 8, 21, 38, 0, tzinfo=timezone.utc)
CUTOFF = "2026-10-08T21:22:00Z"  # now minus the 16-minute SIP guard
AUCTIONS_URL = "https://data.alpaca.markets/v2/stocks/auctions"
TRADES_URL = "https://data.alpaca.markets/v2/stocks/trades"
SIP_DENIED = '{"code":42210000,"message":"subscription does not permit querying recent SIP data"}'


class _FrozenDatetime(datetime):
    """datetime stand-in whose now() stays put. Other constructors still work."""

    @classmethod
    def now(cls, tz=None):
        if tz is None:
            return NOW.replace(tzinfo=None)
        return NOW.astimezone(tz)


def _response(status=200, payload=None, text=None, json_error=None):
    response = MagicMock()
    response.status_code = status
    if payload is None:
        payload = {}
    if json_error is not None:
        response.json.side_effect = json_error
    else:
        response.json.return_value = payload
    response.text = json.dumps(payload) if text is None else text
    return response


def _client(responses):
    client = MagicMock()
    client.get.side_effect = responses
    manager = MagicMock()
    manager.__enter__.return_value = client
    manager.__exit__.return_value = False
    return manager, client


def _opening(price=17.33, exchange="N", condition="Q"):
    return {
        "c": condition,
        "p": price,
        "s": 100,
        "t": "2026-07-21T13:30:00.188390144Z",
        "x": exchange,
    }


def _day(opening_prints, closing_prints=None, when="2026-07-21"):
    day = {"d": when, "o": list(opening_prints)}
    if closing_prints is not None:
        day["c"] = list(closing_prints)
    return day


class ToolContractTests(unittest.TestCase):
    def test_new_tools_are_registered_and_old_signatures_stay(self):
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
        self.assertEqual(
            list(inspect.signature(bridge.get_historical_bars).parameters),
            ["symbol", "timeframe", "start", "end", "feed"],
        )
        self.assertEqual(list(inspect.signature(bridge.get_latest_quote).parameters), ["symbol"])
        self.assertEqual(
            list(inspect.signature(bridge.get_fama_french_5factors).parameters),
            ["start_date", "end_date"],
        )
        self.assertEqual(
            list(inspect.signature(bridge.get_opening_auctions).parameters),
            ["symbols", "start", "end", "feed", "include_closing"],
        )
        self.assertEqual(
            list(inspect.signature(bridge.get_trades).parameters),
            ["symbol", "start", "end", "feed", "conditions"],
        )

    def test_opening_auction_description_states_the_sip_rules(self):
        tool = {item.name: item for item in bridge.mcp._tool_manager.list_tools()}["get_opening_auctions"]
        description = tool.description
        self.assertIn("15 minutes", description)
        self.assertIn("Basic", description)
        self.assertIn("Q", description)
        self.assertIn("listing exchange", description)
        self.assertIn("official open", description)
        self.assertIn("does not choose an exchange", description)
        self.assertIn("10,000", description)
        self.assertIn("partial", description)
        self.assertIn("http_status", description)
        for name in ("symbols", "start", "end", "feed", "include_closing"):
            self.assertIn(name, description)
        params = tool.parameters
        self.assertEqual(params["required"], ["symbols", "start", "end"])
        self.assertEqual(params["properties"]["feed"]["default"], "sip")
        self.assertIn("sip", params["properties"]["feed"]["description"])
        self.assertIn("15 minutes", params["properties"]["feed"]["description"])
        self.assertEqual(params["properties"]["include_closing"]["default"], False)
        self.assertIn("15 minutes", params["properties"]["end"]["description"])
        self.assertIn("end_effective", params["properties"]["end"]["description"])

    def test_trades_description_states_the_filter_and_the_sip_rules(self):
        tool = {item.name: item for item in bridge.mcp._tool_manager.list_tools()}["get_trades"]
        description = tool.description
        self.assertIn("15 minutes", description)
        self.assertIn("Basic", description)
        self.assertIn("Q", description)
        self.assertIn("listing exchange", description)
        self.assertIn("official open", description)
        self.assertIn("narrow", description)
        self.assertIn("not rewritten", description)
        self.assertIn("partial", description)
        self.assertIn("http_status", description)
        for name in ("symbol", "start", "end", "feed", "conditions"):
            self.assertIn(name, description)
        params = tool.parameters
        self.assertEqual(params["required"], ["symbol", "start", "end"])
        feed = params["properties"]["feed"]
        self.assertEqual(feed["default"], "sip")
        self.assertIn("iex", feed["description"])
        self.assertIn("15 minutes", feed["description"])
        conditions = params["properties"]["conditions"]
        self.assertNotIn("conditions", params["required"])
        self.assertIn("Q,O", conditions["description"])
        self.assertIn("listing exchange", conditions["description"])
        self.assertIn("not rewritten", conditions["description"])


class OpeningAuctionTests(unittest.TestCase):
    def _call(self, *args, **kwargs):
        with patch("equity_data_bridge_server.datetime", _FrozenDatetime):
            return bridge.get_opening_auctions(*args, **kwargs)

    @patch("equity_data_bridge_server.httpx.Client")
    def test_returns_opening_prints_unchanged_and_omits_the_close(self, client_cls):
        official = _opening(17.33, "N", "Q")
        other = _opening(17.4, "P", "Q")
        closing = _opening(18.01, "N", "M")
        day = _day([official, other], [closing])
        manager, client = _client([_response(payload={"auctions": {"PCG": [day]}, "next_page_token": None})])
        client_cls.return_value = manager

        result = self._call("PCG", "2026-07-21", "2026-07-21")

        self.assertEqual(client.get.call_args.args[0], AUCTIONS_URL)
        self.assertEqual(client.get.call_args.kwargs["headers"], bridge.ALPACA_HEADERS)
        self.assertEqual(
            client.get.call_args.kwargs["params"],
            {
                "symbols": "PCG",
                "start": "2026-07-21",
                "end": "2026-07-21",
                "feed": "sip",
                "limit": 10000,
            },
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["feed"], "sip")
        self.assertEqual(result["pages"], 1)
        self.assertFalse(result["include_closing"])
        self.assertNotIn("end_requested", result)
        self.assertNotIn("end_effective", result)
        self.assertNotIn("note", result)
        returned = result["auctions"]["PCG"][0]
        self.assertNotIn("c", returned)
        self.assertEqual(returned["d"], "2026-07-21")
        self.assertIs(returned["o"], day["o"])
        self.assertIs(returned["o"][0], official)
        self.assertEqual(official["c"], "Q")
        self.assertEqual(official["p"], 17.33)
        self.assertEqual(official["x"], "N")
        self.assertIn("c", day)
        self.assertEqual(result["fetched_at"], "2026-10-08T21:38:00+00:00")

    @patch("equity_data_bridge_server.httpx.Client")
    def test_include_closing_keeps_the_day_alpaca_sent(self, client_cls):
        day = _day([_opening()], [_opening(18.0, "Q", "6")])
        manager, client = _client([_response(payload={"auctions": {"INTC": [day]}, "next_page_token": None})])
        client_cls.return_value = manager

        result = self._call("INTC", "2026-06-17", "2026-06-17", include_closing=True)

        self.assertTrue(result["include_closing"])
        self.assertIs(result["auctions"]["INTC"][0], day)
        self.assertEqual(client.get.call_args.kwargs["params"]["end"], "2026-06-17")

    @patch("equity_data_bridge_server.httpx.Client")
    def test_pages_are_merged_in_order_without_rewriting_symbols(self, client_cls):
        first = _day([_opening(1.0)], when="2026-07-21")
        second = _day([_opening(2.0)], when="2026-07-22")
        other = _day([_opening(3.0, "Q", "O")], when="2026-07-21")
        manager, client = _client([
            _response(payload={"auctions": {"pcg": [first], "INTC": [other]}, "next_page_token": "tok-1"}),
            _response(payload={"auctions": {"pcg": [second]}, "next_page_token": None}),
        ])
        client_cls.return_value = manager

        result = self._call("pcg, INTC", "2026-07-21", "2026-07-22", feed=" SIP ")

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["pages"], 2)
        self.assertEqual(result["symbols"], "pcg, INTC")
        self.assertEqual([day["d"] for day in result["auctions"]["pcg"]], ["2026-07-21", "2026-07-22"])
        self.assertIs(result["auctions"]["pcg"][0]["o"][0], first["o"][0])
        self.assertIn("INTC", result["auctions"])
        self.assertNotIn("PCG", result["auctions"])
        first_params = client.get.call_args_list[0].kwargs["params"]
        second_params = client.get.call_args_list[1].kwargs["params"]
        self.assertNotIn("page_token", first_params)
        self.assertEqual(first_params["feed"], "sip")
        self.assertEqual(first_params["symbols"], "pcg, INTC")
        self.assertEqual(second_params["page_token"], "tok-1")
        self.assertEqual(
            {key: value for key, value in second_params.items() if key != "page_token"},
            first_params,
        )

    @patch("equity_data_bridge_server.httpx.Client")
    def test_page_cap_returns_partial_and_stops(self, client_cls):
        pages = [
            _response(payload={"auctions": {"PCG": [_day([_opening(float(i))])]}, "next_page_token": f"tok-{i}"})
            for i in range(3)
        ]
        manager, client = _client(pages)
        client_cls.return_value = manager

        with patch("equity_data_bridge_server.ALPACA_MAX_PAGES", 2):
            result = self._call("PCG", "2026-07-01", "2026-07-31")

        self.assertEqual(client.get.call_count, 2)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["pages"], 2)
        self.assertEqual(len(result["auctions"]["PCG"]), 2)
        self.assertIn("stopped after 2 pages of 10000", result["detail"])
        self.assertIn("narrow the window", result["detail"])
        self.assertNotIn("partial", result)

    @patch("equity_data_bridge_server.httpx.Client")
    def test_only_sip_is_accepted(self, client_cls):
        for feed in ("iex", "boats", "otc", "nasdaq", "sip,iex", 1):
            with self.subTest(feed=feed):
                result = self._call("PCG", "2026-07-21", "2026-07-21", feed=feed)
                self.assertEqual(result["status"], "error")
                self.assertIn("sip", result["detail"])
                self.assertIn(repr(feed), result["detail"])
                self.assertNotIn("http_status", result)
        client_cls.assert_not_called()

    @patch("equity_data_bridge_server.httpx.Client")
    def test_blank_or_omitted_feed_uses_sip(self, client_cls):
        for feed in ("   ", None):
            with self.subTest(feed=feed):
                manager, client = _client([_response(payload={"auctions": {}, "next_page_token": None})])
                client_cls.return_value = manager
                result = self._call("PCG", "2026-07-21", "2026-07-21", feed=feed)
                self.assertEqual(result["status"], "ok")
                self.assertEqual(client.get.call_args.kwargs["params"]["feed"], "sip")

    @patch("equity_data_bridge_server.httpx.Client")
    def test_non_200_is_the_status_and_body_with_no_retry(self, client_cls):
        for status in (400, 401, 403, 422, 429, 500):
            with self.subTest(status=status):
                body = SIP_DENIED if status in (403, 422) else f"nope {status}"
                response = _response(status=status, text=body)
                response.json.side_effect = AssertionError("non-200 body must be returned verbatim")
                manager, client = _client([response])
                client_cls.return_value = manager
                result = self._call("PCG", "2026-07-21", "2026-07-21")
                self.assertEqual(set(result), {"status", "http_status", "detail"})
                self.assertEqual(result["status"], "error")
                self.assertEqual(result["http_status"], status)
                self.assertEqual(result["detail"], body)
                self.assertEqual(client.get.call_count, 1)

    @patch("equity_data_bridge_server.httpx.Client")
    def test_later_page_error_is_not_hidden_behind_earlier_rows(self, client_cls):
        denied = _response(status=403, text=SIP_DENIED)
        denied.json.side_effect = AssertionError("non-200 body must be returned verbatim")
        manager, client = _client([
            _response(payload={"auctions": {"PCG": [_day([_opening()])]}, "next_page_token": "tok-1"}),
            denied,
        ])
        client_cls.return_value = manager

        result = self._call("PCG", "2026-07-21", "2026-09-01")

        self.assertEqual(result, {"status": "error", "http_status": 403, "detail": SIP_DENIED})
        self.assertEqual(client.get.call_count, 2)

    @patch("equity_data_bridge_server.httpx.Client")
    def test_malformed_200_returns_the_body(self, client_cls):
        body = "not-json"
        response = _response(text=body, json_error=json.JSONDecodeError("Expecting value", body, 0))
        manager, _client_obj = _client([response])
        client_cls.return_value = manager
        result = self._call("PCG", "2026-07-21", "2026-07-21")
        self.assertEqual(result, {"status": "error", "http_status": 200, "detail": body})

        wrong = '{"auctions":["nope"],"next_page_token":null}'
        manager, _client_obj = _client([_response(payload={"auctions": ["nope"], "next_page_token": None}, text=wrong)])
        client_cls.return_value = manager
        result = self._call("PCG", "2026-07-21", "2026-07-21")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["http_status"], 200)
        self.assertEqual(result["detail"], wrong)

    @patch("equity_data_bridge_server.httpx.Client")
    def test_recent_sip_end_is_clamped_with_the_bars_guard(self, client_cls):
        manager, client = _client([_response(payload={"auctions": {}, "next_page_token": None})])
        client_cls.return_value = manager
        result = self._call("PCG", "2026-10-08T13:30:00Z", "2026-10-08T21:40:00Z")
        self.assertEqual(client.get.call_args.kwargs["params"]["end"], CUTOFF)
        self.assertEqual(client.get.call_args.kwargs["params"]["start"], "2026-10-08T13:30:00Z")
        self.assertEqual(result["end_requested"], "2026-10-08T21:40:00Z")
        self.assertEqual(result["end_effective"], CUTOFF)
        self.assertEqual(result["end"], CUTOFF)
        self.assertIn("15 minutes", result["note"])
        self.assertEqual(client.get.call_count, 1)

    @patch("equity_data_bridge_server.httpx.Client")
    def test_date_only_end_of_today_is_clamped(self, client_cls):
        manager, client = _client([_response(payload={"auctions": {}, "next_page_token": None})])
        client_cls.return_value = manager
        result = self._call("INTC", "2026-10-08", "2026-10-08")
        self.assertEqual(client.get.call_args.kwargs["params"]["end"], CUTOFF)
        self.assertEqual(result["end_requested"], "2026-10-08")
        self.assertEqual(result["end_effective"], CUTOFF)

    @patch("equity_data_bridge_server.httpx.Client")
    def test_window_inside_the_delay_does_not_call_alpaca(self, client_cls):
        result = self._call("PCG", "2026-10-08T21:30:00Z", "2026-10-08T21:40:00Z")
        client_cls.assert_not_called()
        self.assertEqual(result["status"], "error")
        self.assertIn("15 minutes", result["detail"])
        self.assertIn(CUTOFF, result["detail"])

    @patch("equity_data_bridge_server.httpx.Client")
    def test_unparseable_end_does_not_call_alpaca(self, client_cls):
        result = self._call("PCG", "2026-07-21", "next week")
        client_cls.assert_not_called()
        self.assertEqual(result["status"], "error")
        self.assertIn("next week", result["detail"])

    @patch("equity_data_bridge_server.httpx.Client")
    def test_old_window_forwards_an_unparsed_start(self, client_cls):
        manager, client = _client([_response(payload={"auctions": {}, "next_page_token": None})])
        client_cls.return_value = manager
        result = self._call("PCG", "whenever", "2026-07-21")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(client.get.call_args.kwargs["params"]["start"], "whenever")
        self.assertEqual(client.get.call_args.kwargs["params"]["end"], "2026-07-21")


class TradesTests(unittest.TestCase):
    def _call(self, *args, **kwargs):
        with patch("equity_data_bridge_server.datetime", _FrozenDatetime):
            return bridge.get_trades(*args, **kwargs)

    @patch("equity_data_bridge_server.httpx.Client")
    def test_returns_trades_unchanged_for_a_narrow_window(self, client_cls):
        regular = {"c": ["@"], "i": 1, "p": 17.2, "s": 40, "t": "2026-07-21T13:29:01Z", "x": "P", "z": "B"}
        official = {"c": ["Q"], "i": 2, "p": 17.33, "s": 100, "t": "2026-07-21T13:30:00.1Z", "x": "N", "z": "A"}
        manager, client = _client([
            _response(payload={"trades": {"PCG": [regular, official]}, "next_page_token": None})
        ])
        client_cls.return_value = manager

        result = self._call("PCG", "2026-07-21T13:29:00Z", "2026-07-21T13:31:00Z")

        self.assertEqual(client.get.call_args.args[0], TRADES_URL)
        self.assertEqual(client.get.call_args.kwargs["headers"], bridge.ALPACA_HEADERS)
        self.assertEqual(
            client.get.call_args.kwargs["params"],
            {
                "symbols": "PCG",
                "start": "2026-07-21T13:29:00Z",
                "end": "2026-07-21T13:31:00Z",
                "feed": "sip",
                "limit": 10000,
            },
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["trade_count_unfiltered"], 2)
        self.assertEqual(result["trade_count"], 2)
        self.assertIsNone(result["conditions_filter"])
        self.assertIs(result["trades"][0], regular)
        self.assertIs(result["trades"][1], official)
        self.assertNotIn("end_effective", result)

    @patch("equity_data_bridge_server.httpx.Client")
    def test_conditions_keep_any_match_and_do_not_rewrite_rows(self, client_cls):
        regular = {"c": ["@", "T"], "p": 1.0, "x": "P"}
        official = {"c": ["@", "Q"], "p": 17.33, "x": "N"}
        opening = {"c": ["O", "@"], "p": 17.33, "x": "N"}
        both = {"c": ["Q", "O"], "p": 17.33, "x": "Q"}
        missing = {"p": 9.0, "x": "P"}
        manager, client = _client([
            _response(payload={
                "trades": {"PCG": [regular, official, opening, both, missing]},
                "next_page_token": None,
            })
        ])
        client_cls.return_value = manager

        result = self._call(
            "PCG",
            "2026-07-21T13:29:00Z",
            "2026-07-21T13:31:00Z",
            conditions=" Q , O ",
        )

        self.assertNotIn("conditions", client.get.call_args.kwargs["params"])
        self.assertEqual(result["trade_count_unfiltered"], 5)
        self.assertEqual(result["trade_count"], 3)
        self.assertEqual(result["conditions_filter"], " Q , O ")
        self.assertIs(result["trades"][0], official)
        self.assertIs(result["trades"][1], opening)
        self.assertIs(result["trades"][2], both)
        self.assertEqual(official, {"c": ["@", "Q"], "p": 17.33, "x": "N"})

    @patch("equity_data_bridge_server.httpx.Client")
    def test_string_condition_is_one_code_not_characters(self, client_cls):
        whole = {"c": "Q", "p": 17.33, "x": "N"}
        glued = {"c": "QO", "p": 1.0, "x": "P"}
        manager, _client_obj = _client([
            _response(payload={"trades": {"PCG": [whole, glued]}, "next_page_token": None})
        ])
        client_cls.return_value = manager
        result = self._call("PCG", "2026-07-21T13:29:00Z", "2026-07-21T13:31:00Z", conditions="Q,O")
        self.assertEqual(result["trades"], [whole])
        self.assertIs(result["trades"][0], whole)
        self.assertEqual(whole["c"], "Q")

    @patch("equity_data_bridge_server.httpx.Client")
    def test_blank_conditions_do_not_drop_rows(self, client_cls):
        trade = {"c": ["@"], "p": 1.0}
        payload = {"trades": {"PCG": [trade]}, "next_page_token": None}
        for conditions in ("", "  ", " , "):
            with self.subTest(conditions=conditions):
                manager, _client_obj = _client([_response(payload=payload)])
                client_cls.return_value = manager
                result = self._call("PCG", "2026-07-21T13:29:00Z", "2026-07-21T13:31:00Z", conditions=conditions)
                self.assertEqual(result["status"], "ok")
                self.assertIsNone(result["conditions_filter"])
                self.assertIs(result["trades"][0], trade)

    @patch("equity_data_bridge_server.httpx.Client")
    def test_symbol_lookup_is_case_insensitive_and_keeps_alpaca_fields(self, client_cls):
        trade = {"c": ["Q"], "p": 17.33, "x": "N"}
        manager, client = _client([_response(payload={"trades": {"PCG": [trade]}, "next_page_token": None})])
        client_cls.return_value = manager
        result = self._call("pcg", "2026-07-21T13:29:00Z", "2026-07-21T13:31:00Z")
        self.assertEqual(client.get.call_args.kwargs["params"]["symbols"], "pcg")
        self.assertEqual(result["symbol"], "pcg")
        self.assertIs(result["trades"][0], trade)

    @patch("equity_data_bridge_server.httpx.Client")
    def test_pages_are_filtered_after_they_are_merged(self, client_cls):
        early = {"c": ["Q"], "p": 1.0, "t": "2026-06-17T13:30:00Z"}
        middle = {"c": ["@"], "p": 2.0, "t": "2026-06-17T13:30:01Z"}
        late = {"c": ["O"], "p": 3.0, "t": "2026-06-17T13:30:02Z"}
        manager, client = _client([
            _response(payload={"trades": {"INTC": [early, middle]}, "next_page_token": "next"}),
            _response(payload={"trades": {"INTC": [late]}, "next_page_token": None}),
        ])
        client_cls.return_value = manager
        result = self._call("INTC", "2026-06-17T13:29:00Z", "2026-06-17T13:31:00Z", conditions="Q,O")
        self.assertEqual(result["pages"], 2)
        self.assertEqual(result["trade_count_unfiltered"], 3)
        self.assertEqual(result["trades"], [early, late])
        self.assertEqual(client.get.call_args_list[1].kwargs["params"]["page_token"], "next")

    @patch("equity_data_bridge_server.httpx.Client")
    def test_page_cap_is_partial_after_the_filter(self, client_cls):
        pages = []
        for i in range(3):
            pages.append(_response(payload={
                "trades": {"PCG": [{"c": ["Q"] if i == 0 else ["@"], "p": float(i)}]},
                "next_page_token": f"tok-{i}",
            }))
        manager, client = _client(pages)
        client_cls.return_value = manager
        with patch("equity_data_bridge_server.ALPACA_MAX_PAGES", 2):
            result = self._call("PCG", "2026-07-21T13:29:00Z", "2026-07-21T13:31:00Z", conditions="Q")
        self.assertEqual(client.get.call_count, 2)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["trade_count_unfiltered"], 2)
        self.assertEqual(result["trade_count"], 1)
        self.assertEqual(result["trades"][0]["p"], 0.0)
        self.assertIn("stopped after 2 pages", result["detail"])

    @patch("equity_data_bridge_server.httpx.Client")
    def test_non_200_is_the_status_and_body_with_no_retry(self, client_cls):
        response = _response(status=403, text=SIP_DENIED)
        response.json.side_effect = AssertionError("non-200 body must be returned verbatim")
        manager, client = _client([response])
        client_cls.return_value = manager
        result = self._call("INTC", "2026-06-17T13:29:00Z", "2026-06-17T13:31:00Z", conditions="Q,O")
        self.assertEqual(result, {"status": "error", "http_status": 403, "detail": SIP_DENIED})
        self.assertEqual(client.get.call_count, 1)

    @patch("equity_data_bridge_server.httpx.Client")
    def test_recent_sip_end_is_clamped_and_iex_is_not(self, client_cls):
        manager, client = _client([_response(payload={"trades": {"PCG": []}, "next_page_token": None})])
        client_cls.return_value = manager
        result = self._call("PCG", "2026-10-08T13:30:00Z", "2026-10-08T21:40:00Z", feed="sip")
        self.assertEqual(client.get.call_args.kwargs["params"]["feed"], "sip")
        self.assertEqual(client.get.call_args.kwargs["params"]["end"], CUTOFF)
        self.assertEqual(result["end_requested"], "2026-10-08T21:40:00Z")
        self.assertEqual(result["end_effective"], CUTOFF)
        self.assertIn("15 minutes", result["note"])

        manager, client = _client([_response(payload={"trades": {"PCG": []}, "next_page_token": None})])
        client_cls.return_value = manager
        result = self._call("PCG", "2026-10-08T21:30:00Z", "2026-10-08T21:40:00Z", feed="IEX")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["feed"], "iex")
        self.assertEqual(client.get.call_args.kwargs["params"]["end"], "2026-10-08T21:40:00Z")
        self.assertNotIn("end_effective", result)

    @patch("equity_data_bridge_server.httpx.Client")
    def test_sip_window_inside_the_delay_does_not_call_alpaca(self, client_cls):
        result = self._call("PCG", "2026-10-08T21:30:00Z", "2026-10-08T21:40:00Z")
        client_cls.assert_not_called()
        self.assertEqual(result["status"], "error")
        self.assertIn("15 minutes", result["detail"])

    @patch("equity_data_bridge_server.httpx.Client")
    def test_invalid_feed_and_conditions_do_not_call_alpaca(self, client_cls):
        feed_error = self._call("PCG", "2026-07-21", "2026-07-21", feed="boats")
        self.assertEqual(feed_error["status"], "error")
        self.assertIn("boats", feed_error["detail"])
        self.assertIn("'iex'", feed_error["detail"])
        self.assertIn("'sip'", feed_error["detail"])
        conditions_error = self._call("PCG", "2026-07-21", "2026-07-21", conditions=["Q"])
        self.assertEqual(conditions_error["status"], "error")
        self.assertIn("Q,O", conditions_error["detail"])
        client_cls.assert_not_called()

    @patch("equity_data_bridge_server.httpx.Client")
    def test_date_only_end_of_today_is_clamped_for_sip(self, client_cls):
        manager, client = _client([_response(payload={"trades": {}, "next_page_token": None})])
        client_cls.return_value = manager
        result = self._call("INTC", "2026-10-08", "2026-10-08")
        self.assertEqual(client.get.call_args.kwargs["params"]["end"], CUTOFF)
        self.assertEqual(result["end_requested"], "2026-10-08")
        self.assertEqual(result["trade_count"], 0)


if __name__ == "__main__":
    unittest.main()
