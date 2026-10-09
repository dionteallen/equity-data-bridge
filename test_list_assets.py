"""Stdlib checks for the read-only list_assets tool. No network, no extra packages."""

import inspect
import json
import os
import subprocess
import sys
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

os.environ.setdefault("ALPACA_API_KEY_ID", "test-key")
os.environ.setdefault("ALPACA_API_SECRET_KEY", "test-secret")

import equity_data_bridge_server as bridge


NOW = datetime(2026, 10, 8, 21, 38, 0, tzinfo=timezone.utc)
PAPER_BASE = "https://paper-api.alpaca.markets/v2"
LIVE_BASE = "https://api.alpaca.markets/v2"
EXISTING_TOOLS = {
    "get_historical_bars": ["symbol", "timeframe", "start", "end", "feed"],
    "get_latest_quote": ["symbol"],
    "get_fama_french_5factors": ["start_date", "end_date"],
    "get_opening_auctions": ["symbols", "start", "end", "feed", "include_closing"],
    "get_trades": ["symbol", "start", "end", "feed", "conditions"],
}


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
    if json_error is not None:
        response.json.side_effect = json_error
    else:
        response.json.return_value = payload
    if text is None:
        response.text = json.dumps(payload)
    else:
        response.text = text
    return response


def _client(response):
    client = MagicMock()
    client.get.return_value = response
    manager = MagicMock()
    manager.__enter__.return_value = client
    manager.__exit__.return_value = False
    return manager, client


def _asset(**overrides):
    row = {
        "id": "b0b6dd9d-8b9b-48a9-ba46-b9d54906e415",
        "class": "us_equity",
        "exchange": "NASDAQ",
        "symbol": "AAPL",
        "name": "Apple Inc. Common Stock",
        "status": "active",
        "tradable": True,
        "marginable": True,
        "shortable": True,
        "easy_to_borrow": True,
        "fractionable": True,
        "maintenance_margin_requirement": 30,
        "attributes": ["fractional_eh_enabled"],
    }
    row.update(overrides)
    return row


class ListAssetsContractTests(unittest.TestCase):
    def test_tool_is_registered_and_existing_signatures_are_unchanged(self):
        tools = {tool.name: tool for tool in bridge.mcp._tool_manager.list_tools()}
        self.assertEqual(
            sorted(tools),
            [
                "get_fama_french_5factors",
                "get_historical_bars",
                "get_latest_quote",
                "get_opening_auctions",
                "get_trades",
                "list_assets",
            ],
        )
        for name, parameters in EXISTING_TOOLS.items():
            self.assertEqual(list(inspect.signature(getattr(bridge, name)).parameters), parameters)
        signature = inspect.signature(bridge.list_assets)
        self.assertEqual(list(signature.parameters), ["status", "asset_class", "exchange"])
        self.assertEqual(signature.parameters["status"].default, "active")
        self.assertEqual(signature.parameters["asset_class"].default, "us_equity")
        self.assertIsNone(signature.parameters["exchange"].default)
        schema = tools["list_assets"].parameters
        self.assertEqual(schema["properties"]["status"]["default"], "active")
        self.assertEqual(schema["properties"]["asset_class"]["default"], "us_equity")
        self.assertNotIn("status", schema.get("required", []))

    def test_description_keeps_the_caveats_and_holds_inactive_back(self):
        tool = {item.name: item for item in bridge.mcp._tool_manager.list_tools()}["list_assets"]
        description = tool.description
        flat = " ".join(description.split())
        self.assertIn(
            "CAVEATS (state them in every consumer): this is a CURRENT snapshot. "
            "It has no listing date, no delisting date and no ticker history. "
            "A reused ticker shows only the current holder.",
            flat,
        )
        self.assertIn("not to be used yet", description)
        self.assertIn("Read-only", description)
        self.assertIn("cannot legitimately be empty", flat)


class ListAssetsRequestTests(unittest.TestCase):
    def _call(self, *args, **kwargs):
        with patch("equity_data_bridge_server.datetime", _FrozenDatetime):
            return bridge.list_assets(*args, **kwargs)

    @patch("equity_data_bridge_server.httpx.Client")
    def test_default_status_active_is_sent_with_headers_and_the_paper_base(self, client_cls):
        apple = _asset()
        manager, client = _client(_response(payload=[apple]))
        client_cls.return_value = manager

        result = self._call()

        client_cls.assert_called_once_with(timeout=60)
        self.assertEqual(client.get.call_args.args[0], f"{PAPER_BASE}/assets")
        self.assertIs(client.get.call_args.kwargs["headers"], bridge.ALPACA_HEADERS)
        self.assertEqual(
            client.get.call_args.kwargs["params"],
            {"status": "active", "asset_class": "us_equity"},
        )
        self.assertEqual(bridge.ALPACA_TRADING_BASE, PAPER_BASE)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["base"], PAPER_BASE)
        self.assertEqual(result["params"], {"status": "active", "asset_class": "us_equity"})
        self.assertEqual(result["fetched_at"], "2026-10-08T21:38:00+00:00")

    @patch("equity_data_bridge_server.httpx.Client")
    def test_inactive_is_accepted_and_sent(self, client_cls):
        row = _asset(status="inactive", symbol="OLD")
        manager, client = _client(_response(payload=[row]))
        client_cls.return_value = manager

        result = self._call(status=" Inactive ")

        self.assertEqual(result["status"], "ok")
        self.assertEqual(client.get.call_args.kwargs["params"]["status"], "inactive")
        self.assertEqual(result["params"]["status"], "inactive")
        self.assertIs(result["assets"][0], row)

    @patch("equity_data_bridge_server.httpx.Client")
    def test_bad_status_is_an_error_with_no_http(self, client_cls):
        for status in ("all", "pending", "", "   ", "active,inactive", None, 1):
            with self.subTest(status=status):
                result = bridge.list_assets(status=status)
                self.assertEqual(result["status"], "error")
                self.assertIn("active", result["detail"])
                self.assertIn("inactive", result["detail"])
                self.assertNotIn("assets", result)
                self.assertNotIn("http_status", result)
                self.assertNotIn("asset_count", result)
        client_cls.assert_not_called()

    @patch("equity_data_bridge_server.httpx.Client")
    def test_bad_exchange_is_an_error_with_no_http(self, client_cls):
        for exchange in ("LSE", "sip", "NYSE,NASDAQ", "   ", 1):
            with self.subTest(exchange=exchange):
                result = bridge.list_assets(exchange=exchange)
                self.assertEqual(result["status"], "error")
                self.assertIn("NASDAQ", result["detail"])
                self.assertNotIn("assets", result)
                self.assertNotIn("http_status", result)
        client_cls.assert_not_called()

    @patch("equity_data_bridge_server.httpx.Client")
    def test_valid_exchange_is_normalized_and_sent(self, client_cls):
        row = _asset(exchange="NYSE", symbol="IBM")
        manager, client = _client(_response(payload=[row]))
        client_cls.return_value = manager

        result = self._call(exchange=" nyse ")

        self.assertEqual(
            client.get.call_args.kwargs["params"],
            {"status": "active", "asset_class": "us_equity", "exchange": "NYSE"},
        )
        self.assertEqual(result["params"]["exchange"], "NYSE")
        self.assertEqual(result["status"], "ok")

    @patch("equity_data_bridge_server.httpx.Client")
    def test_200_list_passes_through_unchanged_with_its_count(self, client_cls):
        apple = _asset()
        other = _asset(
            id="extra-1",
            symbol="MSFT",
            tradable=False,
            attributes=[],
            unexpected_field={"nested": True},
        )
        payload = [apple, other]
        manager, client = _client(_response(payload=payload))
        client_cls.return_value = manager

        result = self._call()

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["asset_count"], 2)
        self.assertIs(result["assets"], payload)
        self.assertIs(result["assets"][0], apple)
        self.assertIs(result["assets"][1], other)
        self.assertEqual(result["assets"][0]["symbol"], "AAPL")
        self.assertEqual(result["assets"][0]["exchange"], "NASDAQ")
        self.assertIs(result["assets"][0]["tradable"], True)
        self.assertEqual(result["assets"][0]["maintenance_margin_requirement"], 30)
        self.assertEqual(result["assets"][1]["unexpected_field"], {"nested": True})
        self.assertIs(result["assets"][1]["tradable"], False)
        encoded = json.dumps(result)
        self.assertNotIn(bridge.ALPACA_API_KEY_ID, encoded)
        self.assertNotIn(bridge.ALPACA_API_SECRET_KEY, encoded)

    @patch("equity_data_bridge_server.httpx.Client")
    def test_non_200_401_and_403_return_the_body_without_assets(self, client_cls):
        bodies = {
            401: '{"code":40110000,"message":"access key verification failed"}',
            403: '{"message":"forbidden"}',
            422: '{"message":"invalid"}',
            500: "upstream broke",
        }
        for status, body in bodies.items():
            with self.subTest(status=status):
                response = _response(status=status, text=body)
                response.json.side_effect = AssertionError("non-200 body must be returned verbatim")
                manager, client = _client(response)
                client_cls.return_value = manager
                result = bridge.list_assets()
                self.assertEqual(result, {"status": "error", "http_status": status, "detail": body})
                self.assertNotIn("assets", result)
                self.assertEqual(client.get.call_count, 1)

    @patch("equity_data_bridge_server.httpx.Client")
    def test_null_body_is_an_error_without_assets(self, client_cls):
        manager, _client_obj = _client(_response(payload=None, text="null"))
        client_cls.return_value = manager

        result = bridge.list_assets()

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["http_status"], 200)
        self.assertIn("null", result["detail"])
        self.assertNotIn("assets", result)
        self.assertNotIn("asset_count", result)

    @patch("equity_data_bridge_server.httpx.Client")
    def test_empty_list_is_an_error_without_assets(self, client_cls):
        manager, _client_obj = _client(_response(payload=[], text="[]"))
        client_cls.return_value = manager

        result = bridge.list_assets()

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["http_status"], 200)
        self.assertIn("empty", result["detail"])
        self.assertIn("cannot legitimately be empty", result["detail"])
        self.assertNotIn("assets", result)
        self.assertNotIn("asset_count", result)
        self.assertNotEqual(result.get("assets"), [])

    @patch("equity_data_bridge_server.httpx.Client")
    def test_bad_json_is_an_error_without_assets(self, client_cls):
        body = "not-json"
        response = _response(text=body, json_error=json.JSONDecodeError("Expecting value", body, 0))
        manager, _client_obj = _client(response)
        client_cls.return_value = manager

        result = bridge.list_assets()

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["http_status"], 200)
        self.assertIn(body, result["detail"])
        self.assertIn("not valid JSON", result["detail"])
        self.assertNotIn("assets", result)
        self.assertNotIn("asset_count", result)

    @patch("equity_data_bridge_server.httpx.Client")
    def test_non_list_body_is_an_error_without_assets(self, client_cls):
        payload = {"symbol": "AAPL", "tradable": True}
        manager, _client_obj = _client(_response(payload=payload))
        client_cls.return_value = manager

        result = bridge.list_assets()

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["http_status"], 200)
        self.assertIn("not a list", result["detail"])
        self.assertNotIn("assets", result)
        self.assertNotIn("asset_count", result)
        self.assertNotIn("symbol", result)

    @patch("equity_data_bridge_server.httpx.Client")
    def test_url_uses_the_base_env_var(self, client_cls):
        self.assertNotIn("ALPACA_TRADING_BASE", os.environ)
        self.assertEqual(bridge.ALPACA_TRADING_BASE, PAPER_BASE)
        source = inspect.getsource(bridge)
        self.assertIn(
            'ALPACA_TRADING_BASE = os.environ.get("ALPACA_TRADING_BASE", "https://paper-api.alpaca.markets/v2")',
            source,
        )

        manager, client = _client(_response(payload=[_asset()]))
        client_cls.return_value = manager
        with patch.object(bridge, "ALPACA_TRADING_BASE", LIVE_BASE):
            result = self._call()
        self.assertEqual(client.get.call_args.args[0], f"{LIVE_BASE}/assets")
        self.assertIs(client.get.call_args.kwargs["headers"], bridge.ALPACA_HEADERS)
        self.assertEqual(result["base"], LIVE_BASE)
        self.assertEqual(result["status"], "ok")

        script = (
            "import os\n"
            "os.environ['ALPACA_API_KEY_ID'] = 'k'\n"
            "os.environ['ALPACA_API_SECRET_KEY'] = 's'\n"
            "os.environ['ALPACA_TRADING_BASE'] = 'https://api.alpaca.markets/v2'\n"
            "import equity_data_bridge_server as loaded\n"
            "print(loaded.ALPACA_TRADING_BASE)\n"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=os.path.dirname(os.path.abspath(bridge.__file__)),
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), LIVE_BASE)


if __name__ == "__main__":
    unittest.main()
