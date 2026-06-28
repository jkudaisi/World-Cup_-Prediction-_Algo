"""Reusable Kalshi REST client with dry-run order protection."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any
from urllib.parse import urlencode

import requests
from requests.exceptions import RequestException

from kalshi_auth import KalshiAuthError, build_auth_headers, credentials_configured
from trading_config import get_config

log = logging.getLogger(__name__)


class KalshiClientError(Exception):
    pass


def _format_fp_count(count: int | float) -> str:
    return f"{float(count):.2f}"


def _cents_to_dollar_str(cents: int | float) -> str:
    return f"{float(cents) / 100.0:.4f}"


def _legacy_order_to_v2(
    *,
    side: str,
    action: str,
    yes_price: int | None,
    no_price: int | None,
) -> tuple[str, str]:
    """
    Map legacy yes/no + cent prices to V2 book side and dollar price.

    V2 event orders use YES-leg only: bid = buy YES, ask = sell YES.
    Buying NO at P_no is equivalent to selling YES at (1 - P_no).
    """
    side = side.lower()
    action = action.lower()
    if action != "buy":
        raise KalshiClientError(f"Unsupported order action for V2: {action}")

    if side == "yes":
        if yes_price is None:
            raise KalshiClientError("yes_price required for buy YES orders")
        return "bid", _cents_to_dollar_str(yes_price)

    if side == "no":
        if no_price is None:
            raise KalshiClientError("no_price required for buy NO orders")
        yes_equiv_cents = 100.0 - float(no_price)
        return "ask", _cents_to_dollar_str(yes_equiv_cents)

    raise KalshiClientError(f"Unsupported order side: {side}")


def _normalize_v2_order_response(resp: dict, *, ticker: str, legacy_side: str, count: int) -> dict:
    """Keep backward-compatible `order` envelope for callers."""
    order_id = resp.get("order_id")
    client_order_id = resp.get("client_order_id")
    fill_count = resp.get("fill_count")
    remaining_count = resp.get("remaining_count")
    status = "submitted"
    try:
        if fill_count is not None and float(fill_count) > 0:
            status = "filled" if float(remaining_count or 0) <= 0 else "partially_filled"
    except (TypeError, ValueError):
        pass

    order = {
        "order_id": order_id,
        "client_order_id": client_order_id,
        "ticker": ticker,
        "side": legacy_side,
        "count": count,
        "status": status,
        "fill_count": fill_count,
        "remaining_count": remaining_count,
        "average_fill_price": resp.get("average_fill_price"),
    }
    return {**resp, "order": order}


class KalshiClient:
    def __init__(self, *, base_url: str | None = None, timeout: float = 30.0):
        cfg = get_config()
        self.base_url = (base_url or cfg.kalshi_base_url).rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    def _url(self, path: str) -> str:
        if path.startswith("http"):
            return path
        if not path.startswith("/"):
            path = f"/{path}"
        return f"{self.base_url}{path}"

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict | None = None,
        json_body: dict | None = None,
        auth: bool = True,
    ) -> Any:
        query = f"?{urlencode(params)}" if params else ""
        full_path = f"{path}{query}"
        url = self._url(full_path)
        headers: dict[str, str] = {"Accept": "application/json"}
        if auth:
            if not credentials_configured():
                raise KalshiClientError("Kalshi credentials not configured")
            try:
                headers.update(build_auth_headers(method, path, base_url=self.base_url))
            except KalshiAuthError as exc:
                raise KalshiClientError(str(exc)) from exc

        last_err: Exception | None = None
        for attempt in range(3):
            try:
                resp = self._session.request(
                    method.upper(),
                    url,
                    headers=headers,
                    json=json_body,
                    timeout=self.timeout,
                )
                if resp.status_code >= 400:
                    raise KalshiClientError(
                        f"Kalshi {method} {path} failed: {resp.status_code} {resp.text[:500]}"
                    )
                if not resp.content:
                    return {}
                return resp.json()
            except RequestException as exc:
                last_err = exc
                if attempt < 2:
                    time.sleep(0.35 * (attempt + 1))
                    continue
                raise KalshiClientError(f"Kalshi {method} {path} connection failed: {exc}") from exc
        raise KalshiClientError(f"Kalshi {method} {path} failed: {last_err}")

    def get_events(self, **params) -> dict:
        return self._request("GET", "/events", params=params or None)

    def get_markets(self, **params) -> dict:
        return self._request("GET", "/markets", params=params or None)

    def get_market(self, ticker: str) -> dict:
        return self._request("GET", f"/markets/{ticker}")

    def get_orderbook(self, ticker: str, depth: int | None = None) -> dict:
        params = {"depth": depth} if depth else None
        return self._request("GET", f"/markets/{ticker}/orderbook", params=params)

    def get_balance(self, **params) -> dict:
        return self._request("GET", "/portfolio/balance", params=params or None)

    def get_positions(self, **params) -> dict:
        return self._request("GET", "/portfolio/positions", params=params or None)

    def get_orders(self, **params) -> dict:
        return self._request("GET", "/portfolio/orders", params=params or None)

    def get_fills(self, **params) -> dict:
        return self._request("GET", "/portfolio/fills", params=params or None)

    def create_order(
        self,
        ticker: str,
        *,
        side: str,
        action: str = "buy",
        count: int = 1,
        yes_price: int | None = None,
        no_price: int | None = None,
        order_type: str = "limit",
        client_order_id: str | None = None,
        time_in_force: str = "good_till_canceled",
    ) -> dict:
        cfg = get_config()
        cid = client_order_id or str(uuid.uuid4())
        book_side, price_dollars = _legacy_order_to_v2(
            side=side,
            action=action,
            yes_price=yes_price,
            no_price=no_price,
        )

        if order_type != "limit":
            log.warning("Kalshi V2 create_order only supports limit orders; got %s", order_type)

        payload = {
            "ticker": ticker,
            "client_order_id": cid,
            "side": book_side,
            "count": _format_fp_count(count),
            "price": price_dollars,
            "time_in_force": time_in_force,
            "self_trade_prevention_type": "taker_at_cross",
            "post_only": False,
            "cancel_order_on_pause": False,
            "reduce_only": False,
        }

        if cfg.dry_run:
            log.info("DRY RUN order (not sent): %s", payload)
            return {
                "order": {
                    "ticker": ticker,
                    "side": side.lower(),
                    "action": action.lower(),
                    "count": count,
                    "type": order_type,
                    "client_order_id": cid,
                    "order_id": f"dry-run-{cid}",
                    "status": "dry_run",
                    "v2_side": book_side,
                    "v2_price": price_dollars,
                },
                "dry_run": True,
            }

        resp = self._request("POST", "/portfolio/events/orders", json_body=payload)
        return _normalize_v2_order_response(
            resp,
            ticker=ticker,
            legacy_side=side.lower(),
            count=count,
        )

    def cancel_order(self, order_id: str) -> dict:
        cfg = get_config()
        if cfg.dry_run:
            log.info("DRY RUN cancel (not sent): %s", order_id)
            return {"order_id": order_id, "status": "dry_run_cancelled", "dry_run": True}
        return self._request("DELETE", f"/portfolio/events/orders/{order_id}")


def get_client() -> KalshiClient:
    return KalshiClient()
