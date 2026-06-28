"""Kalshi API authentication — RSA-PSS signed request headers."""

from __future__ import annotations

import base64
import logging
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from trading_config import get_config

log = logging.getLogger(__name__)

_private_key: rsa.RSAPrivateKey | None = None
_auth_available: bool | None = None


class KalshiAuthError(Exception):
    """Raised when Kalshi credentials are missing or invalid."""


def _load_private_key(path: str) -> rsa.RSAPrivateKey:
    pem_path = Path(path)
    if not pem_path.exists():
        raise KalshiAuthError(f"Private key not found: {path}")
    pem_bytes = pem_path.read_bytes()
    key = serialization.load_pem_private_key(pem_bytes, password=None, backend=default_backend())
    if not isinstance(key, rsa.RSAPrivateKey):
        raise KalshiAuthError("Private key must be RSA")
    return key


def credentials_configured() -> bool:
    cfg = get_config()
    return bool(cfg.kalshi_api_key_id and cfg.kalshi_private_key_path)


def get_private_key() -> rsa.RSAPrivateKey | None:
    global _private_key, _auth_available
    if not credentials_configured():
        _auth_available = False
        return None
    if _private_key is not None:
        return _private_key
    try:
        cfg = get_config()
        _private_key = _load_private_key(cfg.kalshi_private_key_path)
        _auth_available = True
        return _private_key
    except KalshiAuthError as exc:
        log.warning("Kalshi auth unavailable: %s", exc)
        _auth_available = False
        return None


def is_auth_available() -> bool:
    if _auth_available is not None:
        return _auth_available
    return get_private_key() is not None


def sign_message(private_key: rsa.RSAPrivateKey, message: str) -> str:
    signature = private_key.sign(
        message.encode("utf-8"),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(signature).decode("utf-8")


def signing_path(base_url: str, path: str) -> str:
    """Full API path used in signature (no query string)."""
    if path.startswith("http"):
        return urlparse(path).path.split("?")[0]
    base_path = urlparse(base_url).path.rstrip("/")
    rel = path if path.startswith("/") else f"/{path}"
    rel = rel.split("?")[0]
    if base_path and not rel.startswith(base_path):
        return f"{base_path}{rel}"
    return rel


def build_auth_headers(method: str, path: str, *, base_url: str | None = None) -> dict[str, str]:
    """Return signed headers or raise KalshiAuthError if credentials missing."""
    cfg = get_config()
    key = get_private_key()
    if key is None or not cfg.kalshi_api_key_id:
        raise KalshiAuthError(
            "Kalshi credentials not configured. Set KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH."
        )
    url_base = base_url or cfg.kalshi_base_url
    timestamp = str(int(time.time() * 1000))
    sign_path = signing_path(url_base, path)
    message = f"{timestamp}{method.upper()}{sign_path}"
    signature = sign_message(key, message)
    return {
        "KALSHI-ACCESS-KEY": cfg.kalshi_api_key_id,
        "KALSHI-ACCESS-TIMESTAMP": timestamp,
        "KALSHI-ACCESS-SIGNATURE": signature,
        "Content-Type": "application/json",
    }


def safe_auth_headers(method: str, path: str) -> dict[str, str] | None:
    """Return headers or None if credentials unavailable (fail-safe)."""
    try:
        return build_auth_headers(method, path)
    except KalshiAuthError:
        return None


def auth_status() -> dict[str, Any]:
    cfg = get_config()
    return {
        "configured": credentials_configured(),
        "available": is_auth_available(),
        "dry_run": cfg.dry_run,
        "api_key_id_set": bool(cfg.kalshi_api_key_id),
        "private_key_path": cfg.kalshi_private_key_path or None,
        "base_url": cfg.kalshi_base_url,
    }
