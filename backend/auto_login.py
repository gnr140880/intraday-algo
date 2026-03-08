"""
Automatic Kite Connect token refresh via programmatic login.

Zerodha access tokens expire daily at ~6:00 AM IST.
This module automates the login + TOTP 2FA flow using plain HTTP
requests (no browser/Selenium required).

Required .env variables:
    KITE_USER_ID      – Zerodha client ID  (e.g. "AB1234")
    KITE_PASSWORD     – Zerodha login password
    KITE_TOTP_SECRET  – Base32 TOTP secret from Zerodha 2FA setup
"""

import logging
import time
import pyotp
import requests
from typing import Optional, Dict
from config import settings

logger = logging.getLogger(__name__)

# Zerodha login endpoints
LOGIN_URL = "https://kite.zerodha.com/api/login"
TWOFA_URL = "https://kite.zerodha.com/api/twofa"


def _generate_totp(secret: str) -> str:
    """Generate a time-based OTP from the TOTP secret."""
    totp = pyotp.TOTP(secret)
    return totp.now()


def auto_login() -> Optional[str]:
    """
    Perform Zerodha login + TOTP programmatically.

    Returns:
        request_token (str) on success, None on failure.
    """
    user_id = settings.kite_user_id
    password = settings.kite_password
    totp_secret = settings.kite_totp_secret
    api_key = settings.kite_api_key

    if not all([user_id, password, totp_secret, api_key]):
        logger.error(
            "Auto-login requires KITE_USER_ID, KITE_PASSWORD, "
            "KITE_TOTP_SECRET, and KITE_API_KEY in .env"
        )
        return None

    session = requests.Session()

    try:
        # Step 1 – POST user_id + password
        logger.info("Auto-login: sending credentials for %s …", user_id)
        resp = session.post(
            LOGIN_URL,
            data={"user_id": user_id, "password": password},
        )
        resp.raise_for_status()
        login_data = resp.json()

        if login_data.get("status") != "success":
            logger.error("Auto-login step-1 failed: %s", login_data)
            return None

        request_id = login_data["data"]["request_id"]

        # Step 2 – POST TOTP for two-factor auth
        totp_value = _generate_totp(totp_secret)
        logger.info("Auto-login: submitting TOTP …")
        resp = session.post(
            TWOFA_URL,
            data={
                "user_id": user_id,
                "request_id": request_id,
                "twofa_value": totp_value,
                "twofa_type": "totp",
            },
        )
        resp.raise_for_status()
        twofa_data = resp.json()

        if twofa_data.get("status") != "success":
            logger.error("Auto-login step-2 (TOTP) failed: %s", twofa_data)
            return None

        # Step 3 – Hit the Kite Connect redirect to get request_token
        redirect_url = (
            f"https://kite.trade/connect/login?api_key={api_key}&v=3"
        )
        logger.info("Auto-login: fetching request_token via redirect …")
        resp = session.get(redirect_url, allow_redirects=False)

        # Zerodha redirects to: <callback_url>?request_token=xxx&action=login&status=success
        location = resp.headers.get("Location", "")

        if "request_token=" not in location:
            # Sometimes it needs one more redirect
            if resp.status_code in (200, 302, 303):
                resp = session.get(redirect_url, allow_redirects=True)
                location = resp.url

        if "request_token=" not in str(location):
            logger.error(
                "Auto-login: could not extract request_token. "
                "Location header: %s",
                location,
            )
            return None

        # Parse request_token from URL
        from urllib.parse import urlparse, parse_qs

        parsed = urlparse(str(location))
        params = parse_qs(parsed.query)
        request_token = params.get("request_token", [None])[0]

        if not request_token:
            logger.error("Auto-login: request_token is empty in redirect URL")
            return None

        logger.info("Auto-login: got request_token successfully")
        return request_token

    except requests.exceptions.HTTPError as e:
        logger.error("Auto-login HTTP error: %s", e)
        return None
    except Exception as e:
        logger.error("Auto-login unexpected error: %s", e)
        return None


def refresh_access_token(kite_client) -> Dict:
    """
    Full token refresh: auto-login → generate_session → persist.

    Args:
        kite_client: KiteClient instance

    Returns:
        dict with 'success' key
    """
    logger.info("Starting automatic token refresh …")

    request_token = auto_login()
    if not request_token:
        return {"success": False, "error": "Auto-login failed — check credentials/TOTP secret"}

    result = kite_client.generate_session(request_token)
    if result.get("success"):
        logger.info("Token refreshed successfully via auto-login")
    else:
        logger.error("generate_session failed after auto-login: %s", result)

    return result


def is_token_valid(kite_client) -> bool:
    """Quick check whether the current access token is still valid."""
    try:
        kite_client.kite.profile()
        return True
    except Exception:
        return False
