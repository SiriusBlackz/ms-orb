#!/usr/bin/env python3
"""One-time interactive OAuth flow for the cTrader Open API.

Run this ONCE on your local machine (it needs a browser); it produces the
access/refresh token pair the headless bot then refreshes on its own forever.

    python3 get_token.py

Prerequisites (see deploy/README_DEPLOY.md):
  1. A cTrader ID (id.ctrader.com) with your broker's cTrader account linked.
  2. An application registered at openapi.ctrader.com -> Credentials
     (client ID + secret). Add "http://localhost/" as a redirect URI.
  3. CTRADER_CLIENT_ID / CTRADER_CLIENT_SECRET in .env (or enter below).

Outputs the token pair and, optionally, your ctidTraderAccountId list.
"""

import json
import os
import ssl
import sys
import urllib.parse
import urllib.request

from dotenv import load_dotenv

load_dotenv()

REDIRECT_URI = "http://localhost/"
AUTH_URL = "https://id.ctrader.com/my/settings/openapi/grantingaccess/"
TOKEN_URL = "https://openapi.ctrader.com/apps/token"
ACCOUNTS_URL = "https://api.spotware.com/connect/tradingaccounts"

# python.org builds on macOS ship without root certificates; use certifi's
# bundle when it is installed so HTTPS verification works out of the box.
try:
    import certifi
    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CONTEXT = ssl.create_default_context()


def extract_code(raw: str) -> str:
    """Accept either the bare code or the whole redirected URL."""
    raw = raw.strip()
    if "code=" in raw:
        query = urllib.parse.urlparse(raw).query or raw.split("?", 1)[-1]
        values = urllib.parse.parse_qs(query).get("code")
        if values:
            return values[0]
    return raw


def main() -> None:
    client_id = os.getenv("CTRADER_CLIENT_ID") or input("Client ID: ").strip()
    client_secret = os.getenv("CTRADER_CLIENT_SECRET") or input("Client secret: ").strip()

    params = urllib.parse.urlencode({
        "client_id": client_id,
        "redirect_uri": REDIRECT_URI,
        "scope": "trading",       # required for order placement
        "product": "web",
    })
    print("\n1. Open this URL in a browser, log in with your cTrader ID, and")
    print("   grant access to the trading account(s):\n")
    print(f"   {AUTH_URL}?{params}\n")
    print("2. After granting, the browser redirects to localhost (page won't")
    print("   load — that's fine). Copy the whole address from the URL bar")
    print("   (or just the ?code=... value) and paste it below.")
    print("   NOTE: the code expires in ~60 seconds — paste it promptly.\n")
    code = extract_code(input("Authorization code or redirected URL: "))

    query = urllib.parse.urlencode({
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": REDIRECT_URI,
        "client_id": client_id,
        "client_secret": client_secret,
    })
    with urllib.request.urlopen(f"{TOKEN_URL}?{query}", context=SSL_CONTEXT) as resp:
        tokens = json.loads(resp.read())

    if "accessToken" not in tokens:
        print(f"\nToken exchange failed: {tokens}")
        sys.exit(1)

    print("\n=== SUCCESS — add these to the bot's .env ===\n")
    print(f"CTRADER_ACCESS_TOKEN={tokens['accessToken']}")
    print(f"CTRADER_REFRESH_TOKEN={tokens['refreshToken']}")
    print(f"\n(access token expires in {int(tokens.get('expiresIn', 0)) // 86400} days; "
          f"the bot refreshes and persists rotations automatically)")

    # Best-effort: list trading accounts so the user can grab ctidTraderAccountId
    try:
        q = urllib.parse.urlencode({"access_token": tokens["accessToken"]})
        with urllib.request.urlopen(f"{ACCOUNTS_URL}?{q}", context=SSL_CONTEXT) as resp:
            data = json.loads(resp.read())
        print("\nTrading accounts on this token:")
        for acc in data.get("data", []):
            print(f"  ctidTraderAccountId={acc.get('traderAccountId') or acc.get('accountId')}  "
                  f"broker={acc.get('brokerTitle') or acc.get('brokerName')}  "
                  f"live={acc.get('live')}  number={acc.get('accountNumber')}")
        print("\nUse the DEMO account's ctidTraderAccountId as CTRADER_ACCOUNT_ID.")
    except Exception as e:
        print(f"\n(Could not auto-list accounts: {e} — find the ctidTraderAccountId")
        print(" via the Open API playground or ProtoOAGetAccountListByAccessTokenReq.)")


if __name__ == "__main__":
    main()
