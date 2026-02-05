"""Google OAuth authentication for multiple accounts."""

import json
import logging
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from ..config import (
    GOOGLE_ACCOUNTS,
    GOOGLE_CLIENT_ID,
    GOOGLE_CLIENT_SECRET,
    GOOGLE_EMAILS,
    GOOGLE_SCOPES,
    get_google_credentials_path,
    get_google_token_path,
)

logger = logging.getLogger(__name__)


def get_client_config() -> dict:
    """Get OAuth client configuration from environment or file."""
    # First try to load from file
    creds_path = get_google_credentials_path()
    if creds_path.exists():
        with open(creds_path) as f:
            return json.load(f)

    # Fall back to environment variables
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        raise ValueError(
            "Google OAuth credentials not found. Either set GOOGLE_CLIENT_ID and "
            "GOOGLE_CLIENT_SECRET environment variables, or create "
            f"{creds_path} with your OAuth client credentials."
        )

    return {
        "installed": {
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": ["http://localhost"],
        }
    }


def get_credentials(account: str) -> Credentials | None:
    """Load and refresh OAuth credentials for a Google account.

    Args:
        account: Account identifier (e.g., "arc", "personal").

    Returns:
        Valid credentials, or None if not authenticated.
    """
    if account not in GOOGLE_ACCOUNTS:
        raise ValueError(f"Unknown Google account: {account}")

    token_path = get_google_token_path(account)

    if not token_path.exists():
        logger.warning(f"No token file for account '{account}'. Run OAuth flow first.")
        return None

    try:
        creds = Credentials.from_authorized_user_file(str(token_path), GOOGLE_SCOPES)

        if creds.expired and creds.refresh_token:
            logger.info(f"Refreshing expired token for account '{account}'")
            creds.refresh(Request())
            # Save refreshed token
            _save_credentials(account, creds)

        if not creds.valid:
            logger.warning(f"Invalid credentials for account '{account}'")
            return None

        return creds

    except Exception as e:
        logger.error(f"Error loading credentials for account '{account}': {e}")
        return None


def run_oauth_flow(account: str, open_browser: bool = True) -> Credentials:
    """Run interactive OAuth flow for a Google account.

    Args:
        account: Account identifier.
        open_browser: Whether to automatically open browser.

    Returns:
        Authenticated credentials.
    """
    if account not in GOOGLE_ACCOUNTS:
        raise ValueError(f"Unknown Google account: {account}")

    email = GOOGLE_EMAILS.get(account, "")
    print(f"\n{'='*60}")
    print(f"OAuth Authentication for: {account}")
    print(f"Expected email: {email}")
    print(f"{'='*60}")
    print(f"\nPlease sign in with: {email}")
    print("(Make sure you select the correct account in the browser)\n")

    client_config = get_client_config()

    flow = InstalledAppFlow.from_client_config(client_config, GOOGLE_SCOPES)

    # Use a different port for each account to avoid conflicts
    port = 8080 + GOOGLE_ACCOUNTS.index(account)

    creds = flow.run_local_server(
        port=port,
        open_browser=open_browser,
        prompt="consent",
        authorization_prompt_message=f"Opening browser for {email}...",
    )

    _save_credentials(account, creds)
    logger.info(f"Successfully authenticated account '{account}'")

    return creds


def _save_credentials(account: str, creds: Credentials) -> None:
    """Save credentials to token file."""
    token_path = get_google_token_path(account)
    token_path.parent.mkdir(parents=True, exist_ok=True)

    with open(token_path, "w") as f:
        f.write(creds.to_json())

    try:
        token_path.chmod(0o600)
    except Exception:
        logger.warning(f"Failed to set permissions on {token_path}")

    logger.debug(f"Saved credentials to {token_path}")


def check_all_accounts() -> dict[str, bool]:
    """Check authentication status for all accounts.

    Returns:
        Dictionary mapping account name to authentication status.
    """
    status = {}
    for account in GOOGLE_ACCOUNTS:
        creds = get_credentials(account)
        status[account] = creds is not None and creds.valid
    return status


def revoke_credentials(account: str) -> bool:
    """Revoke and delete credentials for an account.

    Returns:
        True if credentials were deleted.
    """
    token_path = get_google_token_path(account)
    if token_path.exists():
        token_path.unlink()
        logger.info(f"Revoked credentials for account '{account}'")
        return True
    return False
