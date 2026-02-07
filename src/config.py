"""Configuration management for Engram."""

import json
import os
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Load environment variables from .env file
load_dotenv(PROJECT_ROOT / ".env")


def get_env(key: str, default: str | None = None, required: bool = False) -> str | None:
    """Get environment variable with optional default and required check."""
    value = os.getenv(key, default)
    if required and value is None:
        raise ValueError(f"Required environment variable {key} is not set")
    return value


def get_env_list(key: str, default: list[str] | None = None) -> list[str]:
    """Get environment variable as a list (comma-separated)."""
    value = os.getenv(key)
    if not value:
        return default or []
    return [item.strip() for item in value.split(",") if item.strip()]


def get_env_dict(key: str, default: dict[str, str] | None = None) -> dict[str, str]:
    """Get environment variable as a dictionary (JSON format)."""
    value = os.getenv(key)
    if not value:
        return default or {}
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        # Try key=value,key=value format
        result = {}
        for pair in value.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                result[k.strip()] = v.strip()
        return result


# Google Account Configuration
# GOOGLE_ACCOUNTS: comma-separated list of account names (e.g., "work,personal")
# GOOGLE_EMAILS: JSON dict mapping account names to emails
#   e.g., '{"work": "user@company.com", "personal": "user@gmail.com"}'
GOOGLE_ACCOUNTS = get_env_list("GOOGLE_ACCOUNTS")
GOOGLE_EMAILS = get_env_dict("GOOGLE_EMAILS")

# Tiered search configuration - Tier 1 accounts are searched first
# GOOGLE_TIER1: comma-separated list of primary accounts
# GOOGLE_TIER2: comma-separated list of secondary accounts
GOOGLE_TIER1 = get_env_list("GOOGLE_TIER1")
GOOGLE_TIER2 = get_env_list("GOOGLE_TIER2")

# Google OAuth credentials
GOOGLE_CLIENT_ID = get_env("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = get_env("GOOGLE_CLIENT_SECRET")

# Google API scopes
GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",  # For sending emails
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/drive.file",  # For Docs comments on files created/opened by app
    "https://www.googleapis.com/auth/calendar",  # Full calendar access (read/write)
    "https://www.googleapis.com/auth/documents",  # For Docs API access
]


def get_google_token_path(account: str) -> Path:
    """Get the path to the OAuth token file for a Google account."""
    if account not in GOOGLE_ACCOUNTS:
        raise ValueError(f"Unknown Google account: {account}")
    return PROJECT_ROOT / "credentials" / f"google_token_{account}.json"


def get_google_credentials_path() -> Path:
    """Get the path to the Google OAuth client credentials file."""
    return PROJECT_ROOT / "credentials" / "google_client_secret.json"


# GitHub Configuration
GITHUB_TOKEN = get_env("GITHUB_TOKEN")
GITHUB_USERNAME = get_env("GITHUB_USERNAME")
GITHUB_ORG = get_env("GITHUB_ORG")

# Slack Configuration
SLACK_BOT_TOKEN = get_env("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = get_env("SLACK_APP_TOKEN")
SLACK_WORKSPACE = get_env("SLACK_WORKSPACE")

# Notion Configuration
NOTION_API_KEY = get_env("NOTION_API_KEY")
NOTION_WORKSPACE = get_env("NOTION_WORKSPACE", "default")

# Todoist Configuration
TODOIST_API_KEY = get_env("TODOIST_API_KEY")

# Zotero Configuration
ZOTERO_API_KEY = get_env("ZOTERO_API_KEY")
ZOTERO_USER_ID = get_env("ZOTERO_USER_ID")
ZOTERO_LIBRARY_TYPE = get_env("ZOTERO_LIBRARY_TYPE", "user")
ZOTERO_DEFAULT_COLLECTION = get_env("ZOTERO_DEFAULT_COLLECTION", "")

# Default account for create/send operations (first tier1 account)
PRIMARY_ACCOUNT = GOOGLE_TIER1[0] if GOOGLE_TIER1 else (GOOGLE_ACCOUNTS[0] if GOOGLE_ACCOUNTS else "default")


def get_accounts_description() -> str:
    """Build dynamic account list string for tool descriptions."""
    return f"Account name ({', '.join(GOOGLE_ACCOUNTS)})" if GOOGLE_ACCOUNTS else "Account name"


# Authorized Slack users (comma-separated user IDs)
_authorized_users = get_env("SLACK_AUTHORIZED_USERS", "")
SLACK_AUTHORIZED_USERS = [u.strip() for u in _authorized_users.split(",") if u.strip()]
SLACK_ALLOW_ALL_USERS = get_env("SLACK_ALLOW_ALL_USERS", "false").lower() in ("true", "1", "yes")

# OpenAI Configuration
OPENAI_API_KEY = get_env("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 3072  # text-embedding-3-large default

# Anthropic Configuration
ANTHROPIC_API_KEY = get_env("ANTHROPIC_API_KEY")
INTENT_MODEL = "claude-3-haiku-20240307"
AGENT_MODEL = get_env("AGENT_MODEL", "claude-sonnet-4-20250514")

# Bot mode:
# - "intent" for legacy intent routing with handlers
# - "agent" for single agent with tool calling
# - "multi_agent" for orchestrator with specialist agents
BOT_MODE = get_env("BOT_MODE", "agent")

# Enable streaming responses (applies to agent and multi_agent modes)
ENABLE_STREAMING = get_env("ENABLE_STREAMING", "true").lower() in ("true", "1", "yes")

# Allow direct email sending from tools (requires explicit Slack confirmation flow)
ENABLE_DIRECT_EMAIL_SEND = get_env("ENABLE_DIRECT_EMAIL_SEND", "false").lower() in ("true", "1", "yes")

# User timezone (IANA name, e.g., "America/Los_Angeles")
USER_TIMEZONE = get_env("USER_TIMEZONE", "America/Los_Angeles")

# Minimum interval between Slack message updates (in seconds) to avoid rate limiting
STREAMING_UPDATE_INTERVAL = float(get_env("STREAMING_UPDATE_INTERVAL", "0.5"))

# Database paths
KNOWLEDGE_GRAPH_DB = PROJECT_ROOT / get_env("KNOWLEDGE_GRAPH_DB", "data/knowledge_graph.db")
CHROMA_DB_PATH = PROJECT_ROOT / get_env("CHROMA_DB_PATH", "data/chroma")
MEM0_CHROMA_PATH = PROJECT_ROOT / get_env("MEM0_CHROMA_PATH", "data/mem0_chroma")

# Logging configuration
LOG_LEVEL = get_env("LOG_LEVEL", "INFO")
LOG_FILE = PROJECT_ROOT / get_env("LOG_FILE", "logs/engram.log")

# Sync settings
SYNC_BATCH_SIZE = int(get_env("SYNC_BATCH_SIZE", "100"))
EMBEDDING_BATCH_SIZE = int(get_env("EMBEDDING_BATCH_SIZE", "50"))

# Security settings
# Level: "strict" (block suspicious), "moderate" (warn), "permissive" (log only)
SECURITY_LEVEL = get_env("SECURITY_LEVEL", "moderate")

# Rate limiting
RATE_LIMIT_REQUESTS = int(get_env("RATE_LIMIT_REQUESTS", "30"))  # Max requests per window
RATE_LIMIT_WINDOW = int(get_env("RATE_LIMIT_WINDOW", "60"))  # Window in seconds
RATE_LIMIT_BLOCK_DURATION = int(get_env("RATE_LIMIT_BLOCK_DURATION", "300"))  # Block duration

# Audit logging
ENABLE_AUDIT_LOG = get_env("ENABLE_AUDIT_LOG", "true").lower() in ("true", "1", "yes")
AUDIT_LOG_PATH = PROJECT_ROOT / get_env("AUDIT_LOG_PATH", "data/audit.db")
AUDIT_RETENTION_DAYS = int(get_env("AUDIT_RETENTION_DAYS", "90"))
# Whether to store raw message text in audit logs
AUDIT_LOG_MESSAGES = get_env("AUDIT_LOG_MESSAGES", "false").lower() in ("true", "1", "yes")

# Ensure required directories exist
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
CREDENTIALS_DIR = PROJECT_ROOT / "credentials"


def ensure_directories() -> None:
    """Create required directories if they don't exist."""
    for directory in [DATA_DIR, LOGS_DIR, CREDENTIALS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def get_user_timezone() -> ZoneInfo:
    """Get configured user timezone, defaulting to UTC if invalid."""
    try:
        return ZoneInfo(USER_TIMEZONE)
    except Exception:
        return ZoneInfo("UTC")


def get_config() -> dict[str, Any]:
    """Return all configuration as a dictionary (excluding secrets)."""
    return {
        "project_root": str(PROJECT_ROOT),
        "google_accounts": GOOGLE_ACCOUNTS,
        "google_emails": GOOGLE_EMAILS,
        "google_tier1": GOOGLE_TIER1,
        "google_tier2": GOOGLE_TIER2,
        "github_username": GITHUB_USERNAME,
        "github_org": GITHUB_ORG,
        "slack_workspace": SLACK_WORKSPACE,
        "slack_allow_all_users": SLACK_ALLOW_ALL_USERS,
        "embedding_model": EMBEDDING_MODEL,
        "intent_model": INTENT_MODEL,
        "agent_model": AGENT_MODEL,
        "bot_mode": BOT_MODE,
        "enable_streaming": ENABLE_STREAMING,
        "enable_direct_email_send": ENABLE_DIRECT_EMAIL_SEND,
        "user_timezone": USER_TIMEZONE,
        "streaming_update_interval": STREAMING_UPDATE_INTERVAL,
        "knowledge_graph_db": str(KNOWLEDGE_GRAPH_DB),
        "chroma_db_path": str(CHROMA_DB_PATH),
        "log_level": LOG_LEVEL,
        "sync_batch_size": SYNC_BATCH_SIZE,
        "embedding_batch_size": EMBEDDING_BATCH_SIZE,
        "security_level": SECURITY_LEVEL,
        "rate_limit_requests": RATE_LIMIT_REQUESTS,
        "rate_limit_window": RATE_LIMIT_WINDOW,
        "enable_audit_log": ENABLE_AUDIT_LOG,
        "audit_log_messages": AUDIT_LOG_MESSAGES,
        "audit_retention_days": AUDIT_RETENTION_DAYS,
    }


def validate_config() -> list[str]:
    """Validate configuration and return list of missing/invalid items."""
    issues = []

    # Check Google credentials
    if not GOOGLE_CLIENT_ID:
        issues.append("GOOGLE_CLIENT_ID not set")
    if not GOOGLE_CLIENT_SECRET:
        issues.append("GOOGLE_CLIENT_SECRET not set")

    # Check GitHub token
    if not GITHUB_TOKEN:
        issues.append("GITHUB_TOKEN not set")

    # Check Slack tokens
    if not SLACK_BOT_TOKEN:
        issues.append("SLACK_BOT_TOKEN not set")
    if not SLACK_APP_TOKEN:
        issues.append("SLACK_APP_TOKEN not set")

    # Check AI API keys
    if not OPENAI_API_KEY:
        issues.append("OPENAI_API_KEY not set")
    if not ANTHROPIC_API_KEY:
        issues.append("ANTHROPIC_API_KEY not set")

    # Check authorized users
    if not SLACK_AUTHORIZED_USERS:
        issues.append("SLACK_AUTHORIZED_USERS not set (bot will reject all users)")

    return issues


# Initialize directories on import
ensure_directories()
