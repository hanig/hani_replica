"""Tests for configuration module."""

import os
from pathlib import Path

import pytest


def test_project_root_exists():
    """Test that project root is correctly identified."""
    from src.config import PROJECT_ROOT

    assert PROJECT_ROOT.exists()
    assert (PROJECT_ROOT / "src").exists()


def test_google_accounts_defined():
    """Test that Google accounts are properly defined."""
    from src.config import GOOGLE_ACCOUNTS, GOOGLE_EMAILS, GOOGLE_TIER1, GOOGLE_TIER2

    assert len(GOOGLE_ACCOUNTS) >= 1

    # All accounts should have emails
    for account in GOOGLE_ACCOUNTS:
        assert account in GOOGLE_EMAILS
        assert "@" in GOOGLE_EMAILS[account]

    # Tiers should partition all accounts
    assert set(GOOGLE_TIER1 + GOOGLE_TIER2) == set(GOOGLE_ACCOUNTS)


def test_get_google_token_path():
    """Test token path generation."""
    from src.config import get_google_token_path, GOOGLE_ACCOUNTS

    account = GOOGLE_ACCOUNTS[0]
    path = get_google_token_path(account)
    assert path.name == f"google_token_{account}.json"
    assert "credentials" in str(path)


def test_get_google_token_path_invalid():
    """Test that invalid account raises error."""
    from src.config import get_google_token_path

    with pytest.raises(ValueError):
        get_google_token_path("invalid_account")


def test_ensure_directories():
    """Test that ensure_directories creates required dirs."""
    from src.config import DATA_DIR, LOGS_DIR, CREDENTIALS_DIR, ensure_directories

    ensure_directories()

    assert DATA_DIR.exists()
    assert LOGS_DIR.exists()
    assert CREDENTIALS_DIR.exists()


def test_get_config():
    """Test that get_config returns expected structure."""
    from src.config import get_config

    config = get_config()

    assert "project_root" in config
    assert "google_accounts" in config
    assert "google_emails" in config
    assert "github_username" in config
    assert "embedding_model" in config


def test_validate_config():
    """Test configuration validation."""
    from src.config import validate_config

    issues = validate_config()
    # Should return a list (possibly empty or with issues)
    assert isinstance(issues, list)
