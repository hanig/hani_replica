"""Tests for Slack bot app startup helpers."""

import socket
from urllib.error import URLError

from src.bot.app import _is_transient_startup_error


def test_transient_startup_error_detects_url_error():
    """URLError is retried during startup."""
    assert _is_transient_startup_error(URLError(socket.gaierror(8, "dns")))


def test_transient_startup_error_detects_nested_url_error():
    """Wrapped URLError is retried during startup."""
    error = RuntimeError("startup failed")
    error.__cause__ = URLError(socket.gaierror(8, "dns"))

    assert _is_transient_startup_error(error)


def test_transient_startup_error_rejects_config_error():
    """Config errors should fail fast."""
    assert not _is_transient_startup_error(ValueError("SLACK_BOT_TOKEN is required"))
