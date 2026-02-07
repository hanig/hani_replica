"""Tests for action confirmation handling in event handlers."""

from unittest.mock import MagicMock, patch

from src.bot.conversation import ConversationManager
from src.bot.event_handlers import register_event_handlers
from src.bot.security import SecurityGuard


class _FakeApp:
    """Minimal Slack App test double for handler registration."""

    def __init__(self):
        self.action_handlers = {}
        self.client = MagicMock()
        self.client.auth_test.return_value = {"user_id": "B123"}

    def event(self, _name):
        def decorator(func):
            return func
        return decorator

    def action(self, pattern):
        def decorator(func):
            key = getattr(pattern, "pattern", str(pattern))
            self.action_handlers[key] = func
            return func
        return decorator

    def error(self, func):
        return func


class _PendingActionStub:
    """Minimal pending action for confirmation tests."""

    def __init__(self, action_id: str, expired: bool = False):
        self.action_id = action_id
        self._expired = expired
        self.executed = False

    def is_expired(self) -> bool:
        return self._expired

    def get_action_type(self) -> str:
        return "Create Email Draft"

    def execute(self):
        self.executed = True
        return {"message": "ok"}


def _confirm_handler(app: _FakeApp):
    return app.action_handlers["^confirm_action:.*"]


def test_confirm_rejects_mismatched_action_id():
    app = _FakeApp()
    manager = ConversationManager(persist=False)
    context = manager.get_or_create("U123", "C123")
    pending = _PendingActionStub(action_id="expected123")
    context.pending_action = pending

    audit_logger = MagicMock()
    guard = SecurityGuard()
    with patch("src.bot.event_handlers.get_audit_logger", return_value=audit_logger), patch(
        "src.bot.event_handlers.get_security_guard", return_value=guard
    ):
        register_event_handlers(app, manager, mode="intent", enable_streaming=False)

    handler = _confirm_handler(app)
    client = MagicMock()
    body = {
        "user": {"id": "U123"},
        "actions": [{"action_id": "confirm_action:wrong999"}],
        "channel": {"id": "C123"},
        "message": {"ts": "123.456"},
    }

    handler(lambda: None, body, client)

    assert pending.executed is False
    assert context.pending_action is None
    client.chat_update.assert_called_once()
    assert "no longer valid" in client.chat_update.call_args.kwargs["text"].lower()
    audit_logger.log_security_event.assert_called()


def test_confirm_rejects_expired_action():
    app = _FakeApp()
    manager = ConversationManager(persist=False)
    context = manager.get_or_create("U123", "C123")
    pending = _PendingActionStub(action_id="expected123", expired=True)
    context.pending_action = pending

    audit_logger = MagicMock()
    guard = SecurityGuard()
    with patch("src.bot.event_handlers.get_audit_logger", return_value=audit_logger), patch(
        "src.bot.event_handlers.get_security_guard", return_value=guard
    ):
        register_event_handlers(app, manager, mode="intent", enable_streaming=False)

    handler = _confirm_handler(app)
    client = MagicMock()
    body = {
        "user": {"id": "U123"},
        "actions": [{"action_id": "confirm_action:expected123"}],
        "channel": {"id": "C123"},
        "message": {"ts": "123.456"},
    }

    handler(lambda: None, body, client)

    assert pending.executed is False
    assert context.pending_action is None
    client.chat_update.assert_called_once()
    assert "expired" in client.chat_update.call_args.kwargs["text"].lower()
    audit_logger.log_security_event.assert_called()


def test_confirm_finds_thread_context_by_action_id():
    app = _FakeApp()
    manager = ConversationManager(persist=False)
    context = manager.get_or_create("U123", "C123", thread_ts="999.111")
    pending = _PendingActionStub(action_id="expected123", expired=False)
    context.pending_action = pending

    audit_logger = MagicMock()
    guard = SecurityGuard()
    with patch("src.bot.event_handlers.get_audit_logger", return_value=audit_logger), patch(
        "src.bot.event_handlers.get_security_guard", return_value=guard
    ):
        register_event_handlers(app, manager, mode="intent", enable_streaming=False)

    handler = _confirm_handler(app)
    client = MagicMock()
    body = {
        "user": {"id": "U123"},
        "actions": [{"action_id": "confirm_action:expected123"}],
        "channel": {"id": "C123"},
        "message": {"ts": "123.456"},
    }

    handler(lambda: None, body, client)

    assert pending.executed is True
    assert context.pending_action is None
    client.chat_update.assert_called_once()
    assert "action completed" in client.chat_update.call_args.kwargs["text"].lower()
