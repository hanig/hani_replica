"""Tests for chat handler."""

import pytest

from src.bot.handlers.chat import ChatHandler
from src.bot.conversation import ConversationContext
from src.bot.intent_router import Intent


class TestChatHandler:
    """Tests for ChatHandler class."""

    @pytest.fixture
    def handler(self):
        """Create handler without API key (uses fallback)."""
        return ChatHandler(api_key=None)

    @pytest.fixture
    def context(self):
        """Create a conversation context."""
        return ConversationContext(
            user_id="U123",
            channel_id="C456",
            thread_ts="123.456",
        )

    def test_fallback_greeting_hi(self, handler, context):
        """Test fallback response for 'hi'."""
        intent = Intent(intent="chat", entities={"message": "hi"})
        response = handler.handle(intent, context)

        assert "text" in response
        text_lower = response["text"].lower()
        # Accept various valid greeting responses
        assert any(
            word in text_lower
            for word in ["help", "hi", "hello", "assist", "how can i"]
        )

    def test_fallback_greeting_hello(self, handler, context):
        """Test fallback response for 'hello'."""
        intent = Intent(intent="chat", entities={"message": "hello"})
        response = handler.handle(intent, context)

        assert "text" in response
        assert response["text"]  # Not empty

    def test_fallback_what_can_you_do(self, handler, context):
        """Test fallback response for capability question."""
        intent = Intent(intent="chat", entities={"message": "what can you do"})
        response = handler.handle(intent, context)

        assert "text" in response
        # Should mention capabilities
        text_lower = response["text"].lower()
        assert any(
            word in text_lower
            for word in ["email", "calendar", "search", "github", "help"]
        )

    def test_fallback_thanks(self, handler, context):
        """Test fallback response for thanks."""
        intent = Intent(intent="chat", entities={"message": "thanks!"})
        response = handler.handle(intent, context)

        assert "text" in response
        text_lower = response["text"].lower()
        # Accept various valid responses to "thanks"
        assert any(
            word in text_lower
            for word in ["welcome", "pleasure", "glad", "happy", "help", "assist"]
        )

    def test_fallback_default(self, handler, context):
        """Test fallback response for unknown message."""
        intent = Intent(intent="chat", entities={"message": "blah blah"})
        response = handler.handle(intent, context)

        assert "text" in response
        assert response["text"]  # Not empty

    def test_build_messages_empty_history(self, handler, context):
        """Test message building with empty history."""
        messages = handler._build_messages(context, "hello")

        assert len(messages) >= 1
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "hello"

    def test_build_messages_with_history(self, handler, context):
        """Test message building with existing history."""
        context.add_message("user", "hi")
        context.add_message("assistant", "Hello!")

        messages = handler._build_messages(context, "how are you")

        assert len(messages) >= 2
        # Should end with user message
        assert messages[-1]["role"] == "user"

    def test_fix_message_order_empty(self, handler):
        """Test fix_message_order with empty list."""
        fixed = handler._fix_message_order([])

        assert len(fixed) == 1
        assert fixed[0]["role"] == "user"

    def test_fix_message_order_consecutive_users(self, handler):
        """Test fix_message_order removes consecutive same roles."""
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "user", "content": "hello"},
        ]
        fixed = handler._fix_message_order(messages)

        # Should have removed the duplicate
        assert len(fixed) == 1
        assert fixed[0]["content"] == "hi"

    def test_fix_message_order_starts_with_assistant(self, handler):
        """Test fix_message_order when starting with assistant."""
        messages = [
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "hello"},
        ]
        fixed = handler._fix_message_order(messages)

        # Should start with user
        assert fixed[0]["role"] == "user"
