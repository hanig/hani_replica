"""Chat handler for natural conversation."""

import logging
from typing import TYPE_CHECKING, Any

from anthropic import Anthropic

from ...config import ANTHROPIC_API_KEY
from ..conversation import ConversationContext
from ..intent_router import Intent
from .base import BaseHandler

if TYPE_CHECKING:
    from ..user_memory import UserMemory

logger = logging.getLogger(__name__)

# System prompt for conversational responses
CHAT_SYSTEM_PROMPT = """You are Hani's personal assistant. Be friendly, helpful, and concise.

You help Hani with:
- Searching emails, documents, and Slack messages
- Checking calendar and finding availability
- Managing GitHub issues and PRs
- Getting daily briefings and summaries

Keep responses brief and natural. You can have casual conversations but always be ready to help with work tasks.

When asked what you can do, mention these capabilities naturally without being overly formal.
"""


class ChatHandler(BaseHandler):
    """Handle conversational messages that don't need tools."""

    def __init__(
        self,
        api_key: str | None = None,
        user_memory: "UserMemory | None" = None,
    ):
        """Initialize the chat handler.

        Args:
            api_key: Anthropic API key.
            user_memory: Optional UserMemory instance for context.
        """
        self.api_key = api_key or ANTHROPIC_API_KEY
        self._client = None
        self.user_memory = user_memory

    @property
    def client(self) -> Anthropic:
        """Lazy-load the Anthropic client."""
        if self._client is None:
            if not self.api_key:
                raise ValueError("ANTHROPIC_API_KEY not configured")
            self._client = Anthropic(api_key=self.api_key)
        return self._client

    def handle(self, intent: Intent, context: ConversationContext) -> dict[str, Any]:
        """Handle a chat intent.

        Args:
            intent: Classified intent with entities.
            context: Conversation context.

        Returns:
            Response dictionary with 'text'.
        """
        message = intent.entities.get("message", "")

        # Build messages from conversation history
        messages = self._build_messages(context, message)

        # Build system prompt with user context
        system_prompt = self._build_system_prompt(context.user_id)

        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=300,
                system=system_prompt,
                messages=messages,
            )

            response_text = response.content[0].text
            return {"text": response_text}

        except Exception as e:
            logger.error(f"Error generating chat response: {e}")
            # Fallback to simple responses for common cases
            return {"text": self._fallback_response(message)}

    def _build_system_prompt(self, user_id: str) -> str:
        """Build system prompt with user context.

        Args:
            user_id: Slack user ID.

        Returns:
            System prompt string.
        """
        prompt = CHAT_SYSTEM_PROMPT

        # Add user context if available
        if self.user_memory:
            try:
                user_context = self.user_memory.get_context_summary(user_id)
                if user_context:
                    prompt += f"\n\nContext about this user:\n{user_context}"
            except Exception as e:
                logger.warning(f"Failed to get user context: {e}")

        return prompt

    def _build_messages(
        self, context: ConversationContext, current_message: str
    ) -> list[dict]:
        """Build message list from conversation history.

        Args:
            context: Conversation context with history.
            current_message: Current user message.

        Returns:
            List of messages for the API.
        """
        messages = []

        # Add recent history (last few exchanges)
        recent_history = context.get_recent_history(count=6)
        for msg in recent_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if content:
                messages.append({"role": role, "content": content})

        # Add current message if not already in history
        if not messages or messages[-1].get("content") != current_message:
            messages.append({"role": "user", "content": current_message})

        # Ensure messages alternate properly (Claude API requirement)
        messages = self._fix_message_order(messages)

        return messages

    def _fix_message_order(self, messages: list[dict]) -> list[dict]:
        """Ensure messages alternate between user and assistant.

        Args:
            messages: List of messages.

        Returns:
            Fixed list of messages.
        """
        if not messages:
            return [{"role": "user", "content": "hi"}]

        fixed = []
        prev_role = None

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Skip empty messages
            if not content:
                continue

            # Skip duplicate roles
            if role == prev_role:
                continue

            fixed.append({"role": role, "content": content})
            prev_role = role

        # Ensure first message is from user
        if fixed and fixed[0]["role"] != "user":
            fixed = fixed[1:]

        # Ensure last message is from user
        if fixed and fixed[-1]["role"] != "user":
            fixed.append({"role": "user", "content": "hi"})

        # Ensure we have at least one message
        if not fixed:
            fixed = [{"role": "user", "content": "hi"}]

        return fixed

    def _fallback_response(self, message: str) -> str:
        """Provide fallback responses when LLM is unavailable.

        Args:
            message: User message.

        Returns:
            Fallback response text.
        """
        message_lower = message.lower().strip()

        # Greetings
        greetings = {"hi", "hello", "hey", "yo", "sup", "hiya", "howdy"}
        if message_lower in greetings or message_lower.split()[0].rstrip("!,.") in greetings:
            return "Hi! How can I help you today?"

        # Questions about capabilities
        if "what can you do" in message_lower or "help" in message_lower:
            return (
                "I can help you with:\n"
                "- Searching your emails, docs, and Slack messages\n"
                "- Checking your calendar and finding free time\n"
                "- Managing GitHub issues and PRs\n"
                "- Getting daily briefings\n\n"
                "Just ask me anything!"
            )

        # Thanks
        if any(w in message_lower for w in ["thanks", "thank you", "thx", "ty"]):
            return "You're welcome! Let me know if you need anything else."

        # Default
        return "I'm here to help! You can ask me about your calendar, emails, or search for anything."
