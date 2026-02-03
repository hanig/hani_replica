"""LLM-powered intent classification for the Slack bot."""

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from anthropic import Anthropic

from ..config import ANTHROPIC_API_KEY, INTENT_MODEL

logger = logging.getLogger(__name__)


@dataclass
class Intent:
    """Classified intent with extracted entities."""

    intent: str
    entities: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    raw_response: str = ""


INTENT_DEFINITIONS = """
Available intents:
- chat: General conversation, greetings, small talk, questions about the bot itself
- search: General semantic search across all data (emails, docs, Slack messages)
- calendar_check: Check calendar events for a specific date
- calendar_availability: Find free time slots for scheduling
- email_search: Search specifically for emails
- email_draft: Create a new email draft (never sends)
- github_search: Search GitHub code, issues, or PRs
- github_create_issue: Create a new GitHub issue
- github_list_prs: List open pull requests
- briefing: Get a daily summary/briefing
- help: Show help information

Entity types to extract:
- query: Search query text
- date: Date reference (today, tomorrow, next Monday, 2024-01-15, etc.)
- person: Person name or email
- repo: GitHub repository name
- title: Title for issue/email
- body: Body/description text
- labels: Labels/tags (comma-separated)
"""

SYSTEM_PROMPT = f"""You are an intent classifier for a personal assistant bot.
Given a user message, classify the intent and extract relevant entities.

IMPORTANT: Not every message needs a tool!
- Greetings ("hi", "hello", "hey") → intent: "chat"
- Questions about the bot ("what can you do", "who are you") → intent: "chat"
- Casual conversation or small talk → intent: "chat"
- Thanks or acknowledgments ("thanks", "got it") → intent: "chat"
- Only use search/calendar/email intents when user clearly wants data

{INTENT_DEFINITIONS}

Respond with a JSON object containing:
- intent: The classified intent (one of the listed intents)
- entities: Dictionary of extracted entities
- confidence: Your confidence in the classification (0.0 to 1.0)

Examples:

User: "hi"
Response: {{"intent": "chat", "entities": {{}}, "confidence": 0.99}}

User: "what's up?"
Response: {{"intent": "chat", "entities": {{}}, "confidence": 0.95}}

User: "hello, how are you?"
Response: {{"intent": "chat", "entities": {{}}, "confidence": 0.98}}

User: "what can you do?"
Response: {{"intent": "chat", "entities": {{}}, "confidence": 0.95}}

User: "thanks!"
Response: {{"intent": "chat", "entities": {{}}, "confidence": 0.95}}

User: "What's on my calendar today?"
Response: {{"intent": "calendar_check", "entities": {{"date": "today"}}, "confidence": 0.95}}

User: "Search for emails about the quarterly report"
Response: {{"intent": "email_search", "entities": {{"query": "quarterly report"}}, "confidence": 0.9}}

User: "Create an issue in evo-training about the memory leak"
Response: {{"intent": "github_create_issue", "entities": {{"repo": "evo-training", "title": "memory leak"}}, "confidence": 0.85}}

User: "Find documents about the Evo model architecture"
Response: {{"intent": "search", "entities": {{"query": "Evo model architecture"}}, "confidence": 0.9}}

User: "When am I free tomorrow afternoon?"
Response: {{"intent": "calendar_availability", "entities": {{"date": "tomorrow"}}, "confidence": 0.9}}

User: "Show me my open PRs"
Response: {{"intent": "github_list_prs", "entities": {{}}, "confidence": 0.95}}

User: "Draft an email to john@example.com about the meeting"
Response: {{"intent": "email_draft", "entities": {{"person": "john@example.com", "query": "meeting"}}, "confidence": 0.85}}

User: "What did I miss yesterday?"
Response: {{"intent": "briefing", "entities": {{"date": "yesterday"}}, "confidence": 0.8}}

Always respond with valid JSON only, no other text.
"""


class IntentRouter:
    """Routes user messages to appropriate handlers using LLM classification."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        """Initialize the intent router.

        Args:
            api_key: Anthropic API key.
            model: Model to use for classification.
        """
        self.api_key = api_key or ANTHROPIC_API_KEY
        self.model = model or INTENT_MODEL

        if not self.api_key:
            logger.warning("No Anthropic API key - using keyword fallback")
            self._client = None
        else:
            self._client = Anthropic(api_key=self.api_key)

    def classify(self, text: str, history: list[dict] | None = None) -> Intent:
        """Classify user intent from message text.

        Args:
            text: User message text.
            history: Conversation history for context.

        Returns:
            Classified Intent object.
        """
        if not self._client:
            return self._keyword_fallback(text)

        try:
            # Build context from history if available
            context = ""
            if history and len(history) > 0:
                recent = history[-4:]  # Last 2 exchanges
                context = "Recent conversation:\n"
                for msg in recent:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")[:200]
                    context += f"{role}: {content}\n"
                context += "\n"

            prompt = f"{context}User: {text}"

            response = self._client.messages.create(
                model=self.model,
                max_tokens=200,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )

            # Parse response
            raw_text = response.content[0].text.strip()

            # Extract JSON from response
            try:
                # Handle potential markdown code blocks
                if "```" in raw_text:
                    json_match = raw_text.split("```")[1]
                    if json_match.startswith("json"):
                        json_match = json_match[4:]
                    raw_text = json_match.strip()

                data = json.loads(raw_text)

                return Intent(
                    intent=data.get("intent", "search"),
                    entities=data.get("entities", {}),
                    confidence=data.get("confidence", 0.8),
                    raw_response=raw_text,
                )

            except json.JSONDecodeError:
                logger.warning(f"Failed to parse intent response: {raw_text}")
                return self._keyword_fallback(text)

        except Exception as e:
            logger.error(f"Error classifying intent: {e}")
            return self._keyword_fallback(text)

    def _keyword_fallback(self, text: str) -> Intent:
        """Fall back to keyword-based classification.

        Args:
            text: User message text.

        Returns:
            Classified Intent object.
        """
        text_lower = text.lower().strip()

        # Check for greetings/chat FIRST - these should never trigger tools
        greetings = {"hi", "hello", "hey", "sup", "yo", "hiya", "howdy"}
        greeting_phrases = ["good morning", "good afternoon", "good evening", "what's up", "whats up"]

        # Exact match for simple greetings
        if text_lower in greetings:
            return Intent(intent="chat", entities={"message": text}, confidence=0.99)

        # Check for greeting phrases
        if any(phrase in text_lower for phrase in greeting_phrases):
            return Intent(intent="chat", entities={"message": text}, confidence=0.95)

        # Check for greetings at the start (e.g., "hi there", "hello!")
        first_word = text_lower.split()[0].rstrip("!,.")
        if first_word in greetings:
            return Intent(intent="chat", entities={"message": text}, confidence=0.95)

        # Questions about the bot itself
        bot_questions = ["who are you", "what are you", "what can you do", "how do you work"]
        if any(phrase in text_lower for phrase in bot_questions):
            return Intent(intent="chat", entities={"message": text}, confidence=0.95)

        # Thanks/acknowledgments
        thanks_words = ["thanks", "thank you", "thx", "ty", "got it", "ok thanks", "cool", "great"]
        if any(text_lower.startswith(w) or text_lower == w for w in thanks_words):
            return Intent(intent="chat", entities={"message": text}, confidence=0.9)

        # Email keywords - check first (more specific with "draft")
        if any(w in text_lower for w in ["email", "mail", "inbox"]):
            if any(w in text_lower for w in ["draft", "write", "compose", "send"]):
                return Intent(intent="email_draft", entities={"query": text})
            return Intent(intent="email_search", entities={"query": text})

        # Check for availability intent first (doesn't require "calendar" keyword)
        if any(w in text_lower for w in ["free", "available", "availability", "open slot"]):
            return Intent(intent="calendar_availability", entities=self._extract_date(text))

        # Calendar keywords
        if any(w in text_lower for w in ["calendar", "schedule", "meeting", "event"]):
            return Intent(intent="calendar_check", entities=self._extract_date(text))

        # GitHub keywords
        if any(w in text_lower for w in ["github", "repo", "issue", "pr", "pull request", "commit"]):
            if any(w in text_lower for w in ["create", "new", "open"]) and "issue" in text_lower:
                return Intent(intent="github_create_issue", entities={"query": text})
            if any(w in text_lower for w in ["pr", "pull request"]):
                return Intent(intent="github_list_prs", entities={})
            return Intent(intent="github_search", entities={"query": text})

        # Briefing keywords
        if any(w in text_lower for w in ["briefing", "summary", "catch up", "what did i miss"]):
            return Intent(intent="briefing", entities=self._extract_date(text))

        # Help keywords
        if any(w in text_lower for w in ["help", "what can you do", "commands"]):
            return Intent(intent="help", entities={})

        # Default to chat (not search) for short/unclear messages
        # Only use search as default for longer, query-like messages
        if len(text.split()) <= 3:
            return Intent(intent="chat", entities={"message": text}, confidence=0.5)

        # Default to search with date extraction for longer messages
        entities = {"query": text}
        entities.update(self._extract_date(text))
        return Intent(intent="search", entities=entities, confidence=0.5)

    def _extract_date(self, text: str) -> dict:
        """Extract date references from text."""
        text_lower = text.lower()
        entities = {}

        if "today" in text_lower:
            entities["date"] = "today"
        elif "tomorrow" in text_lower:
            entities["date"] = "tomorrow"
        elif "yesterday" in text_lower:
            entities["date"] = "yesterday"
        elif "next week" in text_lower:
            entities["date"] = "next week"
        elif "this week" in text_lower:
            entities["date"] = "this week"

        return entities
