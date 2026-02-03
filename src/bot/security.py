"""Security utilities for the Hani Replica bot.

Provides input sanitization, rate limiting, and action validation
to protect against prompt injection and abuse.
"""

import hashlib
import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class SecurityLevel(str, Enum):
    """Security enforcement levels."""
    STRICT = "strict"      # Block suspicious content
    MODERATE = "moderate"  # Warn but allow
    PERMISSIVE = "permissive"  # Log only


class ThreatType(str, Enum):
    """Types of security threats detected."""
    PROMPT_INJECTION = "prompt_injection"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    UNAUTHORIZED_ACTION = "unauthorized_action"
    SENSITIVE_DATA = "sensitive_data"
    SUSPICIOUS_PATTERN = "suspicious_pattern"


@dataclass
class SecurityEvent:
    """Record of a security-related event."""
    timestamp: datetime
    user_id: str
    threat_type: ThreatType
    severity: str  # "low", "medium", "high", "critical"
    description: str
    original_input: str = ""
    sanitized_input: str = ""
    blocked: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "threat_type": self.threat_type.value,
            "severity": self.severity,
            "description": self.description,
            "original_input_hash": hashlib.sha256(
                self.original_input.encode()
            ).hexdigest()[:16] if self.original_input else None,
            "blocked": self.blocked,
            "metadata": self.metadata,
        }


@dataclass
class RateLimitEntry:
    """Track rate limit state for a user."""
    request_count: int = 0
    window_start: float = field(default_factory=time.time)
    blocked_until: float | None = None


# Patterns that may indicate prompt injection attempts
INJECTION_PATTERNS = [
    # System prompt manipulation
    r"(?i)ignore\s+(previous|all|above)\s+(instructions?|prompts?|rules?)",
    r"(?i)disregard\s+(previous|all|above)\s+(instructions?|prompts?)",
    r"(?i)forget\s+(everything|all|previous)",
    r"(?i)new\s+instructions?:",
    r"(?i)system\s*prompt:",
    r"(?i)you\s+are\s+now\s+a",
    r"(?i)pretend\s+(to\s+be|you\s+are)",
    r"(?i)act\s+as\s+(if|though)",
    r"(?i)roleplay\s+as",

    # Delimiter injection
    r"```\s*system",
    r"<\s*system\s*>",
    r"\[\s*SYSTEM\s*\]",
    r"###\s*SYSTEM",

    # Jailbreak attempts
    r"(?i)dan\s*mode",
    r"(?i)developer\s*mode",
    r"(?i)jailbreak",
    r"(?i)bypass\s+(safety|security|filter)",

    # Output manipulation
    r"(?i)print\s+(everything|all|secret)",
    r"(?i)reveal\s+(your|the)\s+(prompt|instructions?|system)",
    r"(?i)show\s+me\s+(your|the)\s+(prompt|instructions?)",
    r"(?i)what\s+(is|are)\s+your\s+(instructions?|rules?|prompt)",
]

# Patterns for sensitive data
SENSITIVE_PATTERNS = [
    # API keys and tokens
    r"(?i)(api[_-]?key|secret[_-]?key|access[_-]?token)\s*[:=]\s*['\"]?[\w\-]+",
    r"sk-[a-zA-Z0-9]{20,}",  # OpenAI-style keys
    r"xox[baprs]-[a-zA-Z0-9\-]+",  # Slack tokens

    # Passwords
    r"(?i)password\s*[:=]\s*['\"]?[^\s'\"]+",

    # Private keys
    r"-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----",

    # Credit cards (basic pattern)
    r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b",

    # SSN (US)
    r"\b\d{3}[\s\-]?\d{2}[\s\-]?\d{4}\b",
]

# Characters that could be used for encoding attacks
SUSPICIOUS_CHARS = [
    "\u200b",  # Zero-width space
    "\u200c",  # Zero-width non-joiner
    "\u200d",  # Zero-width joiner
    "\u2060",  # Word joiner
    "\ufeff",  # Byte order mark
    "\u00ad",  # Soft hyphen
]


class SecurityGuard:
    """Security guardian for bot interactions.

    Provides:
    - Input sanitization against prompt injection
    - Per-user rate limiting
    - Action validation for sensitive operations
    - Sensitive data detection
    """

    def __init__(
        self,
        level: SecurityLevel = SecurityLevel.MODERATE,
        rate_limit_requests: int = 30,
        rate_limit_window: int = 60,
        rate_limit_block_duration: int = 300,
    ):
        """Initialize the security guard.

        Args:
            level: Security enforcement level.
            rate_limit_requests: Max requests per window.
            rate_limit_window: Window duration in seconds.
            rate_limit_block_duration: How long to block after exceeding limit.
        """
        self.level = level
        self.rate_limit_requests = rate_limit_requests
        self.rate_limit_window = rate_limit_window
        self.rate_limit_block_duration = rate_limit_block_duration

        # Rate limit tracking
        self._rate_limits: dict[str, RateLimitEntry] = defaultdict(RateLimitEntry)

        # Security event history (in-memory, limited)
        self._events: list[SecurityEvent] = []
        self._max_events = 1000

        # Compile patterns for efficiency
        self._injection_patterns = [re.compile(p) for p in INJECTION_PATTERNS]
        self._sensitive_patterns = [re.compile(p) for p in SENSITIVE_PATTERNS]

        logger.info(f"SecurityGuard initialized with level={level.value}")

    def sanitize_input(self, text: str, user_id: str = "unknown") -> tuple[str, list[SecurityEvent]]:
        """Sanitize user input to prevent prompt injection.

        Args:
            text: Raw user input.
            user_id: User identifier for logging.

        Returns:
            Tuple of (sanitized text, list of security events detected).
        """
        events = []
        sanitized = text

        # Check for suspicious unicode characters
        for char in SUSPICIOUS_CHARS:
            if char in sanitized:
                sanitized = sanitized.replace(char, "")
                events.append(SecurityEvent(
                    timestamp=datetime.now(),
                    user_id=user_id,
                    threat_type=ThreatType.SUSPICIOUS_PATTERN,
                    severity="low",
                    description=f"Removed suspicious unicode character: U+{ord(char):04X}",
                    original_input=text,
                    sanitized_input=sanitized,
                ))

        # Check for prompt injection patterns
        for pattern in self._injection_patterns:
            match = pattern.search(sanitized)
            if match:
                event = SecurityEvent(
                    timestamp=datetime.now(),
                    user_id=user_id,
                    threat_type=ThreatType.PROMPT_INJECTION,
                    severity="high",
                    description=f"Potential prompt injection detected: {match.group()[:50]}",
                    original_input=text,
                    blocked=self.level == SecurityLevel.STRICT,
                    metadata={"pattern": pattern.pattern[:100]},
                )
                events.append(event)
                self._record_event(event)

                if self.level == SecurityLevel.STRICT:
                    logger.warning(
                        f"Blocked prompt injection from user {user_id}: {match.group()[:50]}"
                    )
                    return "", events
                elif self.level == SecurityLevel.MODERATE:
                    logger.warning(
                        f"Detected prompt injection from user {user_id}: {match.group()[:50]}"
                    )
                    # Remove the suspicious portion
                    sanitized = pattern.sub("[FILTERED]", sanitized)

        # Check for sensitive data
        for pattern in self._sensitive_patterns:
            if pattern.search(sanitized):
                event = SecurityEvent(
                    timestamp=datetime.now(),
                    user_id=user_id,
                    threat_type=ThreatType.SENSITIVE_DATA,
                    severity="medium",
                    description="Input may contain sensitive data",
                    original_input="[REDACTED]",
                )
                events.append(event)
                self._record_event(event)
                logger.warning(f"Sensitive data pattern detected in input from user {user_id}")

        # Normalize whitespace
        sanitized = " ".join(sanitized.split())

        # Truncate extremely long inputs
        max_length = 10000
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length] + "... [truncated]"
            events.append(SecurityEvent(
                timestamp=datetime.now(),
                user_id=user_id,
                threat_type=ThreatType.SUSPICIOUS_PATTERN,
                severity="low",
                description=f"Input truncated from {len(text)} to {max_length} characters",
                original_input="[LONG INPUT]",
                sanitized_input="[TRUNCATED]",
            ))

        return sanitized, events

    def check_rate_limit(self, user_id: str) -> tuple[bool, SecurityEvent | None]:
        """Check if a user has exceeded their rate limit.

        Args:
            user_id: User identifier.

        Returns:
            Tuple of (allowed, security event if blocked).
        """
        current_time = time.time()
        entry = self._rate_limits[user_id]

        # Check if user is currently blocked
        if entry.blocked_until and current_time < entry.blocked_until:
            remaining = int(entry.blocked_until - current_time)
            event = SecurityEvent(
                timestamp=datetime.now(),
                user_id=user_id,
                threat_type=ThreatType.RATE_LIMIT_EXCEEDED,
                severity="medium",
                description=f"User blocked for {remaining} more seconds",
                blocked=True,
                metadata={"remaining_seconds": remaining},
            )
            self._record_event(event)
            return False, event

        # Reset window if expired
        if current_time - entry.window_start > self.rate_limit_window:
            entry.request_count = 0
            entry.window_start = current_time
            entry.blocked_until = None

        # Increment count
        entry.request_count += 1

        # Check if limit exceeded
        if entry.request_count > self.rate_limit_requests:
            entry.blocked_until = current_time + self.rate_limit_block_duration
            event = SecurityEvent(
                timestamp=datetime.now(),
                user_id=user_id,
                threat_type=ThreatType.RATE_LIMIT_EXCEEDED,
                severity="high",
                description=f"Rate limit exceeded: {entry.request_count} requests in {self.rate_limit_window}s",
                blocked=True,
                metadata={
                    "request_count": entry.request_count,
                    "block_duration": self.rate_limit_block_duration,
                },
            )
            self._record_event(event)
            logger.warning(
                f"Rate limit exceeded for user {user_id}: "
                f"{entry.request_count} requests, blocked for {self.rate_limit_block_duration}s"
            )
            return False, event

        return True, None

    def validate_action(
        self,
        action_type: str,
        user_id: str,
        context: dict[str, Any] | None = None,
    ) -> tuple[bool, SecurityEvent | None]:
        """Validate whether an action should be allowed.

        Args:
            action_type: Type of action (e.g., "create_draft", "create_issue").
            user_id: User identifier.
            context: Additional context about the action.

        Returns:
            Tuple of (allowed, security event if blocked).
        """
        context = context or {}

        # Actions that require extra validation
        sensitive_actions = {
            "create_draft": "Creates an email draft",
            "create_issue": "Creates a GitHub issue",
            "send_message": "Sends a message",
        }

        if action_type in sensitive_actions:
            # Log the sensitive action attempt
            logger.info(
                f"Sensitive action '{action_type}' requested by user {user_id}"
            )

            # In strict mode, we could require additional confirmation
            if self.level == SecurityLevel.STRICT:
                # Check for suspicious patterns in action context
                if context.get("body") or context.get("content"):
                    content = context.get("body") or context.get("content", "")
                    _, events = self.sanitize_input(content, user_id)

                    if any(e.threat_type == ThreatType.PROMPT_INJECTION for e in events):
                        event = SecurityEvent(
                            timestamp=datetime.now(),
                            user_id=user_id,
                            threat_type=ThreatType.UNAUTHORIZED_ACTION,
                            severity="high",
                            description=f"Blocked {action_type}: suspicious content detected",
                            blocked=True,
                            metadata={"action_type": action_type},
                        )
                        self._record_event(event)
                        return False, event

        return True, None

    def get_user_stats(self, user_id: str) -> dict[str, Any]:
        """Get security statistics for a user.

        Args:
            user_id: User identifier.

        Returns:
            Dictionary with user security stats.
        """
        entry = self._rate_limits.get(user_id)
        user_events = [e for e in self._events if e.user_id == user_id]

        return {
            "user_id": user_id,
            "current_request_count": entry.request_count if entry else 0,
            "is_blocked": bool(
                entry and entry.blocked_until and time.time() < entry.blocked_until
            ),
            "blocked_until": (
                datetime.fromtimestamp(entry.blocked_until).isoformat()
                if entry and entry.blocked_until
                else None
            ),
            "total_security_events": len(user_events),
            "events_by_type": {
                t.value: len([e for e in user_events if e.threat_type == t])
                for t in ThreatType
            },
        }

    def get_recent_events(
        self,
        limit: int = 100,
        user_id: str | None = None,
        threat_type: ThreatType | None = None,
    ) -> list[dict[str, Any]]:
        """Get recent security events.

        Args:
            limit: Maximum events to return.
            user_id: Filter by user ID.
            threat_type: Filter by threat type.

        Returns:
            List of event dictionaries.
        """
        events = self._events

        if user_id:
            events = [e for e in events if e.user_id == user_id]
        if threat_type:
            events = [e for e in events if e.threat_type == threat_type]

        # Return most recent first
        events = sorted(events, key=lambda e: e.timestamp, reverse=True)
        return [e.to_dict() for e in events[:limit]]

    def clear_rate_limit(self, user_id: str) -> None:
        """Clear rate limit for a user (admin function).

        Args:
            user_id: User identifier.
        """
        if user_id in self._rate_limits:
            del self._rate_limits[user_id]
            logger.info(f"Cleared rate limit for user {user_id}")

    def _record_event(self, event: SecurityEvent) -> None:
        """Record a security event.

        Args:
            event: Event to record.
        """
        self._events.append(event)

        # Trim if exceeds max
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events:]


# Singleton instance for global access
_security_guard: SecurityGuard | None = None


def get_security_guard() -> SecurityGuard:
    """Get the global SecurityGuard instance.

    Returns:
        SecurityGuard singleton.
    """
    global _security_guard
    if _security_guard is None:
        from ..config import (
            SECURITY_LEVEL,
            RATE_LIMIT_REQUESTS,
            RATE_LIMIT_WINDOW,
            RATE_LIMIT_BLOCK_DURATION,
        )
        _security_guard = SecurityGuard(
            level=SecurityLevel(SECURITY_LEVEL),
            rate_limit_requests=RATE_LIMIT_REQUESTS,
            rate_limit_window=RATE_LIMIT_WINDOW,
            rate_limit_block_duration=RATE_LIMIT_BLOCK_DURATION,
        )
    return _security_guard


def reset_security_guard() -> None:
    """Reset the global SecurityGuard instance (for testing)."""
    global _security_guard
    _security_guard = None
