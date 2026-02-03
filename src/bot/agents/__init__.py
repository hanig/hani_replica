"""Multi-agent architecture for Hani Replica.

This module provides specialized agents for different domains (calendar, email,
GitHub, research) and an orchestrator that routes tasks to the appropriate
specialists.
"""

from .base import BaseAgent, AgentResult
from .calendar_agent import CalendarAgent
from .email_agent import EmailAgent
from .github_agent import GitHubAgent
from .research_agent import ResearchAgent
from .orchestrator import Orchestrator

__all__ = [
    "BaseAgent",
    "AgentResult",
    "CalendarAgent",
    "EmailAgent",
    "GitHubAgent",
    "ResearchAgent",
    "Orchestrator",
]
