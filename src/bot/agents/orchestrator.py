"""Orchestrator agent that coordinates specialist agents."""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Generator

from anthropic import Anthropic

from .base import BaseAgent, AgentType, AgentResult, AgentStreamEvent
from .calendar_agent import CalendarAgent
from .email_agent import EmailAgent
from .github_agent import GitHubAgent
from .research_agent import ResearchAgent
from ..conversation import ConversationContext
from ..user_memory import UserMemory
from ...config import ANTHROPIC_API_KEY, AGENT_MODEL

logger = logging.getLogger(__name__)


@dataclass
class TaskPlan:
    """Plan for executing a user request."""
    needs_specialist: bool
    specialist_types: list[AgentType] = field(default_factory=list)
    subtasks: list[dict[str, Any]] = field(default_factory=list)
    is_conversational: bool = False
    reasoning: str = ""


# Greetings and conversational patterns
GREETINGS = {"hi", "hello", "hey", "sup", "yo", "good morning", "good afternoon", "good evening"}
CONVERSATIONAL_PATTERNS = [
    "how are you", "what's up", "who are you", "what can you do",
    "thanks", "thank you", "great", "awesome", "cool", "ok", "okay",
    "help", "bye", "goodbye", "see you",
]


class Orchestrator(BaseAgent):
    """Orchestrator that routes tasks to specialist agents.

    The orchestrator:
    1. Analyzes user requests
    2. Decides which specialist(s) are needed
    3. Routes subtasks to appropriate specialists
    4. Synthesizes results from multiple agents
    5. Handles conversational interactions directly

    For simple, single-domain requests, it routes to one specialist.
    For complex, multi-domain requests, it coordinates multiple specialists.
    For conversational messages, it responds directly without specialists.
    """

    AGENT_TYPE = AgentType.ORCHESTRATOR
    MAX_ITERATIONS = 3  # Keep low since specialists handle the heavy lifting

    def __init__(
        self,
        api_key: str | None = None,
        user_memory: UserMemory | None = None,
        model: str | None = None,
    ):
        """Initialize orchestrator with specialist agents.

        Args:
            api_key: Anthropic API key.
            user_memory: Optional UserMemory for context injection.
            model: Model to use for orchestration.
        """
        super().__init__(api_key, user_memory, model)

        # Initialize specialist agents
        self.specialists: dict[AgentType, BaseAgent] = {
            AgentType.CALENDAR: CalendarAgent(api_key, user_memory, model),
            AgentType.EMAIL: EmailAgent(api_key, user_memory, model),
            AgentType.GITHUB: GitHubAgent(api_key, user_memory, model),
            AgentType.RESEARCH: ResearchAgent(api_key, user_memory, model),
        }

    @property
    def tool_names(self) -> list[str]:
        """Orchestrator doesn't use tools directly - it delegates."""
        return ["RespondToUserTool"]

    @property
    def system_prompt(self) -> str:
        """Orchestrator system prompt for routing decisions."""
        return """You are Hani's personal AI assistant orchestrator.

Today's date: {current_date}

Your role is to:
1. Understand what the user needs
2. Provide helpful, friendly responses
3. For complex requests, coordinate with specialist agents (which you manage internally)

SPECIALIST DOMAINS (handled automatically):
- Calendar: Events, schedules, availability
- Email: Search, drafts, inbox management
- GitHub: PRs, issues, code search
- Research: Knowledge graph, documents, people

GUIDELINES:
1. Be conversational and friendly
2. For greetings and small talk, respond naturally
3. For questions about your capabilities, explain what you can do
4. For domain-specific tasks, I'll route to the right specialist
5. Always use RespondToUserTool to send your final response

TONE:
- Professional but warm
- Concise but helpful
- Proactive in offering relevant information

Remember: You're Hani's trusted assistant. Be helpful, accurate, and efficient."""

    @property
    def description(self) -> str:
        return "Main orchestrator: routes tasks and handles conversation"

    def _is_conversational(self, message: str) -> bool:
        """Check if message is purely conversational."""
        message_lower = message.lower().strip()

        # Check greetings
        if message_lower in GREETINGS:
            return True
        if any(message_lower.startswith(g + " ") or message_lower.startswith(g + ",")
               for g in GREETINGS):
            return True

        # Check conversational patterns
        for pattern in CONVERSATIONAL_PATTERNS:
            if pattern in message_lower:
                return True

        # Very short messages are often conversational
        if len(message_lower.split()) <= 2 and "?" not in message:
            return True

        return False

    def _select_specialist(self, message: str, context: ConversationContext) -> AgentType | None:
        """Select the best specialist for a message.

        Args:
            message: User message.
            context: Conversation context.

        Returns:
            AgentType of best specialist, or None if conversational.
        """
        if self._is_conversational(message):
            return None

        # Get confidence scores from each specialist
        scores: dict[AgentType, float] = {}
        for agent_type, agent in self.specialists.items():
            score = agent.can_handle(message, context)
            if score > 0:
                scores[agent_type] = score

        if not scores:
            # Default to research for unknown queries
            return AgentType.RESEARCH

        # Return highest scoring specialist
        best = max(scores.items(), key=lambda x: x[1])
        if best[1] >= 0.2:  # Minimum threshold
            return best[0]

        # Low confidence - default to research
        return AgentType.RESEARCH

    def _plan_task(self, message: str, context: ConversationContext) -> TaskPlan:
        """Create a plan for handling the user's request.

        This uses simple heuristics for speed. For complex multi-agent
        coordination, we could use an LLM planner.

        Args:
            message: User message.
            context: Conversation context.

        Returns:
            TaskPlan describing how to handle the request.
        """
        # Check if conversational
        if self._is_conversational(message):
            return TaskPlan(
                needs_specialist=False,
                is_conversational=True,
                reasoning="Conversational message - responding directly",
            )

        # Get scores from all specialists
        scores: dict[AgentType, float] = {}
        for agent_type, agent in self.specialists.items():
            score = agent.can_handle(message, context)
            scores[agent_type] = score

        # Check if multiple specialists are relevant
        relevant = [(t, s) for t, s in scores.items() if s >= 0.3]

        if not relevant:
            # Default to research
            return TaskPlan(
                needs_specialist=True,
                specialist_types=[AgentType.RESEARCH],
                reasoning="No strong match - using research agent for general search",
            )

        if len(relevant) == 1:
            # Single specialist
            return TaskPlan(
                needs_specialist=True,
                specialist_types=[relevant[0][0]],
                reasoning=f"Single domain match: {relevant[0][0].value}",
            )

        # Multiple specialists - check for multi-domain request
        # Keywords that suggest multi-domain
        multi_indicators = ["and", "also", "both", "plus", "as well"]
        message_lower = message.lower()

        if any(ind in message_lower for ind in multi_indicators):
            # Sort by score and take top matches
            sorted_relevant = sorted(relevant, key=lambda x: x[1], reverse=True)
            return TaskPlan(
                needs_specialist=True,
                specialist_types=[t for t, _ in sorted_relevant[:2]],
                reasoning=f"Multi-domain request: {', '.join(t.value for t, _ in sorted_relevant[:2])}",
            )

        # Single best match
        best = max(relevant, key=lambda x: x[1])
        return TaskPlan(
            needs_specialist=True,
            specialist_types=[best[0]],
            reasoning=f"Best match: {best[0].value} (score: {best[1]:.2f})",
        )

    def _get_chat_response(self, message: str, context: ConversationContext) -> str:
        """Generate a direct chat response without tools.

        Args:
            message: User message.
            context: Conversation context.

        Returns:
            Chat response string.
        """
        system = f"""You are Hani's friendly personal AI assistant.

Today's date: {datetime.now().strftime("%Y-%m-%d %A")}

You're having a casual conversation. Respond naturally and helpfully.
Keep responses concise but warm. You can help with:
- Calendar (check schedule, find availability)
- Email (search, create drafts)
- GitHub (PRs, issues, code search)
- General questions about documents and people

If asked what you can do, briefly explain your capabilities.
For greetings, respond warmly and offer to help."""

        messages = self._build_messages(context, message)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=system,
                messages=messages,
            )
            return self._extract_text(response) or "Hi! How can I help you today?"
        except Exception as e:
            logger.error(f"Chat response error: {e}")
            return "Hi! I'm here to help. What can I do for you?"

    def run(
        self,
        message: str,
        context: ConversationContext,
        max_iterations: int | None = None,
    ) -> AgentResult:
        """Execute the orchestrator.

        Routes to specialists or handles conversationally.

        Args:
            message: User message.
            context: Conversation context.
            max_iterations: Override for max iterations.

        Returns:
            AgentResult with combined response.
        """
        # Create task plan
        plan = self._plan_task(message, context)
        logger.info(f"Task plan: {plan.reasoning}")

        # Handle conversational messages directly with simple chat
        if plan.is_conversational:
            response_text = self._get_chat_response(message, context)
            return AgentResult(
                response=response_text,
                agent_type=self.AGENT_TYPE,
                iterations=1,
                metadata={"conversational": True},
            )

        # For non-conversational without specialists, use base run
        if not plan.needs_specialist:
            return super().run(message, context, max_iterations)

        # Route to specialist(s)
        if len(plan.specialist_types) == 1:
            # Single specialist - delegate fully
            specialist = self.specialists[plan.specialist_types[0]]
            return specialist.run(message, context)

        # Multiple specialists - execute in sequence and synthesize
        results: list[AgentResult] = []
        for agent_type in plan.specialist_types:
            specialist = self.specialists[agent_type]
            result = specialist.run(message, context)
            results.append(result)

        # Synthesize results
        return self._synthesize_results(message, results, context)

    def run_streaming(
        self,
        message: str,
        context: ConversationContext,
        max_iterations: int | None = None,
    ) -> Generator[AgentStreamEvent, None, AgentResult]:
        """Execute the orchestrator with streaming.

        Args:
            message: User message.
            context: Conversation context.
            max_iterations: Override for max iterations.

        Yields:
            AgentStreamEvent objects during execution.

        Returns:
            AgentResult with combined response.
        """
        # Create task plan
        plan = self._plan_task(message, context)
        logger.info(f"Task plan: {plan.reasoning}")

        # Handle conversational messages directly with simple chat
        if plan.is_conversational:
            yield AgentStreamEvent(
                event_type="thinking",
                data="Processing...",
                agent_type=self.AGENT_TYPE,
            )
            response_text = self._get_chat_response(message, context)
            yield AgentStreamEvent(
                event_type="done",
                data=response_text,
                agent_type=self.AGENT_TYPE,
            )
            return AgentResult(
                response=response_text,
                agent_type=self.AGENT_TYPE,
                iterations=1,
                metadata={"conversational": True},
            )

        # For non-conversational without specialists, use base streaming
        if not plan.needs_specialist:
            yield AgentStreamEvent(
                event_type="thinking",
                data="Processing your request...",
                agent_type=self.AGENT_TYPE,
            )
            return (yield from super().run_streaming(message, context, max_iterations))

        # Route to specialist(s)
        if len(plan.specialist_types) == 1:
            # Single specialist - delegate fully
            specialist = self.specialists[plan.specialist_types[0]]
            yield AgentStreamEvent(
                event_type="thinking",
                data=f"Routing to {specialist.AGENT_TYPE.value} specialist...",
                agent_type=self.AGENT_TYPE,
            )
            return (yield from specialist.run_streaming(message, context))

        # Multiple specialists
        results: list[AgentResult] = []
        for agent_type in plan.specialist_types:
            specialist = self.specialists[agent_type]
            yield AgentStreamEvent(
                event_type="thinking",
                data=f"Consulting {agent_type.value} specialist...",
                agent_type=self.AGENT_TYPE,
            )

            # Run each specialist (non-streaming for simplicity in multi-agent)
            result = specialist.run(message, context)
            results.append(result)

            yield AgentStreamEvent(
                event_type="tool_done",
                data=f"{agent_type.value} complete",
                agent_type=agent_type,
                tool_result=result.response[:100],
            )

        # Synthesize results
        yield AgentStreamEvent(
            event_type="thinking",
            data="Combining results...",
            agent_type=self.AGENT_TYPE,
        )

        final_result = self._synthesize_results(message, results, context)

        yield AgentStreamEvent(
            event_type="done",
            data=final_result.response,
            agent_type=self.AGENT_TYPE,
        )

        return final_result

    def _synthesize_results(
        self,
        original_message: str,
        results: list[AgentResult],
        context: ConversationContext,
    ) -> AgentResult:
        """Synthesize results from multiple specialists.

        Uses the LLM to create a coherent combined response.

        Args:
            original_message: Original user message.
            results: Results from specialist agents.
            context: Conversation context.

        Returns:
            Combined AgentResult.
        """
        # Build synthesis prompt
        results_text = "\n\n".join([
            f"=== {r.agent_type.value.upper()} AGENT ===\n{r.response}"
            for r in results
        ])

        synthesis_prompt = f"""The user asked: "{original_message}"

Multiple specialist agents provided the following information:

{results_text}

Please synthesize these results into a single, coherent response that:
1. Addresses all parts of the user's question
2. Organizes information logically
3. Avoids repetition
4. Is concise but complete

Provide the synthesized response:"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                messages=[{"role": "user", "content": synthesis_prompt}],
            )

            synthesized = response.content[0].text if response.content else ""

            # Combine metadata
            all_tool_calls = []
            total_iterations = 0
            for r in results:
                all_tool_calls.extend(r.tool_calls)
                total_iterations += r.iterations

            return AgentResult(
                response=synthesized,
                agent_type=self.AGENT_TYPE,
                tool_calls=all_tool_calls,
                iterations=total_iterations + 1,
                metadata={
                    "specialists_used": [r.agent_type.value for r in results],
                    "synthesized": True,
                },
            )

        except Exception as e:
            logger.error(f"Synthesis error: {e}", exc_info=True)
            # Fallback: concatenate responses
            combined = "\n\n".join([r.response for r in results])
            return AgentResult(
                response=combined,
                agent_type=self.AGENT_TYPE,
                tool_calls=[tc for r in results for tc in r.tool_calls],
                iterations=sum(r.iterations for r in results),
                metadata={"specialists_used": [r.agent_type.value for r in results]},
            )

    def can_handle(self, message: str, context: ConversationContext) -> float:
        """Orchestrator can handle everything."""
        return 1.0

    def get_specialist(self, agent_type: AgentType) -> BaseAgent | None:
        """Get a specialist agent by type.

        Args:
            agent_type: Type of specialist to get.

        Returns:
            The specialist agent, or None if not found.
        """
        return self.specialists.get(agent_type)

    def get_available_specialists(self) -> list[str]:
        """Get list of available specialist types.

        Returns:
            List of specialist type names.
        """
        return [t.value for t in self.specialists.keys()]
