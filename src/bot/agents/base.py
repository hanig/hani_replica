"""Base agent class for specialized domain agents."""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Generator

from anthropic import Anthropic

from ..conversation import ConversationContext
from ..tools import ToolResult, get_tool_schemas, TOOL_NAME_MAP
from ..user_memory import UserMemory
from ...config import ANTHROPIC_API_KEY, AGENT_MODEL

logger = logging.getLogger(__name__)


class AgentType(str, Enum):
    """Types of specialized agents."""
    CALENDAR = "calendar"
    EMAIL = "email"
    GITHUB = "github"
    RESEARCH = "research"
    ORCHESTRATOR = "orchestrator"


@dataclass
class AgentResult:
    """Result from an agent execution."""
    response: str
    agent_type: AgentType
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    iterations: int = 0
    success: bool = True
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "response": self.response,
            "agent_type": self.agent_type.value,
            "tool_calls": self.tool_calls,
            "iterations": self.iterations,
            "success": self.success,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class AgentStreamEvent:
    """Event emitted during agent streaming execution."""
    event_type: str  # "text_delta", "tool_start", "tool_done", "thinking", "error", "done"
    data: str = ""
    agent_type: AgentType | None = None
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    tool_result: str | None = None
    error: str | None = None
    iteration: int = 0


class BaseAgent(ABC):
    """Abstract base class for specialized domain agents.

    Each agent has:
    - A specific domain focus (calendar, email, GitHub, etc.)
    - A subset of relevant tools
    - A domain-specific system prompt
    - Optional integration with user memory
    """

    # Subclasses should override these
    AGENT_TYPE: AgentType = AgentType.ORCHESTRATOR
    MAX_ITERATIONS: int = 5

    def __init__(
        self,
        api_key: str | None = None,
        user_memory: UserMemory | None = None,
        model: str | None = None,
    ):
        """Initialize the agent.

        Args:
            api_key: Anthropic API key. Uses config default if not provided.
            user_memory: Optional UserMemory for context injection.
            model: Model to use. Defaults to AGENT_MODEL from config.
        """
        self.api_key = api_key or ANTHROPIC_API_KEY
        if not self.api_key:
            raise ValueError("Anthropic API key is required")

        self.client = Anthropic(api_key=self.api_key)
        self.model = model or AGENT_MODEL
        self.user_memory = user_memory

        # Initialize tool executor (lazy loaded)
        self._tool_executor = None

    @property
    def tool_executor(self):
        """Lazy-load tool executor."""
        if self._tool_executor is None:
            from ..executor import ToolExecutor
            self._tool_executor = ToolExecutor()
        return self._tool_executor

    @property
    @abstractmethod
    def tool_names(self) -> list[str]:
        """Return list of tool names this agent can use.

        Subclasses must implement this to define their tool subset.
        """
        pass

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Return the domain-specific system prompt.

        Subclasses must implement this to define their expertise.
        """
        pass

    @property
    def description(self) -> str:
        """Return a short description of this agent's capabilities.

        Used by the orchestrator to decide which agent to route to.
        """
        return f"{self.AGENT_TYPE.value} specialist agent"

    def get_tools(self) -> list[dict[str, Any]]:
        """Get tool schemas for this agent's tools.

        Returns:
            List of tool schemas in Claude API format.
        """
        all_tools = get_tool_schemas()
        return [t for t in all_tools if t["name"] in self.tool_names]

    def _build_system_prompt(self, context: ConversationContext) -> str:
        """Build the full system prompt with context injection.

        Args:
            context: Conversation context for user info.

        Returns:
            Complete system prompt string.
        """
        prompt = self.system_prompt

        # Add current date
        current_date = datetime.now().strftime("%Y-%m-%d %A")
        prompt = prompt.replace("{current_date}", current_date)

        # Inject user memory context if available
        if self.user_memory:
            user_context = self.user_memory.get_context_summary(context.user_id)
            if user_context:
                prompt += f"\n\nUser context:\n{user_context}"

        return prompt

    def _build_messages(
        self,
        context: ConversationContext,
        message: str,
    ) -> list[dict[str, Any]]:
        """Build messages list for the API call.

        Args:
            context: Conversation context with history.
            message: Current user message.

        Returns:
            List of messages for Claude API.
        """
        messages = []

        # Add recent history (limit to last 4 messages = 2 exchanges)
        recent = context.get_recent_history(4)
        for msg in recent:
            messages.append({
                "role": msg["role"],
                "content": msg["content"],
            })

        # Add current message
        messages.append({
            "role": "user",
            "content": message,
        })

        return messages

    def run(
        self,
        message: str,
        context: ConversationContext,
        max_iterations: int | None = None,
    ) -> AgentResult:
        """Execute the agent synchronously.

        Args:
            message: User message to process.
            context: Conversation context.
            max_iterations: Override for max iterations.

        Returns:
            AgentResult with response and metadata.
        """
        max_iter = max_iterations or self.MAX_ITERATIONS
        tool_calls_history = []
        iterations = 0

        system = self._build_system_prompt(context)
        messages = self._build_messages(context, message)
        tools = self.get_tools()

        while iterations < max_iter:
            iterations += 1
            logger.info(f"{self.AGENT_TYPE.value} agent iteration {iterations}")

            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=system,
                    tools=tools if tools else None,
                    messages=messages,
                )

                # Check if we're done
                if response.stop_reason == "end_turn":
                    text = self._extract_text(response)
                    return AgentResult(
                        response=text,
                        agent_type=self.AGENT_TYPE,
                        tool_calls=tool_calls_history,
                        iterations=iterations,
                    )

                # Handle tool use
                if response.stop_reason == "tool_use":
                    tool_results = []

                    for content in response.content:
                        if content.type == "tool_use":
                            tool_name = content.name
                            tool_input = content.input
                            tool_id = content.id

                            # Check for RespondToUserTool
                            if tool_name == "RespondToUserTool":
                                return AgentResult(
                                    response=tool_input.get("message", ""),
                                    agent_type=self.AGENT_TYPE,
                                    tool_calls=tool_calls_history,
                                    iterations=iterations,
                                )

                            logger.info(f"Executing tool: {tool_name}")

                            # Execute tool
                            result = self.tool_executor.execute(
                                tool_name,
                                tool_input,
                                context=context,
                            )

                            # Record tool call
                            tool_calls_history.append({
                                "tool": tool_name,
                                "input": tool_input,
                                "result": result.to_content()[:500],
                                "success": result.success,
                            })

                            if (
                                tool_name == "SendEmailTool"
                                and result.success
                                and isinstance(result.data, dict)
                                and result.data.get("requires_confirmation")
                            ):
                                confirmation = result.data.get("confirmation", {})
                                return AgentResult(
                                    response=confirmation.get(
                                        "text", "Please confirm sending this email."
                                    ),
                                    agent_type=self.AGENT_TYPE,
                                    tool_calls=tool_calls_history,
                                    iterations=iterations,
                                    metadata={
                                        "response_blocks": confirmation.get("blocks")
                                    },
                                )

                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tool_id,
                                "content": result.to_content(),
                            })

                    # Add to conversation
                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({"role": "user", "content": tool_results})

                else:
                    # Unexpected stop reason
                    text = self._extract_text(response)
                    return AgentResult(
                        response=text or "Task completed.",
                        agent_type=self.AGENT_TYPE,
                        tool_calls=tool_calls_history,
                        iterations=iterations,
                    )

            except Exception as e:
                logger.error(f"Agent error: {e}", exc_info=True)
                return AgentResult(
                    response=f"I encountered an error: {str(e)}",
                    agent_type=self.AGENT_TYPE,
                    tool_calls=tool_calls_history,
                    iterations=iterations,
                    success=False,
                    error=str(e),
                )

        # Max iterations reached
        return AgentResult(
            response="I reached the maximum number of steps. Please try a more specific request.",
            agent_type=self.AGENT_TYPE,
            tool_calls=tool_calls_history,
            iterations=iterations,
            success=False,
            error="Max iterations reached",
        )

    def run_streaming(
        self,
        message: str,
        context: ConversationContext,
        max_iterations: int | None = None,
    ) -> Generator[AgentStreamEvent, None, AgentResult]:
        """Execute the agent with streaming.

        Args:
            message: User message to process.
            context: Conversation context.
            max_iterations: Override for max iterations.

        Yields:
            AgentStreamEvent objects during execution.

        Returns:
            AgentResult with final response and metadata.
        """
        max_iter = max_iterations or self.MAX_ITERATIONS
        tool_calls_history = []
        iterations = 0

        system = self._build_system_prompt(context)
        messages = self._build_messages(context, message)
        tools = self.get_tools()

        while iterations < max_iter:
            iterations += 1
            logger.info(f"{self.AGENT_TYPE.value} agent streaming iteration {iterations}")

            try:
                with self.client.messages.stream(
                    model=self.model,
                    max_tokens=4096,
                    system=system,
                    tools=tools if tools else None,
                    messages=messages,
                ) as stream:
                    current_text = ""
                    tool_uses = []

                    for event in stream:
                        if event.type == "content_block_start":
                            if hasattr(event, "content_block"):
                                block = event.content_block
                                if block.type == "tool_use":
                                    tool_uses.append({
                                        "id": block.id,
                                        "name": block.name,
                                        "input": {},
                                    })
                                    yield AgentStreamEvent(
                                        event_type="tool_start",
                                        agent_type=self.AGENT_TYPE,
                                        tool_name=block.name,
                                        iteration=iterations,
                                    )

                        elif event.type == "content_block_delta":
                            if hasattr(event, "delta"):
                                delta = event.delta
                                if delta.type == "text_delta":
                                    current_text += delta.text
                                    yield AgentStreamEvent(
                                        event_type="text_delta",
                                        data=delta.text,
                                        agent_type=self.AGENT_TYPE,
                                        iteration=iterations,
                                    )

                    # Get final message
                    response = stream.get_final_message()

                # Process response
                if response.stop_reason == "end_turn":
                    text = self._extract_text(response)
                    yield AgentStreamEvent(
                        event_type="done",
                        data=text,
                        agent_type=self.AGENT_TYPE,
                        iteration=iterations,
                    )
                    return AgentResult(
                        response=text,
                        agent_type=self.AGENT_TYPE,
                        tool_calls=tool_calls_history,
                        iterations=iterations,
                    )

                if response.stop_reason == "tool_use":
                    tool_results = []

                    for content in response.content:
                        if content.type == "tool_use":
                            tool_name = content.name
                            tool_input = content.input
                            tool_id = content.id

                            # Check for RespondToUserTool
                            if tool_name == "RespondToUserTool":
                                response_text = tool_input.get("message", "")
                                yield AgentStreamEvent(
                                    event_type="done",
                                    data=response_text,
                                    agent_type=self.AGENT_TYPE,
                                    iteration=iterations,
                                )
                                return AgentResult(
                                    response=response_text,
                                    agent_type=self.AGENT_TYPE,
                                    tool_calls=tool_calls_history,
                                    iterations=iterations,
                                )

                            yield AgentStreamEvent(
                                event_type="thinking",
                                data=f"Using {tool_name}...",
                                agent_type=self.AGENT_TYPE,
                                tool_name=tool_name,
                                iteration=iterations,
                            )

                            result = self.tool_executor.execute(
                                tool_name,
                                tool_input,
                                context=context,
                            )

                            tool_calls_history.append({
                                "tool": tool_name,
                                "input": tool_input,
                                "result": result.to_content()[:500],
                                "success": result.success,
                            })

                            if (
                                tool_name == "SendEmailTool"
                                and result.success
                                and isinstance(result.data, dict)
                                and result.data.get("requires_confirmation")
                            ):
                                confirmation = result.data.get("confirmation", {})
                                response_text = confirmation.get(
                                    "text", "Please confirm sending this email."
                                )
                                yield AgentStreamEvent(
                                    event_type="done",
                                    data=response_text,
                                    agent_type=self.AGENT_TYPE,
                                    iteration=iterations,
                                )
                                return AgentResult(
                                    response=response_text,
                                    agent_type=self.AGENT_TYPE,
                                    tool_calls=tool_calls_history,
                                    iterations=iterations,
                                    metadata={
                                        "response_blocks": confirmation.get("blocks")
                                    },
                                )

                            yield AgentStreamEvent(
                                event_type="tool_done",
                                agent_type=self.AGENT_TYPE,
                                tool_name=tool_name,
                                tool_input=tool_input,
                                tool_result=result.to_content()[:200],
                                iteration=iterations,
                            )

                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tool_id,
                                "content": result.to_content(),
                            })

                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({"role": "user", "content": tool_results})

                else:
                    text = self._extract_text(response)
                    yield AgentStreamEvent(
                        event_type="done",
                        data=text or "Task completed.",
                        agent_type=self.AGENT_TYPE,
                        iteration=iterations,
                    )
                    return AgentResult(
                        response=text or "Task completed.",
                        agent_type=self.AGENT_TYPE,
                        tool_calls=tool_calls_history,
                        iterations=iterations,
                    )

            except Exception as e:
                logger.error(f"Agent streaming error: {e}", exc_info=True)
                yield AgentStreamEvent(
                    event_type="error",
                    error=str(e),
                    agent_type=self.AGENT_TYPE,
                    iteration=iterations,
                )
                return AgentResult(
                    response=f"I encountered an error: {str(e)}",
                    agent_type=self.AGENT_TYPE,
                    tool_calls=tool_calls_history,
                    iterations=iterations,
                    success=False,
                    error=str(e),
                )

        # Max iterations
        yield AgentStreamEvent(
            event_type="error",
            error="Max iterations reached",
            agent_type=self.AGENT_TYPE,
            iteration=iterations,
        )
        return AgentResult(
            response="I reached the maximum number of steps.",
            agent_type=self.AGENT_TYPE,
            tool_calls=tool_calls_history,
            iterations=iterations,
            success=False,
            error="Max iterations reached",
        )

    def _extract_text(self, response) -> str:
        """Extract text content from response."""
        for content in response.content:
            if content.type == "text":
                return content.text
        return ""

    def can_handle(self, message: str, context: ConversationContext) -> float:
        """Estimate how well this agent can handle a message.

        Args:
            message: User message.
            context: Conversation context.

        Returns:
            Confidence score from 0.0 to 1.0.
        """
        # Default implementation - subclasses can override for smarter routing
        return 0.0

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} type={self.AGENT_TYPE.value}>"
