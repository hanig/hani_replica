"""Slack event handlers for the bot."""

import logging
import re
import time
from typing import TYPE_CHECKING, Any, Literal

from slack_bolt import App

from ..config import SLACK_AUTHORIZED_USERS, BOT_MODE, ENABLE_STREAMING, STREAMING_UPDATE_INTERVAL
from .conversation import ConversationManager
from .formatters import format_error_message, format_help_message
from .intent_router import IntentRouter, Intent
from .security import SecurityGuard, SecurityLevel, ThreatType, get_security_guard
from .audit import AuditLogger, AuditEventType, get_audit_logger

if TYPE_CHECKING:
    from .user_memory import UserMemory
    from .feedback_loop import FeedbackLoop
    from .executor import AgentExecutor, StreamEvent
    from .agents.orchestrator import Orchestrator

logger = logging.getLogger(__name__)


def register_event_handlers(
    app: App,
    conversation_manager: ConversationManager,
    user_memory: "UserMemory | None" = None,
    feedback_loop: "FeedbackLoop | None" = None,
    mode: Literal["intent", "agent", "multi_agent"] | None = None,
    enable_streaming: bool | None = None,
) -> None:
    """Register all event handlers with the app.

    Args:
        app: Slack App instance.
        conversation_manager: Conversation manager instance.
        user_memory: Optional UserMemory instance for long-term memory.
        feedback_loop: Optional FeedbackLoop instance for interaction learning.
        mode: Bot mode - "intent" for legacy routing, "agent" for tool calling,
              "multi_agent" for specialist agent orchestration.
              Defaults to BOT_MODE from config.
        enable_streaming: Whether to enable streaming responses (agent/multi_agent mode).
                         Defaults to ENABLE_STREAMING from config.
    """
    # Determine mode and streaming
    bot_mode = mode or BOT_MODE
    streaming_enabled = enable_streaming if enable_streaming is not None else ENABLE_STREAMING
    logger.info(f"Bot running in '{bot_mode}' mode (streaming: {streaming_enabled})")

    # Initialize based on mode
    agent_executor: "AgentExecutor | None" = None
    orchestrator: "Orchestrator | None" = None
    intent_router: IntentRouter | None = None
    handlers: dict | None = None

    if bot_mode == "multi_agent":
        # Use multi-agent architecture with orchestrator
        from .agents.orchestrator import Orchestrator
        orchestrator = Orchestrator(user_memory=user_memory)
        logger.info(f"Initialized Orchestrator with specialists: {orchestrator.get_available_specialists()}")
    elif bot_mode == "agent":
        # Use new agent executor with tool calling
        from .executor import AgentExecutor
        agent_executor = AgentExecutor(user_memory=user_memory)
        logger.info("Initialized AgentExecutor for tool calling")
    else:
        # Use legacy intent routing (streaming not supported)
        streaming_enabled = False
        intent_router = IntentRouter()

        # Import handlers
        from .handlers.chat import ChatHandler
        from .handlers.search import SearchHandler
        from .handlers.calendar import CalendarHandler
        from .handlers.email import EmailHandler
        from .handlers.github import GitHubHandler
        from .handlers.briefing import BriefingHandler

        # Initialize handlers (lazy - they'll load resources when needed)
        handlers = {
            "chat": ChatHandler(user_memory=user_memory),
            "search": SearchHandler(),
            "calendar": CalendarHandler(),
            "email": EmailHandler(),
            "github": GitHubHandler(),
            "briefing": BriefingHandler(),
        }
        logger.info("Initialized legacy intent routing")

    @app.event("app_mention")
    def handle_mention(event: dict, say, client) -> None:
        """Handle @mentions of the bot."""
        _handle_message(event, say, client, is_dm=False)

    @app.event("message")
    def handle_dm(event: dict, say, client) -> None:
        """Handle direct messages to the bot."""
        # Only handle DMs (channel type "im")
        if event.get("channel_type") == "im":
            # Ignore bot's own messages
            if event.get("bot_id"):
                return
            _handle_message(event, say, client, is_dm=True)

    @app.action(re.compile(r"^confirm_action:.*"))
    def handle_confirm(ack, body, client) -> None:
        """Handle action confirmation buttons."""
        ack()
        _handle_action_confirmation(body, client, confirmed=True)

    @app.action(re.compile(r"^cancel_action:.*"))
    def handle_cancel(ack, body, client) -> None:
        """Handle action cancellation buttons."""
        ack()
        _handle_action_confirmation(body, client, confirmed=False)

    # Initialize security and audit
    security_guard = get_security_guard()
    audit_logger = get_audit_logger()

    def _handle_message(event: dict, say, client, is_dm: bool) -> None:
        """Common message handling logic."""
        start_time = time.time()
        user_id = event.get("user")
        channel_id = event.get("channel")
        thread_ts = event.get("thread_ts") or event.get("ts")
        text = event.get("text", "")

        # Log message received
        audit_logger.log_message_received(
            user_id=user_id,
            channel_id=channel_id,
            message=text,
            thread_ts=thread_ts,
            is_mention=not is_dm,
        )

        # Check authorization
        if not _is_authorized(user_id):
            say(
                text="Sorry, you're not authorized to use this bot.",
                thread_ts=thread_ts,
            )
            audit_logger.log_security_event(
                event_type=AuditEventType.UNAUTHORIZED,
                user_id=user_id,
                description="Unauthorized access attempt",
                blocked=True,
            )
            return

        # Check rate limit
        allowed, rate_event = security_guard.check_rate_limit(user_id)
        if not allowed:
            remaining = rate_event.metadata.get("remaining_seconds", 0) if rate_event else 0
            say(
                text=f"You're sending messages too quickly. Please wait {remaining} seconds.",
                thread_ts=thread_ts,
            )
            audit_logger.log_security_event(
                event_type=AuditEventType.RATE_LIMITED,
                user_id=user_id,
                description="Rate limit exceeded",
                details={"remaining_seconds": remaining},
                blocked=True,
            )
            return

        # Strip bot mention from text
        text = _strip_bot_mention(text, client)

        if not text.strip():
            say(text=format_help_message(), thread_ts=thread_ts)
            return

        # Sanitize input for security
        sanitized_text, security_events = security_guard.sanitize_input(text, user_id)

        # Log any security events
        for sec_event in security_events:
            audit_logger.log_security_event(
                event_type=(
                    AuditEventType.SECURITY_BLOCKED if sec_event.blocked
                    else AuditEventType.SECURITY_WARNING
                ),
                user_id=user_id,
                description=sec_event.description,
                details=sec_event.metadata,
                blocked=sec_event.blocked,
            )

        # If input was blocked, inform user
        if not sanitized_text and security_events:
            say(
                text="Your message couldn't be processed due to security concerns. Please rephrase your request.",
                thread_ts=thread_ts,
            )
            return

        # Use sanitized text for processing
        text = sanitized_text

        # Get or create conversation context
        context = conversation_manager.get_or_create(
            user_id=user_id,
            channel_id=channel_id,
            thread_ts=thread_ts,
        )

        # Add user message to history
        context.add_message("user", text)

        try:
            if bot_mode == "multi_agent" and orchestrator:
                if streaming_enabled:
                    # Use streaming multi-agent orchestrator
                    response, intent = _handle_with_multi_agent_streaming(
                        text=text,
                        context=context,
                        orchestrator=orchestrator,
                        client=client,
                        channel_id=channel_id,
                        thread_ts=thread_ts,
                    )
                else:
                    # Use non-streaming multi-agent orchestrator
                    response, intent = _handle_with_multi_agent(
                        text=text,
                        context=context,
                        orchestrator=orchestrator,
                    )
            elif bot_mode == "agent" and agent_executor:
                if streaming_enabled:
                    # Use streaming agent executor
                    response, intent = _handle_with_agent_streaming(
                        text=text,
                        context=context,
                        agent_executor=agent_executor,
                        client=client,
                        channel_id=channel_id,
                        thread_ts=thread_ts,
                    )
                else:
                    # Use non-streaming agent executor
                    response, intent = _handle_with_agent(
                        text=text,
                        context=context,
                        agent_executor=agent_executor,
                    )
            else:
                # Use legacy intent routing
                # Check for pending action that needs input
                if context.pending_action:
                    response = _handle_pending_action_input(context, text, handlers)
                    intent = None
                else:
                    # Route to appropriate handler
                    response, intent = _route_message(
                        text=text,
                        context=context,
                        intent_router=intent_router,
                        handlers=handlers,
                    )

            # Add assistant response to history
            if response:
                context.add_message("assistant", response.get("text", ""))

            # Persist conversation after each interaction
            conversation_manager.update(context)

            # Record query pattern in feedback loop
            if feedback_loop and intent:
                try:
                    # Normalize query pattern (lowercase, strip extra spaces)
                    pattern = " ".join(text.lower().split())
                    feedback_loop.record_query_pattern(
                        user_id=user_id,
                        pattern=pattern,
                        intent=intent.intent,
                        success=response is not None,
                    )
                except Exception as e:
                    logger.warning(f"Failed to record query pattern: {e}")

            # Calculate duration
            duration_ms = int((time.time() - start_time) * 1000)

            # Send response
            if response:
                _send_response(say, response, thread_ts)
                audit_logger.log_message_sent(
                    channel_id=channel_id,
                    message=response.get("text", ""),
                    thread_ts=thread_ts,
                    user_id=user_id,
                )
            else:
                fallback_msg = "I'm not sure how to help with that. Try asking about your calendar, emails, or searching for information."
                say(text=fallback_msg, thread_ts=thread_ts)
                audit_logger.log_message_sent(
                    channel_id=channel_id,
                    message=fallback_msg,
                    thread_ts=thread_ts,
                    user_id=user_id,
                )

        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)
            error_msg = format_error_message(str(e))
            say(text=error_msg, thread_ts=thread_ts)
            audit_logger.log_error(
                error=str(e),
                user_id=user_id,
                channel_id=channel_id,
                details={"message": text[:100]},
            )

    def _handle_action_confirmation(body: dict, client, confirmed: bool) -> None:
        """Handle action confirmation or cancellation."""
        user_id = body.get("user", {}).get("id")
        action_id = body.get("actions", [{}])[0].get("action_id", "")
        channel_id = body.get("channel", {}).get("id")
        message_ts = body.get("message", {}).get("ts")

        # Extract action key from action_id (e.g., "confirm_action:abc123")
        action_key = action_id.split(":", 1)[1] if ":" in action_id else ""

        # Get conversation context
        context = conversation_manager.get(user_id, channel_id)

        if not context or not context.pending_action:
            # Update message to show expired
            client.chat_update(
                channel=channel_id,
                ts=message_ts,
                text="This action has expired.",
                blocks=[],
            )
            return

        pending = context.pending_action
        action_type = pending.get_action_type() if hasattr(pending, 'get_action_type') else "unknown"

        # Validate action with security guard
        allowed, sec_event = security_guard.validate_action(
            action_type=action_type,
            user_id=user_id,
            context={"action_key": action_key},
        )
        if not allowed:
            audit_logger.log_security_event(
                event_type=AuditEventType.SECURITY_BLOCKED,
                user_id=user_id,
                description=f"Action blocked: {action_type}",
                blocked=True,
            )
            client.chat_update(
                channel=channel_id,
                ts=message_ts,
                text="This action was blocked for security reasons.",
                blocks=[],
            )
            context.pending_action = None
            return

        if confirmed:
            # Log action confirmation
            audit_logger.log_action(
                action_type=action_type,
                event_type=AuditEventType.ACTION_CONFIRMED,
                user_id=user_id,
                details={"action_key": action_key},
            )

            try:
                # Execute the action
                result = pending.execute()

                # Log successful execution
                audit_logger.log_action(
                    action_type=action_type,
                    event_type=AuditEventType.ACTION_EXECUTED,
                    user_id=user_id,
                    details={"result": str(result)[:200]},
                    success=True,
                )

                # Update message with success
                client.chat_update(
                    channel=channel_id,
                    ts=message_ts,
                    text=f"Action completed: {result.get('message', 'Success')}",
                    blocks=[],
                )

            except Exception as e:
                logger.error(f"Error executing action: {e}")
                audit_logger.log_action(
                    action_type=action_type,
                    event_type=AuditEventType.ACTION_EXECUTED,
                    user_id=user_id,
                    success=False,
                    error=str(e),
                )
                client.chat_update(
                    channel=channel_id,
                    ts=message_ts,
                    text=f"Error executing action: {str(e)}",
                    blocks=[],
                )
        else:
            # Log action cancellation
            audit_logger.log_action(
                action_type=action_type,
                event_type=AuditEventType.ACTION_CANCELLED,
                user_id=user_id,
                details={"action_key": action_key},
            )

            # Update message with cancellation
            client.chat_update(
                channel=channel_id,
                ts=message_ts,
                text="Action cancelled.",
                blocks=[],
            )

        # Clear pending action
        context.pending_action = None


def _handle_with_agent(
    text: str,
    context,
    agent_executor: "AgentExecutor",
) -> tuple[dict[str, Any] | None, Intent | None]:
    """Handle message using agent executor with tool calling.

    Args:
        text: User message text.
        context: Conversation context.
        agent_executor: Agent executor instance.

    Returns:
        Tuple of (response dictionary, None for intent since agent doesn't use intents).
    """
    try:
        result = agent_executor.run(message=text, context=context)

        if result.success:
            response = {"text": result.response}

            # Log tool calls for debugging
            if result.tool_calls:
                logger.info(f"Agent used {len(result.tool_calls)} tools in {result.iterations} iterations")
                for tc in result.tool_calls:
                    logger.debug(f"  Tool: {tc['tool']} - Success: {tc['success']}")

            return response, None
        else:
            logger.warning(f"Agent execution failed: {result.error}")
            return {"text": result.response}, None

    except Exception as e:
        logger.error(f"Error in agent execution: {e}", exc_info=True)
        return {"text": f"I encountered an error processing your request: {str(e)}"}, None


def _handle_with_agent_streaming(
    text: str,
    context,
    agent_executor: "AgentExecutor",
    client,
    channel_id: str,
    thread_ts: str,
) -> tuple[dict[str, Any] | None, Intent | None]:
    """Handle message using agent executor with streaming responses.

    Posts an initial "Thinking..." message and updates it as response chunks arrive.

    Args:
        text: User message text.
        context: Conversation context.
        agent_executor: Agent executor instance.
        client: Slack WebClient for message updates.
        channel_id: Channel to post to.
        thread_ts: Thread timestamp for replies.

    Returns:
        Tuple of (response dictionary, None for intent).
    """
    from .executor import StreamEventType

    try:
        # Post initial "Thinking..." message
        initial_response = client.chat_postMessage(
            channel=channel_id,
            thread_ts=thread_ts,
            text="Thinking...",
        )
        message_ts = initial_response["ts"]

        # Track state for message updates
        accumulated_text = ""
        current_status = "Thinking..."
        last_update_time = time.time()
        tool_count = 0

        # Process streaming events
        stream_gen = agent_executor.run_streaming(message=text, context=context)
        final_result = None

        for event in stream_gen:
            try:
                if event.event_type == StreamEventType.TEXT_DELTA:
                    # Accumulate text chunks
                    accumulated_text += event.data

                    # Update message at intervals to avoid rate limiting
                    current_time = time.time()
                    if current_time - last_update_time >= STREAMING_UPDATE_INTERVAL:
                        display_text = accumulated_text if accumulated_text else current_status
                        _update_message_safe(client, channel_id, message_ts, display_text)
                        last_update_time = current_time

                elif event.event_type == StreamEventType.TOOL_START:
                    # Show tool being used
                    tool_count += 1
                    current_status = f"Using {event.tool_name}..."
                    if not accumulated_text:
                        _update_message_safe(client, channel_id, message_ts, current_status)
                        last_update_time = time.time()

                elif event.event_type == StreamEventType.TOOL_DONE:
                    # Tool completed
                    logger.debug(f"Tool {event.tool_name} completed")

                elif event.event_type == StreamEventType.THINKING:
                    # Status update
                    current_status = event.data
                    if not accumulated_text:
                        _update_message_safe(client, channel_id, message_ts, current_status)
                        last_update_time = time.time()

                elif event.event_type == StreamEventType.TEXT_DONE:
                    # Text block completed
                    pass

                elif event.event_type == StreamEventType.ERROR:
                    # Error occurred
                    error_text = f"Error: {event.error}"
                    _update_message_safe(client, channel_id, message_ts, error_text)
                    return {"text": error_text}, None

                elif event.event_type == StreamEventType.DONE:
                    # Streaming complete
                    final_text = event.data or accumulated_text
                    _update_message_safe(client, channel_id, message_ts, final_text)

            except Exception as e:
                logger.warning(f"Error processing stream event: {e}")

        # Get final result from generator
        try:
            final_result = stream_gen.send(None)
        except StopIteration as e:
            final_result = e.value

        if final_result:
            # Ensure final message is updated
            final_text = final_result.response
            _update_message_safe(client, channel_id, message_ts, final_text)

            # Log tool usage
            if final_result.tool_calls:
                logger.info(
                    f"Agent used {len(final_result.tool_calls)} tools "
                    f"in {final_result.iterations} iterations (streaming)"
                )

            # Return response (already posted via streaming updates)
            return {"text": final_text, "_streaming_sent": True}, None
        else:
            # Fallback if no result
            fallback_text = accumulated_text or "I'm not sure how to respond."
            _update_message_safe(client, channel_id, message_ts, fallback_text)
            return {"text": fallback_text, "_streaming_sent": True}, None

    except Exception as e:
        logger.error(f"Error in streaming agent execution: {e}", exc_info=True)
        error_response = f"I encountered an error: {str(e)}"

        # Try to update the message with the error
        try:
            if message_ts:
                _update_message_safe(client, channel_id, message_ts, error_response)
                return {"text": error_response, "_streaming_sent": True}, None
        except Exception:
            pass

        return {"text": error_response}, None


def _handle_with_multi_agent(
    text: str,
    context,
    orchestrator: "Orchestrator",
) -> tuple[dict[str, Any] | None, Intent | None]:
    """Handle message using multi-agent orchestrator.

    Args:
        text: User message text.
        context: Conversation context.
        orchestrator: Orchestrator instance.

    Returns:
        Tuple of (response dictionary, None for intent).
    """
    try:
        result = orchestrator.run(message=text, context=context)

        if result.success:
            response = {"text": result.response}

            # Log agent usage
            if result.metadata.get("specialists_used"):
                logger.info(
                    f"Orchestrator used specialists: {result.metadata['specialists_used']} "
                    f"with {len(result.tool_calls)} tool calls in {result.iterations} iterations"
                )
            elif result.tool_calls:
                logger.info(
                    f"Orchestrator used {len(result.tool_calls)} tools "
                    f"in {result.iterations} iterations"
                )

            return response, None
        else:
            logger.warning(f"Orchestrator execution failed: {result.error}")
            return {"text": result.response}, None

    except Exception as e:
        logger.error(f"Error in multi-agent execution: {e}", exc_info=True)
        return {"text": f"I encountered an error processing your request: {str(e)}"}, None


def _handle_with_multi_agent_streaming(
    text: str,
    context,
    orchestrator: "Orchestrator",
    client,
    channel_id: str,
    thread_ts: str,
) -> tuple[dict[str, Any] | None, Intent | None]:
    """Handle message using multi-agent orchestrator with streaming.

    Posts an initial "Thinking..." message and updates as specialists work.

    Args:
        text: User message text.
        context: Conversation context.
        orchestrator: Orchestrator instance.
        client: Slack WebClient for message updates.
        channel_id: Channel to post to.
        thread_ts: Thread timestamp for replies.

    Returns:
        Tuple of (response dictionary, None for intent).
    """
    try:
        # Post initial message
        initial_response = client.chat_postMessage(
            channel=channel_id,
            thread_ts=thread_ts,
            text="Thinking...",
        )
        message_ts = initial_response["ts"]

        # Track state
        accumulated_text = ""
        current_status = "Thinking..."
        last_update_time = time.time()

        # Process streaming events
        stream_gen = orchestrator.run_streaming(message=text, context=context)
        final_result = None

        for event in stream_gen:
            try:
                if event.event_type == "text_delta":
                    accumulated_text += event.data

                    # Rate-limited updates
                    current_time = time.time()
                    if current_time - last_update_time >= STREAMING_UPDATE_INTERVAL:
                        display_text = accumulated_text if accumulated_text else current_status
                        _update_message_safe(client, channel_id, message_ts, display_text)
                        last_update_time = current_time

                elif event.event_type == "tool_start":
                    current_status = f"Using {event.tool_name}..."
                    if not accumulated_text:
                        _update_message_safe(client, channel_id, message_ts, current_status)
                        last_update_time = time.time()

                elif event.event_type == "thinking":
                    current_status = event.data
                    if not accumulated_text:
                        _update_message_safe(client, channel_id, message_ts, current_status)
                        last_update_time = time.time()

                elif event.event_type == "tool_done":
                    # Show which specialist completed
                    if event.agent_type:
                        logger.debug(f"Specialist {event.agent_type.value} completed")

                elif event.event_type == "error":
                    error_text = f"Error: {event.error}"
                    _update_message_safe(client, channel_id, message_ts, error_text)
                    return {"text": error_text, "_streaming_sent": True}, None

                elif event.event_type == "done":
                    final_text = event.data or accumulated_text
                    _update_message_safe(client, channel_id, message_ts, final_text)

            except Exception as e:
                logger.warning(f"Error processing stream event: {e}")

        # Get final result
        try:
            final_result = stream_gen.send(None)
        except StopIteration as e:
            final_result = e.value

        if final_result:
            final_text = final_result.response
            _update_message_safe(client, channel_id, message_ts, final_text)

            # Log usage
            if final_result.metadata.get("specialists_used"):
                logger.info(
                    f"Orchestrator (streaming) used specialists: "
                    f"{final_result.metadata['specialists_used']}"
                )

            return {"text": final_text, "_streaming_sent": True}, None
        else:
            fallback_text = accumulated_text or "I'm not sure how to respond."
            _update_message_safe(client, channel_id, message_ts, fallback_text)
            return {"text": fallback_text, "_streaming_sent": True}, None

    except Exception as e:
        logger.error(f"Error in multi-agent streaming: {e}", exc_info=True)
        error_response = f"I encountered an error: {str(e)}"

        try:
            if message_ts:
                _update_message_safe(client, channel_id, message_ts, error_response)
                return {"text": error_response, "_streaming_sent": True}, None
        except Exception:
            pass

        return {"text": error_response}, None


def _update_message_safe(client, channel_id: str, message_ts: str, text: str) -> None:
    """Safely update a Slack message, handling errors gracefully.

    Args:
        client: Slack WebClient.
        channel_id: Channel ID.
        message_ts: Message timestamp.
        text: New message text.
    """
    try:
        client.chat_update(
            channel=channel_id,
            ts=message_ts,
            text=text,
        )
    except Exception as e:
        logger.warning(f"Failed to update message: {e}")


def _is_authorized(user_id: str) -> bool:
    """Check if a user is authorized to use the bot."""
    # If no authorized users configured, allow all
    if not SLACK_AUTHORIZED_USERS:
        return True
    return user_id in SLACK_AUTHORIZED_USERS


def _strip_bot_mention(text: str, client) -> str:
    """Remove bot mention from message text."""
    # Get bot user ID
    try:
        auth_response = client.auth_test()
        bot_user_id = auth_response.get("user_id", "")
        # Remove <@BOT_ID> pattern
        text = re.sub(f"<@{bot_user_id}>", "", text)
    except Exception:
        pass
    return text.strip()


def _route_message(
    text: str,
    context,
    intent_router: IntentRouter,
    handlers: dict,
) -> tuple[dict[str, Any] | None, Intent | None]:
    """Route message to appropriate handler based on intent.

    Returns:
        Tuple of (response dictionary, classified intent).
    """
    # Classify intent
    intent = intent_router.classify(text, context.history)

    logger.info(f"Classified intent: {intent.intent} with entities: {intent.entities}")

    # Map intent to handler
    intent_to_handler = {
        "chat": "chat",
        "search": "search",
        "calendar_check": "calendar",
        "calendar_availability": "calendar",
        "email_search": "email",
        "email_draft": "email",
        "github_search": "github",
        "github_create_issue": "github",
        "github_list_prs": "github",
        "briefing": "briefing",
        "help": None,  # Handled specially
    }

    handler_name = intent_to_handler.get(intent.intent)

    if handler_name is None:
        if intent.intent == "help":
            return {"text": format_help_message()}, intent
        return None, intent

    handler = handlers.get(handler_name)
    if not handler:
        return {"text": f"Handler '{handler_name}' not available."}, intent

    # Execute handler
    return handler.handle(intent, context), intent


def _handle_pending_action_input(context, text: str, handlers: dict) -> dict[str, Any]:
    """Handle input for a pending action that needs more information."""
    pending = context.pending_action

    # Update action with new input
    pending.update_from_input(text)

    # Check if action is ready
    if pending.is_ready():
        # Return confirmation prompt
        return pending.get_confirmation_prompt()
    else:
        # Ask for next required input
        return {"text": pending.get_next_prompt()}


def _send_response(say, response: dict, thread_ts: str) -> None:
    """Send response to Slack.

    Args:
        say: Slack say function.
        response: Response dictionary with 'text' and optionally 'blocks'.
        thread_ts: Thread timestamp for replies.
    """
    # Skip if response was already sent via streaming
    if response.get("_streaming_sent"):
        return

    kwargs = {"thread_ts": thread_ts}

    if "blocks" in response:
        kwargs["blocks"] = response["blocks"]
        kwargs["text"] = response.get("text", "")
    else:
        kwargs["text"] = response.get("text", "")

    say(**kwargs)
