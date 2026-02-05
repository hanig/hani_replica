"""Main Slack bot application using Socket Mode."""

import atexit
import logging
from typing import Callable

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from ..config import SLACK_APP_TOKEN, SLACK_BOT_TOKEN, SLACK_AUTHORIZED_USERS, BOT_MODE, ENABLE_STREAMING
from .conversation import ConversationManager
from .event_handlers import register_event_handlers
from .feedback_loop import FeedbackLoop
from .formatters import format_error_message
from .heartbeat import HeartbeatManager
from .proactive_settings import ProactiveSettingsStore
from .user_memory import UserMemory

logger = logging.getLogger(__name__)


def create_bot_app(
    bot_token: str | None = None,
    app_token: str | None = None,
    enable_persistence: bool = True,
    enable_proactive: bool = True,
    mode: str | None = None,
    enable_streaming: bool | None = None,
) -> tuple[App, SocketModeHandler, BackgroundScheduler | None]:
    """Create and configure the Slack bot application.

    Args:
        bot_token: Slack bot token. Defaults to environment variable.
        app_token: Slack app token for Socket Mode. Defaults to environment variable.
        enable_persistence: Whether to enable persistent memory (default True).
        enable_proactive: Whether to enable proactive features (default True).
        mode: Bot mode - "intent" or "agent". Defaults to BOT_MODE from config.
        enable_streaming: Whether to enable streaming responses. Defaults to ENABLE_STREAMING.

    Returns:
        Tuple of (App, SocketModeHandler, BackgroundScheduler or None).
    """
    bot_token = bot_token or SLACK_BOT_TOKEN
    app_token = app_token or SLACK_APP_TOKEN
    bot_mode = mode or BOT_MODE
    streaming = enable_streaming if enable_streaming is not None else ENABLE_STREAMING

    if not bot_token:
        raise ValueError("SLACK_BOT_TOKEN is required")
    if not app_token:
        raise ValueError("SLACK_APP_TOKEN is required for Socket Mode")

    # Create the app
    app = App(token=bot_token)

    # Initialize conversation manager with persistence
    conversation_manager = ConversationManager(persist=enable_persistence)

    # Initialize memory systems
    user_memory = UserMemory() if enable_persistence else None
    feedback_loop = FeedbackLoop() if enable_persistence else None

    if enable_persistence:
        logger.info("Persistent memory enabled")

        # Register shutdown handler to persist conversations
        def on_shutdown():
            logger.info("Persisting conversations on shutdown...")
            conversation_manager.persist_all()

        atexit.register(on_shutdown)

    # Register event handlers with memory systems
    register_event_handlers(
        app,
        conversation_manager,
        user_memory=user_memory,
        feedback_loop=feedback_loop,
        mode=bot_mode,
        enable_streaming=streaming,
    )

    # Add global error handler
    @app.error
    def global_error_handler(error, body, logger):
        logger.error(f"Error: {error}")
        logger.error(f"Request body: {body}")

    # Create Socket Mode handler
    handler = SocketModeHandler(app, app_token)

    # Set up proactive features
    scheduler = None
    if enable_proactive:
        scheduler = _setup_proactive_scheduler(app.client, enable_persistence)
        logger.info("Proactive features enabled")

    logger.info("Slack bot app created successfully")
    return app, handler, scheduler


def _setup_proactive_scheduler(
    slack_client,
    enable_persistence: bool = True,
) -> BackgroundScheduler:
    """Set up the background scheduler for proactive features.

    Args:
        slack_client: Slack WebClient for sending messages.
        enable_persistence: Whether persistence is enabled.

    Returns:
        Configured BackgroundScheduler instance.
    """
    # Initialize proactive settings store
    settings_store = ProactiveSettingsStore() if enable_persistence else None

    # Initialize heartbeat manager
    heartbeat = HeartbeatManager(
        slack_client=slack_client,
        settings_store=settings_store,
    )

    # Create scheduler
    scheduler = BackgroundScheduler(
        job_defaults={
            "coalesce": True,  # Combine missed runs
            "max_instances": 1,  # Only one instance at a time
            "misfire_grace_time": 60,  # Allow 60 seconds for misfires
        }
    )

    # Add calendar reminder check job (every 5 minutes)
    scheduler.add_job(
        heartbeat.check_calendar_reminders,
        IntervalTrigger(minutes=5),
        id="calendar_reminders",
        name="Check calendar reminders",
        replace_existing=True,
    )

    # Add important email check job (every 10 minutes)
    scheduler.add_job(
        heartbeat.check_important_emails,
        IntervalTrigger(minutes=10),
        id="email_alerts",
        name="Check important emails",
        replace_existing=True,
    )

    # Add daily briefing job (runs hourly, checks per-user settings)
    scheduler.add_job(
        heartbeat.send_daily_briefings,
        CronTrigger(minute=0),  # Run at the top of every hour
        id="daily_briefing",
        name="Check daily briefings",
        replace_existing=True,
    )

    # Add cleanup job (daily at 3 AM)
    scheduler.add_job(
        heartbeat.cleanup,
        CronTrigger(hour=3, minute=0),
        id="cleanup",
        name="Clean up old notifications",
        replace_existing=True,
    )

    logger.info("Proactive scheduler configured with 4 jobs")
    return scheduler


def run_bot(
    bot_token: str | None = None,
    app_token: str | None = None,
    enable_persistence: bool = True,
    enable_proactive: bool = True,
    mode: str | None = None,
    enable_streaming: bool | None = None,
) -> None:
    """Run the Slack bot.

    Args:
        bot_token: Slack bot token.
        app_token: Slack app token for Socket Mode.
        enable_persistence: Whether to enable persistent memory.
        enable_proactive: Whether to enable proactive features.
        mode: Bot mode - "intent", "agent", or "multi_agent". Defaults to BOT_MODE from config.
        enable_streaming: Whether to enable streaming responses.
    """
    bot_mode = mode or BOT_MODE
    streaming = enable_streaming if enable_streaming is not None else ENABLE_STREAMING

    app, handler, scheduler = create_bot_app(
        bot_token, app_token, enable_persistence, enable_proactive,
        mode=bot_mode, enable_streaming=streaming
    )

    logger.info("Starting Slack bot in Socket Mode...")
    print("Bot is running! Press Ctrl+C to stop.")

    # Mode description
    mode_descriptions = {
        "intent": "legacy intent routing",
        "agent": "single agent with tool calling",
        "multi_agent": "orchestrator with specialist agents",
    }
    print(f"Mode: {bot_mode} ({mode_descriptions.get(bot_mode, 'unknown')})")

    if bot_mode in ("agent", "multi_agent"):
        print(f"Streaming: {'enabled' if streaming else 'disabled'}")
    if enable_persistence:
        print("Persistent memory is enabled - conversations will survive restarts.")
    if enable_proactive and scheduler:
        print("Proactive features are enabled:")
        # Show actual user settings
        settings_store = ProactiveSettingsStore()
        # Get first authorized user's settings for display
        if SLACK_AUTHORIZED_USERS:
            user_settings = settings_store.get(SLACK_AUTHORIZED_USERS[0])
            if user_settings.calendar_reminders_enabled:
                print(f"  - Calendar reminders ({user_settings.reminder_minutes_before} min before meetings)")
            else:
                print("  - Calendar reminders (disabled)")
            if user_settings.email_alerts_enabled:
                print("  - Important email alerts")
            else:
                print("  - Important email alerts (disabled)")
            if user_settings.daily_briefing_enabled:
                days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                day_names = [days[d] for d in user_settings.briefing_days]
                if len(day_names) == 7:
                    day_str = "daily"
                elif day_names == ["Mon", "Tue", "Wed", "Thu", "Fri"]:
                    day_str = "weekdays"
                else:
                    day_str = ", ".join(day_names)
                print(f"  - Daily briefings ({user_settings.briefing_hour}:{user_settings.briefing_minute:02d} AM {day_str})")
            else:
                print("  - Daily briefings (disabled)")
        else:
            print("  - Calendar reminders, email alerts, daily briefings (no users configured)")

    try:
        # Start the scheduler if enabled
        if scheduler:
            scheduler.start()
            logger.info("Background scheduler started")

        # Start the Slack handler
        handler.start()

    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot error: {e}")
        raise
    finally:
        # Clean up scheduler
        if scheduler and scheduler.running:
            scheduler.shutdown(wait=False)
            logger.info("Background scheduler stopped")


class BotContext:
    """Context object passed to handlers."""

    def __init__(
        self,
        app: App,
        conversation_manager: ConversationManager,
        user_memory: UserMemory | None = None,
        feedback_loop: FeedbackLoop | None = None,
        heartbeat_manager: HeartbeatManager | None = None,
        proactive_settings: ProactiveSettingsStore | None = None,
    ):
        """Initialize bot context.

        Args:
            app: Slack App instance.
            conversation_manager: Conversation manager instance.
            user_memory: Optional UserMemory instance.
            feedback_loop: Optional FeedbackLoop instance.
            heartbeat_manager: Optional HeartbeatManager instance.
            proactive_settings: Optional ProactiveSettingsStore instance.
        """
        self.app = app
        self.conversations = conversation_manager
        self.user_memory = user_memory
        self.feedback_loop = feedback_loop
        self.heartbeat = heartbeat_manager
        self.proactive_settings = proactive_settings

        # Lazy-loaded components
        self._query_engine = None
        self._semantic_indexer = None
        self._multi_google = None
        self._github_client = None

    @property
    def query_engine(self):
        """Get query engine (lazy loaded)."""
        if self._query_engine is None:
            from ..query.engine import QueryEngine
            self._query_engine = QueryEngine()
        return self._query_engine

    @property
    def semantic_indexer(self):
        """Get semantic indexer (lazy loaded)."""
        if self._semantic_indexer is None:
            from ..semantic.semantic_indexer import SemanticIndexer
            self._semantic_indexer = SemanticIndexer()
        return self._semantic_indexer

    @property
    def multi_google(self):
        """Get multi-Google manager (lazy loaded)."""
        if self._multi_google is None:
            from ..integrations.google_multi import MultiGoogleManager
            self._multi_google = MultiGoogleManager()
        return self._multi_google

    @property
    def github_client(self):
        """Get GitHub client (lazy loaded)."""
        if self._github_client is None:
            from ..integrations.github_client import GitHubClient
            self._github_client = GitHubClient()
        return self._github_client
