# Engram

A personal knowledge graph system that aggregates data from multiple Google accounts, GitHub, Slack, Notion, Todoist, and Zotero, with semantic search and an interactive Slack bot interface powered by intelligent agents.

> Disclaimer: This project is built by a scientist, not a security specialist. Review, harden, and use it at your own discretion, especially before handling sensitive data or deploying in production environments.

## Public Deployment Warning

This project is designed first for personal/local use. Do not expose it publicly or use it in multi-user/production environments without a full security review, least-privilege credentials, network hardening, and strict access controls.

## Features

### Core Capabilities
- **Multi-Account Google Integration**: Sync Gmail, Google Drive, and Google Calendar from up to 6 accounts with tiered search (primary accounts searched first)
- **Google Write Capabilities**: Create email drafts, create/modify/cancel calendar events, and comment/reply/resolve comments on Google Docs, with confirmation before writes
- **Zotero Integration**: Search papers, add references by DOI/URL with automatic metadata extraction (CrossRef + page scraping)
- **Notion & Todoist**: Search pages, manage tasks, create content
- **Knowledge Graph**: SQLite-based storage of entities (people, repos, files) and content with relationship tracking
- **Semantic Search**: OpenAI embeddings (text-embedding-3-large) with ChromaDB vector store for intelligent content retrieval
- **Slack Bot**: Interactive assistant using Socket Mode (no public URL required)
- **Daily Briefings**: Aggregated calendar, email counts, GitHub activity, and Todoist overdue tasks

### Advanced Agent Features
- **Natural Conversation**: Chat naturally without triggering tool searches - greetings, questions about the bot, and general conversation are handled intelligently
- **Multi-Agent Architecture**: Orchestrator routes tasks to specialist agents (Calendar, Email, GitHub, Research) for domain expertise
- **Streaming Responses**: Slack-friendly streaming with readable partial updates instead of choppy token-level edits
- **Tool Calling**: LLM-driven tool selection with multi-step execution capabilities
- **Persistent Memory**: Conversation history and user preferences survive restarts
- **Proactive Alerts**: Calendar reminders, important email notifications, and daily briefings
- **Slack-Configurable Notifications**: Inspect and update proactive reminder/briefing/quiet-hours settings from Slack
- **Confirmation-Gated Actions**: Write actions use explicit Slack confirmation buttons with action-ID validation

### Security
- **Prompt Injection Protection**: Pattern-based detection and sanitization of malicious inputs
- **Rate Limiting**: Per-user request throttling to prevent abuse
- **Comprehensive Audit Logging**: All bot interactions logged to SQLite for security review
- **Input Sanitization**: Removal of suspicious unicode and content filtering

## Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                         Slack Bot                             │
│  (Socket Mode - DMs and @mentions)                            │
├───────────────────────────────────────────────────────────────┤
│                      Security Layer                           │
│  Rate Limiting │ Input Sanitization │ Audit Logging           │
├───────────────────────────────────────────────────────────────┤
│                      Bot Mode Router                          │
│  intent (legacy) │ agent (single) │ multi_agent (specialists) │
├───────────────────────────────────────────────────────────────┤
│                      Agent Executor                           │
│  Tool Calling │ Streaming │ Multi-step Execution              │
├──────────────────────┬────────────────────────────────────────┤
│    Orchestrator      │         Specialist Agents              │
│  (Task Planning)     │  Calendar │ Email │ GitHub │ Research  │
├──────────────────────┴────────────────────────────────────────┤
│                       Tools Layer                             │
│  search_emails │ send_email │ create_event │ add_comment │ .. │
├───────────────────────────────────────────────────────────────┤
│              Query Engine + Semantic Search                   │
├─────────────────────┬─────────────────────────────────────────┤
│  Knowledge Graph    │          Vector Store                   │
│  (SQLite)           │          (ChromaDB)                     │
├─────────────────────┴─────────────────────────────────────────┤
│                        Indexers                               │
│  Gmail │ Drive │ Calendar │ GitHub │ Slack │ Zotero │ Notion  │
├───────────────────────────────────────────────────────────────┤
│                      Integrations                             │
│  Google │ GitHub │ Slack │ Zotero │ Notion │ Todoist          │
└───────────────────────────────────────────────────────────────┘
```

## Requirements

- Python 3.11+
- Google Cloud Project with OAuth credentials
- GitHub Personal Access Token
- Slack App with Socket Mode enabled
- OpenAI API key (for embeddings)
- Anthropic API key (for intent classification)
- Zotero API key (optional, for paper management)

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/engram.git
   cd engram
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your credentials (see Configuration section)
   ```

## Configuration

### Google Cloud Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a new project
3. Enable APIs: Gmail, Google Drive, Google Calendar, Google Docs
4. Create OAuth 2.0 credentials (Desktop application)
5. Download and note the `client_id` and `client_secret`

**Required OAuth Scopes** (configured automatically):
- `gmail.readonly` - Read emails
- `gmail.send` - Send emails
- `drive.readonly` - Read Drive files
- `drive.file` - Comment on Docs
- `calendar` - Full calendar access (read/write)
- `documents` - Access Google Docs

### Slack App Setup

1. Go to [Slack API](https://api.slack.com/apps) and create a new app
2. Enable **Socket Mode** (Settings → Socket Mode)
3. Add **Bot Token Scopes** (OAuth & Permissions):
   - `app_mentions:read`
   - `chat:write`
   - `im:history`
   - `im:read`
   - `im:write`
   - `users:read`
4. **Subscribe to bot events** (Event Subscriptions):
   - `app_mention`
   - `message.im`
5. Enable **Messages Tab** (App Home → Show Tabs)
6. Install app to workspace
7. Copy `Bot User OAuth Token` and `App-Level Token`

### Environment Variables

Edit `.env` with your credentials:

```bash
# Google OAuth (from GCP Console)
GOOGLE_CLIENT_ID=your_client_id
GOOGLE_CLIENT_SECRET=your_client_secret

# GitHub Personal Access Token
# Create at: https://github.com/settings/tokens
# Required scopes: repo, read:org, read:user
GITHUB_TOKEN=ghp_xxxxx
GITHUB_USERNAME=your_username
GITHUB_ORG=your_org

# Slack Tokens
SLACK_BOT_TOKEN=xoxb-xxxxx
SLACK_APP_TOKEN=xapp-xxxxx
SLACK_WORKSPACE=your_workspace

# Notion API
# Create integration at: https://www.notion.so/my-integrations
# Share each target page/database with the integration
NOTION_API_KEY=secret_xxxxx
NOTION_WORKSPACE=default

# Authorized Slack User IDs (comma-separated)
# Find your ID: Click profile → More → Copy member ID
SLACK_AUTHORIZED_USERS=U12345678
# Set to true to allow all users if no allowlist is configured (default: false)
SLACK_ALLOW_ALL_USERS=false

# User timezone (IANA name, default: America/Los_Angeles)
USER_TIMEZONE=America/Los_Angeles

# OpenAI API Key (for embeddings)
OPENAI_API_KEY=sk-xxxxx

# Anthropic API Key (for intent classification and agents)
ANTHROPIC_API_KEY=sk-ant-xxxxx

# Todoist API Key (for tasks and overdue items in briefings)
TODOIST_API_KEY=xxxxx

# Zotero API (optional - for paper management)
# Get API key at: https://www.zotero.org/settings/keys
# User ID is shown on the same page
ZOTERO_API_KEY=xxxxx
ZOTERO_USER_ID=12345678
ZOTERO_LIBRARY_TYPE=user  # "user" for personal, "group" for shared library
ZOTERO_DEFAULT_COLLECTION=MyCollection  # Default collection for new papers

# Bot Mode: "intent" (legacy), "agent" (single agent), "multi_agent" (specialists)
BOT_MODE=agent

# Model profiles
# Keep these explicit so model upgrades can be smoke-tested before restart.
INTENT_MODEL=claude-haiku-4-5
AGENT_MODEL=claude-sonnet-5
BRIEFING_MODEL=claude-sonnet-5
DEEP_RESEARCH_MODEL=claude-opus-4-8
IDEASPARK_MODEL=claude-sonnet-5
HEAVY_AGENT_MODEL=claude-opus-4-8

# Enable streaming responses (applies to agent and multi_agent modes)
ENABLE_STREAMING=true

# Slack message edit interval for streaming responses.
# Slack redraws edited messages; 1.0-1.5s usually feels smoother than token-level updates.
STREAMING_UPDATE_INTERVAL=1.25

# Direct email send behavior
# false (default): draft-only (recommended)
# true: SendEmailTool enabled, but still requires explicit Slack confirmation button
ENABLE_DIRECT_EMAIL_SEND=false

# Security Settings
SECURITY_LEVEL=moderate          # strict, moderate, or permissive
RATE_LIMIT_REQUESTS=30           # Max requests per window
RATE_LIMIT_WINDOW=60             # Window in seconds
RATE_LIMIT_BLOCK_DURATION=300    # Block duration when limit exceeded

# Audit Logging
ENABLE_AUDIT_LOG=true
AUDIT_LOG_PATH=data/audit.db
AUDIT_RETENTION_DAYS=90
AUDIT_LOG_MESSAGES=false  # Store raw message text in audit logs
JOB_DB_PATH=data/jobs.db
JOB_RETENTION_DAYS=30
ENABLE_TRACE_LOG=true
TRACE_LOG_PATH=logs/traces.jsonl
```

### Notion Setup

1. Go to [Notion integrations](https://www.notion.so/my-integrations) and create an **Internal Integration**.
2. Copy the **Internal Integration Token** and set `NOTION_API_KEY` in `.env`.
3. In Notion, open each page/database you want indexed, click **Share**, and invite your integration.
4. Verify access:

```bash
python -c "from src.integrations.notion_client import NotionClient; print(NotionClient().test_connection())"
```

### Google Account Authentication

Run the OAuth setup for each Google account:

```bash
# Authenticate all accounts interactively
python scripts/google_auth_setup.py --all

# Or authenticate specific accounts
python scripts/google_auth_setup.py --account arc
python scripts/google_auth_setup.py --account personal
```

## Usage

### Initial Sync

Run the full sync to index all data (takes several hours depending on data volume):

```bash
python scripts/full_sync_pipeline.py
```

This will:
- Index Gmail messages from all accounts
- Index Google Drive files
- Index Calendar events
- Index GitHub repos, issues, and PRs
- Index Slack messages
- Index Zotero papers, notes, and collections
- Index Notion pages and Todoist tasks
- Generate embeddings for semantic search

### Daily Sync

Run incremental sync to catch up on new data:

```bash
python scripts/daily_delta_sync.py
```

### Daily Briefing

Get a summary of your day:

```bash
python scripts/daily_briefing.py
```

Output includes:
- Today's calendar events (merged from all accounts)
- Unread email counts per account
- Open GitHub PRs and assigned issues
- Todoist overdue tasks
- Available time slots

### Start the Slack Bot

```bash
python scripts/run_bot.py
```

The bot runs in Socket Mode (no public URL needed). Keep it running to respond to messages.

**Run in background:**
```bash
nohup python scripts/run_bot.py > logs/bot.log 2>&1 &
```

### Model and Eval Checks

```bash
# Smoke-test configured Anthropic model profiles
python scripts/model_smoke_test.py --all

# Validate public synthetic Slack workflow evals
python scripts/run_evals.py --cases evals/slack_workflows.example.jsonl --list

# Score saved eval predictions against text, routing, confirmation, and model expectations
python scripts/run_evals.py \
  --cases evals/slack_workflows.example.jsonl \
  --responses evals/responses.example.jsonl

# Generate and score dry-run routing/safety predictions without model/tool calls
python scripts/run_evals.py \
  --cases evals/slack_workflows.example.jsonl \
  --generate-predictions evals/generated.local.jsonl

# Summarize metadata-only runtime traces
python scripts/trace_summary.py

# Fail automation when trace thresholds are exceeded
python scripts/trace_summary.py --fail-on-warning
```

## Bot Modes

The bot supports three operating modes, configured via `BOT_MODE`:

### Intent Mode (`intent`)
Legacy mode using intent classification with hardcoded handlers.
- Fast and predictable
- Limited to predefined intents
- Best for simple, specific queries

### Agent Mode (`agent`) - Default
Single agent with LLM-driven tool calling.
- Dynamic tool selection
- Multi-step execution
- Slack-friendly streaming responses
- Natural conversation support

### Multi-Agent Mode (`multi_agent`)
Orchestrator routes to specialist agents.
- **Calendar Agent**: View events, answer next/upcoming questions using current local time, check availability, create/update/cancel events with attendee notifications
- **Email Agent**: Search, drafts, and optional send (feature-flagged)
- **GitHub Agent**: PRs, issues, repository activity
- **Research Agent**: Semantic search, briefings

Each specialist has domain expertise and relevant tools. Direct conversational turns use the router model profile, normal specialist tool work uses the agent profile, and multi-specialist synthesis uses the heavy profile.

## Slack Bot Commands

Talk to the bot via DM or @mention in channels:

| Query | Description |
|-------|-------------|
| `Hi` / `Hello` | Natural greeting - no tool search triggered |
| `What can you do?` | Help and capabilities overview |
| `What's on my calendar today?` | Show today's events from all accounts |
| `What's my next event?` | Show the next event using your configured local timezone |
| `What's my schedule for tomorrow?` | Show tomorrow's calendar |
| `When am I free this week?` | Find available time slots |
| `Search for emails about [topic]` | Semantic search across emails |
| `Send an email to [person] about [topic]` | Create draft by default, or send via explicit confirmation if enabled |
| `Create a meeting with [person] tomorrow at 2pm` | Create calendar events |
| `Move event [id] to tomorrow at 3pm` | Update calendar events after confirmation |
| `Cancel event [id]` | Cancel calendar events after confirmation |
| `Find documents about [topic]` | Search Google Drive files |
| `Comment on Google Doc [id]: [comment]` | Add Google Doc comments after confirmation |
| `Show my open PRs` | List your GitHub pull requests |
| `What issues are assigned to me?` | List assigned GitHub issues |
| `Search my papers for [topic]` | Search Zotero library |
| `What papers did I add recently?` | List recent Zotero additions |
| `Add this paper: [DOI or URL]` | Add paper to Zotero with metadata |
| `Find papers by [author]` | Search papers by author |
| `What did I miss yesterday?` | Daily briefing for a specific date |
| `Help` | Show available commands |
| `Set my briefing to 8am weekdays` | Update proactive notification settings |
| `jobs` | Show recent background jobs |
| `job status <id>` | Check a background job |
| `job cancel <id>` | Cancel a queued/running background job |
| `job retry <id>` | Retry a finished or failed background job |

### Example Interactions

```
You: Hi!
Bot: Hi! How can I help you today? I can check your calendar, search
     emails, look up GitHub activity, or just chat.

You: What can you do?
Bot: I'm your personal assistant with access to:
     • Calendar (6 Google accounts)
     • Email search and drafts (optional send with confirmation)
     • Google Drive documents
     • GitHub repos, PRs, and issues
     • Slack message history

     Just ask naturally - "what's on my calendar?" or "find emails about X"

You: What's on my calendar today?
Bot: 📅 Today's Events (Tuesday, Feb 3):
     • 9:00 AM - Team Standup (arc)
     • 11:00 AM - 1:1 with John (personal)
     • 2:00 PM - Project Review (tahoe)

You: Search for emails about quarterly report
Bot: 📧 Found 5 relevant emails:
     1. "Q4 Report Draft" from alice@company.com (Dec 15)
     2. "Re: Quarterly Numbers" from bob@company.com (Dec 18)
     ...

You: When am I free tomorrow?
Bot: 🟢 Available slots tomorrow:
     • 8:00 AM - 9:30 AM
     • 12:00 PM - 1:00 PM
     • 4:00 PM - 6:00 PM

You: Create a meeting with alice@company.com tomorrow at 2pm for 30 minutes
Bot: Please confirm creating this calendar event:
     Event: Meeting
     When: 2024-02-04 2:00 PM (30 min)
     Attendees: alice@company.com
```

## Project Structure

```
engram/
├── src/
│   ├── config.py                 # Configuration and environment
│   ├── knowledge_graph.py        # SQLite storage layer
│   │
│   ├── integrations/             # External service clients
│   │   ├── google_auth.py        # OAuth flow
│   │   ├── google_multi.py       # Multi-account manager
│   │   ├── gmail.py              # Gmail API client (read + send)
│   │   ├── gdrive.py             # Drive API client
│   │   ├── gdocs.py              # Docs API client (comments)
│   │   ├── gcalendar.py          # Calendar API client (read + write)
│   │   ├── github_client.py      # GitHub API client
│   │   ├── slack.py              # Slack API client
│   │   ├── zotero_client.py      # Zotero API client
│   │   ├── notion_client.py      # Notion API client
│   │   └── todoist_client.py     # Todoist API client
│   │
│   ├── indexers/                 # Data indexing pipelines
│   │   ├── gmail_indexer.py
│   │   ├── gdrive_indexer.py
│   │   ├── gcal_indexer.py
│   │   ├── github_indexer.py
│   │   ├── slack_indexer.py
│   │   ├── zotero_indexer.py     # Papers, notes, collections
│   │   ├── notion_indexer.py
│   │   └── todoist_indexer.py
│   │
│   ├── semantic/                 # Semantic search layer
│   │   ├── embedder.py           # OpenAI embeddings
│   │   ├── chunker.py            # Text chunking
│   │   ├── vector_store.py       # ChromaDB wrapper
│   │   └── semantic_indexer.py   # Embedding pipeline
│   │
│   ├── bot/                      # Slack bot
│   │   ├── app.py                # Main bot application
│   │   ├── event_handlers.py     # Message handlers with security
│   │   ├── intent_router.py      # LLM intent classification
│   │   ├── conversation.py       # Conversation state + persistence
│   │   ├── datetime_utils.py     # Shared date/time parsing helpers
│   │   ├── formatters.py         # Slack Block Kit formatting
│   │   ├── tools.py              # Tool definitions for LLM
│   │   ├── executor.py           # Agent executor with streaming
│   │   ├── user_memory.py        # Long-term user preferences
│   │   ├── heartbeat.py          # Proactive notifications
│   │   ├── security.py           # Input sanitization + rate limiting
│   │   ├── audit.py              # Comprehensive audit logging
│   │   ├── actions/              # Confirmable write actions
│   │   ├── handlers/             # Intent-specific handlers
│   │   │   ├── calendar.py
│   │   │   ├── email.py
│   │   │   ├── search.py
│   │   │   ├── github.py
│   │   │   ├── briefing.py
│   │   │   └── chat.py           # Natural conversation handler
│   │   └── agents/               # Multi-agent architecture
│   │       ├── base.py           # BaseAgent class
│   │       ├── orchestrator.py   # Task routing + synthesis
│   │       ├── calendar_agent.py # Calendar specialist
│   │       ├── email_agent.py    # Email specialist
│   │       ├── github_agent.py   # GitHub specialist
│   │       └── research_agent.py # Search/briefing specialist
│   │
│   ├── mcp/                      # Model Context Protocol
│   │   ├── server.py             # MCP server (expose tools)
│   │   └── client.py             # MCP client (external tools)
│   │
│   └── query/                    # Query engine
│       ├── engine.py             # Unified query interface
│       └── calendar_aggregator.py
│
├── scripts/                      # CLI tools
│   ├── google_auth_setup.py      # OAuth setup
│   ├── full_sync_pipeline.py     # Initial sync
│   ├── daily_delta_sync.py       # Incremental sync
│   ├── daily_briefing.py         # Daily summary
│   └── run_bot.py                # Start Slack bot
│
├── data/                         # Local databases (gitignored)
│   ├── knowledge_graph.db        # Main knowledge graph
│   ├── chroma/                   # Vector store
│   ├── conversations.db          # Persistent conversations
│   ├── user_memory.db            # User preferences
│   └── audit.db                  # Security audit log
├── logs/                         # Log files (gitignored)
├── credentials/                  # OAuth tokens (gitignored)
└── tests/                        # Test suite (361 passing, 6 skipped)
```

## Automation (macOS)

Set up launchd jobs for automatic syncing:

**Daily sync at 6 AM:**
```bash
cp docs/com.engram.daily.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.engram.daily.plist
```

**Keep bot running:**
```bash
cp docs/com.engram.bot.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.engram.bot.plist
```

**Watchdog health check every 5 minutes:**
```bash
cp docs/com.engram.bot-healthcheck.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.engram.bot-healthcheck.plist
```

The watchdog runs `scripts/bot_healthcheck.py`, verifies `com.engram.bot` is loaded with a running `scripts/run_bot.py` process, and bootstraps/kickstarts the service if needed. Logs are written to `logs/bot_healthcheck.log`.

## Security

### Access Control
- **Credentials**: All tokens and OAuth credentials are stored locally in `credentials/` (gitignored)
- **Data**: All indexed data stays local in `data/` (gitignored)
- **Bot Access**: Only Slack users listed in `SLACK_AUTHORIZED_USERS` can interact with the bot
- **Email Sending**: Draft-only by default. Set `ENABLE_DIRECT_EMAIL_SEND=true` to enable send, which still requires explicit confirmation.
- **Calendar Events**: The bot answers "next/upcoming" queries using `USER_TIMEZONE` and excludes events that already ended today. Creating, updating, and cancelling events require confirmation.
- **Doc Comments**: The bot can add, reply to, and resolve comments on Google Docs you have access to
- **Other Write Actions**: Email drafts, GitHub issues, Todoist task changes, Notion writes, and Zotero additions require explicit confirmation
- **Action Integrity**: Confirmation clicks are validated by action ID and thread-aware context lookup to prevent stale/mismatched execution
- **Confirmation Timeout**: Pending confirmations expire after 5 minutes and must be re-requested

### Prompt Injection Protection
The bot includes pattern-based detection for common injection attempts:
- System prompt manipulation ("ignore previous instructions")
- Delimiter injection (```system, <system>, [SYSTEM])
- Jailbreak attempts ("DAN mode", "developer mode")
- Output manipulation ("reveal your prompt")

Security levels (`SECURITY_LEVEL`):
- `strict`: Block suspicious content entirely
- `moderate`: Warn and filter suspicious content (default)
- `permissive`: Log only, allow all content

### Rate Limiting
Per-user request throttling prevents abuse:
- Default: 30 requests per 60-second window
- Exceeded: 5-minute block
- Configurable via environment variables

### Audit Logging
All bot interactions are logged to `data/audit.db`:
- Messages received (user, channel, content hash)
- Tool executions (name, input, result, duration)
- Security events (threats detected, actions blocked)
- Agent activity (routing decisions, synthesis)

Query audit logs:
```bash
python -c "from src.bot.audit import get_audit_logger; print(get_audit_logger().query(limit=10))"
```

Logs are retained for 90 days by default (`AUDIT_RETENTION_DAYS`).

## MCP Integration

The bot exposes its capabilities via the [Model Context Protocol](https://modelcontextprotocol.io/), allowing external tools to query your knowledge graph.

**Start MCP server:**
```bash
python -m src.mcp.server
```

**Available MCP tools:**
- `search_emails` - Search emails across accounts
- `get_calendar_events` - Get calendar events for a date
- `check_availability` - Find available time slots
- `search_documents` - Search Google Drive files
- `get_github_activity` - Get GitHub PRs and issues

**Connect from Claude Desktop:**
Add to `~/.claude/mcp.json`:
```json
{
  "servers": {
    "engram": {
      "command": "python",
      "args": ["-m", "src.mcp.server"],
      "cwd": "/path/to/engram"
    }
  }
}
```

## Development

**Run tests:**
```bash
python -m pytest tests/ -v

# Run specific test modules
pytest tests/test_agents.py -v      # Multi-agent tests
pytest tests/test_security.py -v    # Security tests
pytest tests/test_executor.py -v    # Executor + streaming tests

# Run with coverage
pytest --cov=src tests/
```

**Check logs:**
```bash
tail -f logs/engram.log
```

**Query the knowledge graph directly:**
```bash
python scripts/query_knowledge.py "search term"
```

**Debug bot modes:**
```bash
# Test intent classification
python -c "from src.bot.intent_router import IntentRouter; r = IntentRouter(); print(r.route('hi'))"

# Test agent execution
python -c "from src.bot.executor import AgentExecutor; e = AgentExecutor(); print(e.run('what is on my calendar'))"
```

## Cost Estimate

| Service | Initial Sync | Monthly |
|---------|--------------|---------|
| OpenAI Embeddings | $15-30 | $15-25 |
| Anthropic (Haiku - intent) | - | $5-10 |
| Anthropic (Sonnet - agent) | - | $20-50 |
| Google/GitHub/Slack APIs | Free | Free |
| **Total** | **$15-30** | **$40-85** |

*Agent mode costs vary based on usage. Multi-agent mode may use more tokens due to specialist prompts.*

## License

Private repository - not for distribution.

## Acknowledgments

Built with Claude Code (Anthropic).

Inspired by modern agent architectures:
- [OpenClaw](https://openclaw.ai/) - Multi-agent patterns
- [LangGraph](https://www.langchain.com/langgraph) - Agent orchestration
- [Model Context Protocol](https://modelcontextprotocol.io/) - Tool integration standard
