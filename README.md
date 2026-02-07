# Engram

A personal knowledge graph system that aggregates data from multiple Google accounts, GitHub, Slack, Notion, Todoist, and Zotero, with semantic search and an interactive Slack bot interface powered by intelligent agents.

> Disclaimer: This project is built by a scientist, not a security specialist. Review, harden, and use it at your own discretion, especially before handling sensitive data or deploying in production environments.

## Public Deployment Warning

This project is designed first for personal/local use. Do not expose it publicly or use it in multi-user/production environments without a full security review, least-privilege credentials, network hardening, and strict access controls.

## Features

### Core Capabilities
- **Multi-Account Google Integration**: Sync Gmail, Google Drive, and Google Calendar from up to 6 accounts with tiered search (primary accounts searched first)
- **Google Write Capabilities**: Create email drafts, create/modify calendar events, and comment on Google Docs
- **Zotero Integration**: Search papers, add references by DOI/URL with automatic metadata extraction (CrossRef + page scraping)
- **Notion & Todoist**: Search pages, manage tasks, create content
- **Knowledge Graph**: SQLite-based storage of entities (people, repos, files) and content with relationship tracking
- **Semantic Search**: OpenAI embeddings (text-embedding-3-large) with ChromaDB vector store for intelligent content retrieval
- **Slack Bot**: Interactive assistant using Socket Mode (no public URL required)
- **Daily Briefings**: Aggregated calendar, email counts, GitHub activity, and Todoist overdue tasks

### Advanced Agent Features
- **Natural Conversation**: Chat naturally without triggering tool searches - greetings, questions about the bot, and general conversation are handled intelligently
- **Multi-Agent Architecture**: Orchestrator routes tasks to specialist agents (Calendar, Email, GitHub, Research) for domain expertise
- **Streaming Responses**: Real-time token-by-token response streaming for better UX
- **Tool Calling**: LLM-driven tool selection with multi-step execution capabilities
- **Persistent Memory**: Conversation history and user preferences survive restarts
- **Proactive Alerts**: Calendar reminders, important email notifications, and daily briefings
- **Confirmation-Gated Actions**: Sensitive actions use explicit Slack confirmation buttons with action-ID validation

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

# Agent model for tool calling (default: claude-sonnet-4-20250514)
AGENT_MODEL=claude-sonnet-4-20250514

# Enable streaming responses (applies to agent and multi_agent modes)
ENABLE_STREAMING=true

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
- Streaming responses
- Natural conversation support

### Multi-Agent Mode (`multi_agent`)
Orchestrator routes to specialist agents.
- **Calendar Agent**: View events, check availability, create events with attendee invites
- **Email Agent**: Search, drafts, and optional send (feature-flagged)
- **GitHub Agent**: PRs, issues, repository activity
- **Research Agent**: Semantic search, briefings

Each specialist has domain expertise and relevant tools.

## Slack Bot Commands

Talk to the bot via DM or @mention in channels:

| Query | Description |
|-------|-------------|
| `Hi` / `Hello` | Natural greeting - no tool search triggered |
| `What can you do?` | Help and capabilities overview |
| `What's on my calendar today?` | Show today's events from all accounts |
| `What's my schedule for tomorrow?` | Show tomorrow's calendar |
| `When am I free this week?` | Find available time slots |
| `Search for emails about [topic]` | Semantic search across emails |
| `Send an email to [person] about [topic]` | Create draft by default, or send via explicit confirmation if enabled |
| `Create a meeting with [person] tomorrow at 2pm` | Create calendar events |
| `Find documents about [topic]` | Search Google Drive files |
| `Show my open PRs` | List your GitHub pull requests |
| `What issues are assigned to me?` | List assigned GitHub issues |
| `Search my papers for [topic]` | Search Zotero library |
| `What papers did I add recently?` | List recent Zotero additions |
| `Add this paper: [DOI or URL]` | Add paper to Zotero with metadata |
| `Find papers by [author]` | Search papers by author |
| `What did I miss yesterday?` | Daily briefing for a specific date |
| `Help` | Show available commands |

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
Bot: Created event "Meeting" on 2024-02-04 at 2:00 PM.
     Calendar invite sent to alice@company.com.
     https://calendar.google.com/event?eid=xxx
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
│   │   ├── formatters.py         # Slack Block Kit formatting
│   │   ├── tools.py              # Tool definitions for LLM
│   │   ├── executor.py           # Agent executor with streaming
│   │   ├── user_memory.py        # Long-term user preferences
│   │   ├── heartbeat.py          # Proactive notifications
│   │   ├── security.py           # Input sanitization + rate limiting
│   │   ├── audit.py              # Comprehensive audit logging
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
└── tests/                        # Test suite (360 tests)
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

## Security

### Access Control
- **Credentials**: All tokens and OAuth credentials are stored locally in `credentials/` (gitignored)
- **Data**: All indexed data stays local in `data/` (gitignored)
- **Bot Access**: Only Slack users listed in `SLACK_AUTHORIZED_USERS` can interact with the bot
- **Email Sending**: Draft-only by default. Set `ENABLE_DIRECT_EMAIL_SEND=true` to enable send, which still requires explicit confirmation.
- **Calendar Events**: The bot can create/modify events but confirms before making changes
- **Doc Comments**: The bot can add comments to Google Docs you have access to
- **GitHub Actions**: Issue creation requires explicit confirmation
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
