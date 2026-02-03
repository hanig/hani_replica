# Hani Replica

A personal knowledge graph system that aggregates data from multiple Google accounts, GitHub, and Slack, with semantic search and an interactive Slack bot interface.

## Features

- **Multi-Account Google Integration**: Sync Gmail, Google Drive, and Google Calendar from up to 6 accounts with tiered search (primary accounts searched first)
- **Knowledge Graph**: SQLite-based storage of entities (people, repos, files) and content with relationship tracking
- **Semantic Search**: OpenAI embeddings (text-embedding-3-large) with ChromaDB vector store for intelligent content retrieval
- **Slack Bot**: Interactive assistant using Socket Mode (no public URL required) with LLM-powered intent classification
- **Daily Briefings**: Aggregated calendar, email counts, and GitHub activity summaries

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Slack Bot                                │
│  (Socket Mode - DMs and @mentions)                              │
├─────────────────────────────────────────────────────────────────┤
│                     Intent Router                                │
│  (Claude Haiku for classification)                              │
├─────────────────────────────────────────────────────────────────┤
│                      Handlers                                    │
│  Calendar │ Email │ Search │ GitHub │ Briefing                  │
├─────────────────────────────────────────────────────────────────┤
│              Query Engine + Semantic Search                      │
├──────────────────────┬──────────────────────────────────────────┤
│   Knowledge Graph    │           Vector Store                    │
│   (SQLite)           │           (ChromaDB)                      │
├──────────────────────┴──────────────────────────────────────────┤
│                       Indexers                                   │
│  Gmail │ Drive │ Calendar │ GitHub │ Slack                      │
├─────────────────────────────────────────────────────────────────┤
│                    Integrations                                  │
│  Google (OAuth) │ GitHub (PAT) │ Slack (Bot Token)              │
└─────────────────────────────────────────────────────────────────┘
```

## Requirements

- Python 3.11+
- Google Cloud Project with OAuth credentials
- GitHub Personal Access Token
- Slack App with Socket Mode enabled
- OpenAI API key (for embeddings)
- Anthropic API key (for intent classification)

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/hanig/hani_replica.git
   cd hani_replica
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
3. Enable APIs: Gmail, Google Drive, Google Calendar
4. Create OAuth 2.0 credentials (Desktop application)
5. Download and note the `client_id` and `client_secret`

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

# OpenAI API Key (for embeddings)
OPENAI_API_KEY=sk-xxxxx

# Anthropic API Key (for intent classification)
ANTHROPIC_API_KEY=sk-ant-xxxxx
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

## Slack Bot Commands

Talk to the bot via DM or @mention in channels:

| Query | Description |
|-------|-------------|
| `What's on my calendar today?` | Show today's events from all accounts |
| `What's my schedule for tomorrow?` | Show tomorrow's calendar |
| `When am I free this week?` | Find available time slots |
| `Search for emails about [topic]` | Semantic search across emails |
| `Find documents about [topic]` | Search Google Drive files |
| `Show my open PRs` | List your GitHub pull requests |
| `What issues are assigned to me?` | List assigned GitHub issues |
| `What did I miss yesterday?` | Daily briefing for a specific date |
| `Help` | Show available commands |

### Example Interactions

```
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
```

## Project Structure

```
hani_replica/
├── src/
│   ├── config.py                 # Configuration and environment
│   ├── knowledge_graph.py        # SQLite storage layer
│   │
│   ├── integrations/             # External service clients
│   │   ├── google_auth.py        # OAuth flow
│   │   ├── google_multi.py       # Multi-account manager
│   │   ├── gmail.py              # Gmail API client
│   │   ├── gdrive.py             # Drive API client
│   │   ├── gcalendar.py          # Calendar API client
│   │   ├── github_client.py      # GitHub API client
│   │   └── slack.py              # Slack API client
│   │
│   ├── indexers/                 # Data indexing pipelines
│   │   ├── gmail_indexer.py
│   │   ├── gdrive_indexer.py
│   │   ├── gcal_indexer.py
│   │   ├── github_indexer.py
│   │   └── slack_indexer.py
│   │
│   ├── semantic/                 # Semantic search layer
│   │   ├── embedder.py           # OpenAI embeddings
│   │   ├── chunker.py            # Text chunking
│   │   ├── vector_store.py       # ChromaDB wrapper
│   │   └── semantic_indexer.py   # Embedding pipeline
│   │
│   ├── bot/                      # Slack bot
│   │   ├── app.py                # Main bot application
│   │   ├── event_handlers.py     # Message handlers
│   │   ├── intent_router.py      # LLM intent classification
│   │   ├── conversation.py       # Conversation state
│   │   ├── formatters.py         # Slack Block Kit formatting
│   │   └── handlers/             # Intent-specific handlers
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
├── logs/                         # Log files (gitignored)
├── credentials/                  # OAuth tokens (gitignored)
└── tests/                        # Test suite
```

## Automation (macOS)

Set up launchd jobs for automatic syncing:

**Daily sync at 6 AM:**
```bash
cp docs/com.hani.replica.daily.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.hani.replica.daily.plist
```

**Keep bot running:**
```bash
cp docs/com.hani.replica.bot.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.hani.replica.bot.plist
```

## Security Notes

- **Credentials**: All tokens and OAuth credentials are stored locally in `credentials/` (gitignored)
- **Data**: All indexed data stays local in `data/` (gitignored)
- **Bot Access**: Only Slack users listed in `SLACK_AUTHORIZED_USERS` can interact with the bot
- **Email Drafts**: The bot can create email drafts but never sends emails automatically
- **GitHub Actions**: Issue creation requires explicit confirmation

## Development

**Run tests:**
```bash
python -m pytest tests/ -v
```

**Check logs:**
```bash
tail -f logs/hani_replica.log
```

**Query the knowledge graph directly:**
```bash
python scripts/query_knowledge.py "search term"
```

## Cost Estimate

| Service | Initial Sync | Monthly |
|---------|--------------|---------|
| OpenAI Embeddings | $15-30 | $15-25 |
| Anthropic (Haiku) | - | $5-10 |
| Google/GitHub/Slack APIs | Free | Free |
| **Total** | **$15-30** | **$20-35** |

## License

Private repository - not for distribution.

## Acknowledgments

Built with Claude Code (Anthropic).
