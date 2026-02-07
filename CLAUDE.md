# Hani Replica - Claude Code Instructions

Personal knowledge graph system aggregating data from 6 Google accounts, GitHub, and Slack with semantic search and Slack bot interface.

## Project Structure

```
hani_replica/
├── src/
│   ├── config.py                 # Environment + account configuration
│   ├── knowledge_graph.py        # SQLite-based storage
│   ├── integrations/             # External service clients
│   │   ├── google_auth.py        # OAuth flow for all accounts
│   │   ├── google_multi.py       # Multi-account manager with tiered search
│   │   ├── gmail.py, gdrive.py, gcalendar.py
│   │   ├── github_client.py
│   │   └── slack.py
│   ├── indexers/                 # Content indexers
│   │   ├── gmail_indexer.py, gdrive_indexer.py, gcal_indexer.py
│   │   ├── github_indexer.py, slack_indexer.py
│   ├── semantic/                 # Embedding and vector search
│   │   ├── embedder.py           # OpenAI text-embedding-3-large
│   │   ├── chunker.py            # Text chunking
│   │   ├── vector_store.py       # ChromaDB wrapper
│   │   └── semantic_indexer.py
│   ├── bot/                      # Slack bot
│   │   ├── app.py                # Main bot (Socket Mode)
│   │   ├── intent_router.py      # LLM intent classification
│   │   ├── handlers/             # Intent handlers
│   │   └── actions/              # Confirmable actions
│   └── query/                    # Query engine
├── scripts/                      # Orchestration scripts
├── tests/                        # Pytest tests
└── credentials/                  # OAuth tokens (gitignored)
```

## Key Configuration

### Google Accounts
Configured via environment variables in `.env`:
- `GOOGLE_ACCOUNTS`: comma-separated list of account names
- `GOOGLE_TIER1`: primary accounts (searched first)
- `GOOGLE_TIER2`: secondary accounts (searched only if no tier-1 results)
- `GOOGLE_EMAILS`: JSON dict mapping account names to email addresses

### Environment Variables
All secrets in `.env`:
- `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` - OAuth credentials
- `GITHUB_TOKEN` - Personal access token
- `SLACK_BOT_TOKEN`, `SLACK_APP_TOKEN` - For Socket Mode
- `OPENAI_API_KEY` - For embeddings
- `ANTHROPIC_API_KEY` - For intent classification
- `SLACK_AUTHORIZED_USERS` - Comma-separated user IDs

## Common Commands

```bash
# Setup Google OAuth
python scripts/google_auth_setup.py --all

# Full initial sync (takes hours)
python scripts/full_sync_pipeline.py

# Daily incremental sync
python scripts/daily_delta_sync.py

# Generate briefing
python scripts/daily_briefing.py

# Query knowledge graph
python scripts/query_knowledge.py search "your query"
python scripts/query_knowledge.py stats

# Run Slack bot
python scripts/run_bot.py

# Run tests
pytest tests/
```

## Architecture Notes

### Knowledge Graph
SQLite database with tables:
- `entities` - People, repos, channels
- `content` - Emails, files, events, messages
- `relationships` - Links between entities
- `sync_state` - Track incremental sync progress

### Semantic Search
- OpenAI `text-embedding-3-large` for embeddings
- ChromaDB for vector storage
- Chunks stored by content type (email, file, event, etc.)
- Embedding cache prevents re-embedding unchanged content

### Slack Bot
- Socket Mode (no public URL needed)
- Claude Haiku for intent classification
- Multi-turn conversations with 30-min TTL
- Actions that modify data require confirmation

### Security
- OAuth tokens stored locally in `credentials/`
- Slack bot restricted to `SLACK_AUTHORIZED_USERS`
- Email drafts are created but NEVER sent automatically
- All bot interactions are logged

## Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_knowledge_graph.py

# Run with coverage
pytest --cov=src
```

## Debugging

### Check OAuth status
```bash
python scripts/google_auth_setup.py --status
```

### Validate configuration
```bash
python scripts/run_bot.py --validate-only
```

### Check sync state
```bash
python scripts/query_knowledge.py stats
```

### Bot debug mode
```bash
python scripts/run_bot.py --debug
```

## Adding New Features

### New Intent
1. Add intent name to `INTENT_DEFINITIONS` in `intent_router.py`
2. Create handler in `src/bot/handlers/`
3. Register in `_route_message` in `event_handlers.py`

### New Data Source
1. Create client in `src/integrations/`
2. Create indexer in `src/indexers/`
3. Add to sync pipeline scripts

### New Action
1. Create action class extending `PendingAction` in `src/bot/actions/`
2. Implement `is_ready()`, `get_next_prompt()`, `update_from_input()`, `execute()`
