#!/usr/bin/env python3
"""Daily incremental sync for updates since last sync."""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import GOOGLE_ACCOUNTS, LOG_FILE, LOG_LEVEL, ensure_directories
from src.indexers.gcal_indexer import CalendarIndexer
from src.indexers.gdrive_indexer import DriveIndexer
from src.indexers.github_indexer import GitHubIndexer
from src.indexers.gmail_indexer import GmailIndexer
from src.indexers.notion_indexer import NotionIndexer
from src.indexers.slack_indexer import SlackIndexer
from src.indexers.todoist_indexer import TodoistIndexer
from src.knowledge_graph import KnowledgeGraph
from src.semantic.semantic_indexer import SemanticIndexer

# Configure logging
ensure_directories()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE),
    ],
)
logger = logging.getLogger(__name__)


def delta_sync_google(kg: KnowledgeGraph, accounts: list[str] | None = None) -> dict:
    """Delta sync all Google services.

    Args:
        kg: Knowledge graph instance.
        accounts: Accounts to sync. None for all.

    Returns:
        Statistics dictionary.
    """
    accounts = accounts or GOOGLE_ACCOUNTS
    stats = {"gmail": {}, "drive": {}, "calendar": {}}

    gmail_indexer = GmailIndexer(kg)
    drive_indexer = DriveIndexer(kg)
    calendar_indexer = CalendarIndexer(kg)

    for account in accounts:
        logger.info(f"Delta sync for Google account: {account}")

        # Gmail
        try:
            stats["gmail"][account] = gmail_indexer.index_delta(account)
        except Exception as e:
            logger.error(f"Gmail delta error for {account}: {e}")
            stats["gmail"][account] = {"error": str(e)}

        # Drive
        try:
            stats["drive"][account] = drive_indexer.index_delta(account)
        except Exception as e:
            logger.error(f"Drive delta error for {account}: {e}")
            stats["drive"][account] = {"error": str(e)}

        # Calendar
        try:
            stats["calendar"][account] = calendar_indexer.index_delta(account)
        except Exception as e:
            logger.error(f"Calendar delta error for {account}: {e}")
            stats["calendar"][account] = {"error": str(e)}

    return stats


def delta_sync_github(kg: KnowledgeGraph) -> dict:
    """Delta sync GitHub data.

    Args:
        kg: Knowledge graph instance.

    Returns:
        Statistics dictionary.
    """
    logger.info("Delta sync for GitHub")
    indexer = GitHubIndexer(kg)
    try:
        return indexer.index_delta(days_back=1)
    except Exception as e:
        logger.error(f"GitHub delta error: {e}")
        return {"error": str(e)}


def delta_sync_slack(kg: KnowledgeGraph) -> dict:
    """Delta sync Slack data.

    Args:
        kg: Knowledge graph instance.

    Returns:
        Statistics dictionary.
    """
    logger.info("Delta sync for Slack")
    indexer = SlackIndexer(kg)
    try:
        return indexer.index_delta(days_back=1)
    except Exception as e:
        logger.error(f"Slack delta error: {e}")
        return {"error": str(e)}


def delta_sync_notion(kg: KnowledgeGraph) -> dict:
    """Delta sync Notion data.

    Args:
        kg: Knowledge graph instance.

    Returns:
        Statistics dictionary.
    """
    logger.info("Delta sync for Notion")
    indexer = NotionIndexer(kg)
    try:
        return indexer.index_delta(hours_back=24)
    except Exception as e:
        logger.error(f"Notion delta error: {e}")
        return {"error": str(e)}


def delta_sync_todoist(kg: KnowledgeGraph) -> dict:
    """Delta sync Todoist data.

    Args:
        kg: Knowledge graph instance.

    Returns:
        Statistics dictionary.
    """
    logger.info("Delta sync for Todoist")
    indexer = TodoistIndexer(kg)
    try:
        return indexer.index_delta()
    except Exception as e:
        logger.error(f"Todoist delta error: {e}")
        return {"error": str(e)}


def update_semantic_index(kg: KnowledgeGraph) -> dict:
    """Update semantic index with new content.

    Args:
        kg: Knowledge graph instance.

    Returns:
        Statistics dictionary.
    """
    logger.info("Updating semantic index")
    indexer = SemanticIndexer(kg)
    try:
        # Re-index everything (incremental would be more efficient but complex)
        # For now, just run on all content - embedder cache prevents re-embedding
        return indexer.index_all(show_progress=False)
    except Exception as e:
        logger.error(f"Semantic update error: {e}")
        return {"error": str(e)}


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Daily delta sync for Hani Replica"
    )
    parser.add_argument(
        "--skip-google",
        action="store_true",
        help="Skip Google services sync",
    )
    parser.add_argument(
        "--skip-github",
        action="store_true",
        help="Skip GitHub sync",
    )
    parser.add_argument(
        "--skip-slack",
        action="store_true",
        help="Skip Slack sync",
    )
    parser.add_argument(
        "--skip-notion",
        action="store_true",
        help="Skip Notion sync",
    )
    parser.add_argument(
        "--skip-todoist",
        action="store_true",
        help="Skip Todoist sync",
    )
    parser.add_argument(
        "--skip-semantic",
        action="store_true",
        help="Skip semantic index update",
    )

    args = parser.parse_args()

    start_time = time.time()
    logger.info("Starting daily delta sync...")

    kg = KnowledgeGraph()
    all_stats = {}

    if not args.skip_google:
        all_stats["google"] = delta_sync_google(kg)

    if not args.skip_github:
        all_stats["github"] = delta_sync_github(kg)

    if not args.skip_slack:
        all_stats["slack"] = delta_sync_slack(kg)

    if not args.skip_notion:
        all_stats["notion"] = delta_sync_notion(kg)

    if not args.skip_todoist:
        all_stats["todoist"] = delta_sync_todoist(kg)

    if not args.skip_semantic:
        all_stats["semantic"] = update_semantic_index(kg)

    elapsed = time.time() - start_time
    logger.info(f"Delta sync complete in {elapsed:.1f} seconds")

    # Print summary
    for service, stats in all_stats.items():
        if isinstance(stats, dict) and "error" not in stats:
            logger.info(f"{service}: {stats}")


if __name__ == "__main__":
    main()
