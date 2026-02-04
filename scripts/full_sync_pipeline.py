#!/usr/bin/env python3
"""Full sync pipeline for initial data ingestion."""

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


def sync_google(kg: KnowledgeGraph, accounts: list[str] | None = None) -> dict:
    """Sync all Google services for specified accounts.

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
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Syncing Google account: {account}")
        logger.info(f"{'=' * 60}")

        # Gmail
        logger.info(f"Indexing Gmail for {account}...")
        try:
            stats["gmail"][account] = gmail_indexer.index_all(account)
        except Exception as e:
            logger.error(f"Gmail error for {account}: {e}")
            stats["gmail"][account] = {"error": str(e)}

        # Drive
        logger.info(f"Indexing Drive for {account}...")
        try:
            stats["drive"][account] = drive_indexer.index_all(account)
        except Exception as e:
            logger.error(f"Drive error for {account}: {e}")
            stats["drive"][account] = {"error": str(e)}

        # Calendar
        logger.info(f"Indexing Calendar for {account}...")
        try:
            stats["calendar"][account] = calendar_indexer.index_all(account)
        except Exception as e:
            logger.error(f"Calendar error for {account}: {e}")
            stats["calendar"][account] = {"error": str(e)}

    return stats


def sync_github(kg: KnowledgeGraph) -> dict:
    """Sync GitHub data.

    Args:
        kg: Knowledge graph instance.

    Returns:
        Statistics dictionary.
    """
    logger.info(f"\n{'=' * 60}")
    logger.info("Syncing GitHub")
    logger.info(f"{'=' * 60}")

    indexer = GitHubIndexer(kg)
    try:
        return indexer.index_all()
    except Exception as e:
        logger.error(f"GitHub error: {e}")
        return {"error": str(e)}


def sync_slack(kg: KnowledgeGraph) -> dict:
    """Sync Slack data.

    Args:
        kg: Knowledge graph instance.

    Returns:
        Statistics dictionary.
    """
    logger.info(f"\n{'=' * 60}")
    logger.info("Syncing Slack")
    logger.info(f"{'=' * 60}")

    indexer = SlackIndexer(kg)
    try:
        return indexer.index_all()
    except Exception as e:
        logger.error(f"Slack error: {e}")
        return {"error": str(e)}


def sync_notion(kg: KnowledgeGraph) -> dict:
    """Sync Notion data.

    Args:
        kg: Knowledge graph instance.

    Returns:
        Statistics dictionary.
    """
    logger.info(f"\n{'=' * 60}")
    logger.info("Syncing Notion")
    logger.info(f"{'=' * 60}")

    indexer = NotionIndexer(kg)
    try:
        return indexer.index_all()
    except Exception as e:
        logger.error(f"Notion error: {e}")
        return {"error": str(e)}


def sync_todoist(kg: KnowledgeGraph) -> dict:
    """Sync Todoist data.

    Args:
        kg: Knowledge graph instance.

    Returns:
        Statistics dictionary.
    """
    logger.info(f"\n{'=' * 60}")
    logger.info("Syncing Todoist")
    logger.info(f"{'=' * 60}")

    indexer = TodoistIndexer(kg)
    try:
        return indexer.index_all()
    except Exception as e:
        logger.error(f"Todoist error: {e}")
        return {"error": str(e)}


def build_semantic_index(kg: KnowledgeGraph) -> dict:
    """Build semantic embeddings index.

    Args:
        kg: Knowledge graph instance.

    Returns:
        Statistics dictionary.
    """
    logger.info(f"\n{'=' * 60}")
    logger.info("Building Semantic Index")
    logger.info(f"{'=' * 60}")

    indexer = SemanticIndexer(kg)
    try:
        return indexer.index_all(show_progress=True)
    except Exception as e:
        logger.error(f"Semantic indexing error: {e}")
        return {"error": str(e)}


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Full sync pipeline for Hani Replica"
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
        help="Skip semantic indexing",
    )
    parser.add_argument(
        "--google-accounts",
        type=str,
        nargs="+",
        choices=GOOGLE_ACCOUNTS,
        help="Specific Google accounts to sync",
    )

    args = parser.parse_args()

    start_time = time.time()
    logger.info("Starting full sync pipeline...")

    # Initialize knowledge graph
    kg = KnowledgeGraph()

    all_stats = {}

    # Sync Google
    if not args.skip_google:
        accounts = args.google_accounts or GOOGLE_ACCOUNTS
        all_stats["google"] = sync_google(kg, accounts)
    else:
        logger.info("Skipping Google sync")

    # Sync GitHub
    if not args.skip_github:
        all_stats["github"] = sync_github(kg)
    else:
        logger.info("Skipping GitHub sync")

    # Sync Slack
    if not args.skip_slack:
        all_stats["slack"] = sync_slack(kg)
    else:
        logger.info("Skipping Slack sync")

    # Sync Notion
    if not args.skip_notion:
        all_stats["notion"] = sync_notion(kg)
    else:
        logger.info("Skipping Notion sync")

    # Sync Todoist
    if not args.skip_todoist:
        all_stats["todoist"] = sync_todoist(kg)
    else:
        logger.info("Skipping Todoist sync")

    # Build semantic index
    if not args.skip_semantic:
        all_stats["semantic"] = build_semantic_index(kg)
    else:
        logger.info("Skipping semantic indexing")

    # Print summary
    elapsed = time.time() - start_time
    logger.info(f"\n{'=' * 60}")
    logger.info("SYNC COMPLETE")
    logger.info(f"{'=' * 60}")
    logger.info(f"Total time: {elapsed / 60:.1f} minutes")
    logger.info(f"\nKnowledge Graph Stats:")

    kg_stats = kg.get_stats()
    logger.info(f"  Total entities: {kg_stats.get('total_entities', 0)}")
    logger.info(f"  Total content: {kg_stats.get('total_content', 0)}")
    logger.info(f"  Total relationships: {kg_stats.get('total_relationships', 0)}")

    if "semantic" in all_stats and "error" not in all_stats["semantic"]:
        logger.info(f"\nSemantic Index Stats:")
        logger.info(f"  Chunks created: {all_stats['semantic'].get('chunks_created', 0)}")
        logger.info(f"  Embeddings: {all_stats['semantic'].get('embeddings_generated', 0)}")


if __name__ == "__main__":
    main()
