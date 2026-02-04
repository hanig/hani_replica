"""Todoist content indexer."""

import logging
from datetime import datetime, timezone
from typing import Any

from ..integrations.todoist_client import TodoistClient
from ..knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)


class TodoistIndexer:
    """Indexer for Todoist tasks and projects."""

    def __init__(self, kg: KnowledgeGraph | None = None):
        """Initialize the Todoist indexer.

        Args:
            kg: Knowledge graph instance. Creates new one if not provided.
        """
        self.kg = kg or KnowledgeGraph()

    def index_all(
        self,
        include_projects: bool = True,
        include_tasks: bool = True,
        include_labels: bool = True,
    ) -> dict[str, Any]:
        """Index all Todoist content.

        Args:
            include_projects: Index projects.
            include_tasks: Index active tasks.
            include_labels: Index labels.

        Returns:
            Statistics about the indexing operation.
        """
        logger.info("Starting full Todoist index")

        try:
            client = TodoistClient()
        except ValueError as e:
            logger.error(f"Cannot initialize Todoist client: {e}")
            return {"error": str(e)}

        stats = {
            "projects_indexed": 0,
            "tasks_indexed": 0,
            "labels_indexed": 0,
            "errors": 0,
        }

        # Build project lookup for task context
        project_map = {}

        try:
            # Index projects
            if include_projects:
                projects = client.list_projects()
                for project in projects:
                    project_map[project["id"]] = project["name"]
                    self._index_project(project, stats)
                logger.info(f"Indexed {stats['projects_indexed']} projects")

            # Index labels
            if include_labels:
                labels = client.list_labels()
                for label in labels:
                    self._index_label(label, stats)
                logger.info(f"Indexed {stats['labels_indexed']} labels")

            # Index active tasks
            if include_tasks:
                tasks = client.list_tasks()
                for task in tasks:
                    project_name = project_map.get(task.get("project_id"), "Inbox")
                    self._index_task(task, project_name, stats)
                logger.info(f"Indexed {stats['tasks_indexed']} tasks")

        except Exception as e:
            logger.error(f"Error in Todoist indexing: {e}")
            stats["errors"] += 1

        self.kg.set_last_sync(
            source="todoist",
            account="default",
            last_sync=datetime.now(timezone.utc),
            metadata={"type": "full", "stats": stats},
        )

        logger.info(f"Todoist indexing complete: {stats}")
        return stats

    def index_delta(self, hours_back: int = 24) -> dict[str, Any]:
        """Index Todoist tasks (re-syncs all active tasks).

        Note: Todoist API doesn't provide change tracking, so we re-sync
        all active tasks. This is fast since there are typically few active tasks.

        Args:
            hours_back: Not used - included for API consistency.

        Returns:
            Statistics about the indexing operation.
        """
        logger.info("Starting delta Todoist sync (re-syncing active tasks)")

        try:
            client = TodoistClient()
        except ValueError as e:
            logger.error(f"Cannot initialize Todoist client: {e}")
            return {"error": str(e)}

        stats = {
            "tasks_synced": 0,
            "errors": 0,
        }

        try:
            # Get project names
            projects = client.list_projects()
            project_map = {p["id"]: p["name"] for p in projects}

            # Re-index all active tasks
            tasks = client.list_tasks()
            for task in tasks:
                project_name = project_map.get(task.get("project_id"), "Inbox")
                self._index_task(task, project_name, stats)
                stats["tasks_synced"] += 1

        except Exception as e:
            logger.error(f"Error in Todoist delta sync: {e}")
            stats["errors"] += 1

        self.kg.set_last_sync(
            source="todoist",
            account="default",
            last_sync=datetime.now(timezone.utc),
            metadata={"type": "delta", "stats": stats},
        )

        logger.info(f"Todoist delta sync complete: {stats}")
        return stats

    def _index_project(self, project: dict, stats: dict[str, int]) -> None:
        """Index a project."""
        content_id = f"todoist:project:{project['id']}"

        self.kg.upsert_content(
            content_id=content_id,
            content_type="project",
            source="todoist",
            source_account="default",
            title=project["name"],
            body=f"Todoist project: {project['name']}",
            source_id=project["id"],
            url=project.get("url"),
            timestamp=datetime.now(timezone.utc),
            metadata={
                "color": project.get("color"),
                "is_favorite": project.get("is_favorite"),
                "is_inbox": project.get("is_inbox_project"),
            },
        )

        stats["projects_indexed"] += 1

    def _index_task(
        self,
        task: dict,
        project_name: str,
        stats: dict[str, int],
    ) -> None:
        """Index a task."""
        content_id = f"todoist:task:{task['id']}"

        # Build body with description and metadata
        body_parts = []
        if task.get("description"):
            body_parts.append(task["description"])

        # Add due date info
        due = task.get("due")
        if due:
            due_str = due.get("string") or due.get("date")
            if due_str:
                body_parts.append(f"Due: {due_str}")

        # Add labels
        if task.get("labels"):
            body_parts.append(f"Labels: {', '.join(task['labels'])}")

        body = "\n".join(body_parts)

        # Build title with project context
        title = f"[{project_name}] {task['content']}"

        # Parse created_at timestamp
        created_at = None
        if task.get("created_at"):
            try:
                created_at = datetime.fromisoformat(
                    task["created_at"].replace("Z", "+00:00")
                )
            except (ValueError, AttributeError):
                pass

        self.kg.upsert_content(
            content_id=content_id,
            content_type="task",
            source="todoist",
            source_account="default",
            title=title,
            body=body,
            source_id=task["id"],
            url=task.get("url"),
            timestamp=created_at,
            metadata={
                "project_id": task.get("project_id"),
                "project_name": project_name,
                "priority": task.get("priority"),
                "labels": task.get("labels", []),
                "due": task.get("due"),
                "is_completed": task.get("is_completed", False),
            },
        )

        stats["tasks_indexed"] += 1

    def _index_label(self, label: dict, stats: dict[str, int]) -> None:
        """Index a label as an entity."""
        entity_id = f"todoist:label:{label['id']}"

        self.kg.upsert_entity(
            entity_id=entity_id,
            entity_type="label",
            name=label["name"],
            source="todoist",
            source_account="default",
            metadata={
                "color": label.get("color"),
                "is_favorite": label.get("is_favorite"),
            },
        )

        stats["labels_indexed"] += 1
