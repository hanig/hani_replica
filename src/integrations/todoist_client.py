"""Todoist API client."""

import logging
from datetime import datetime
from typing import Any

import httpx

from ..config import TODOIST_API_KEY

logger = logging.getLogger(__name__)

TODOIST_BASE_URL = "https://api.todoist.com/rest/v2"


class TodoistClient:
    """Client for interacting with Todoist API."""

    def __init__(self, api_key: str | None = None):
        """Initialize Todoist client.

        Args:
            api_key: Todoist API token.
        """
        self.api_key = api_key or TODOIST_API_KEY

        if not self.api_key:
            raise ValueError("Todoist API key is required")

        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _request(
        self,
        method: str,
        endpoint: str,
        json: dict | None = None,
        params: dict | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Make a request to the Todoist API.

        Args:
            method: HTTP method.
            endpoint: API endpoint (without base URL).
            json: JSON body for POST requests.
            params: Query parameters.

        Returns:
            Response JSON.

        Raises:
            httpx.HTTPStatusError: If the request fails.
        """
        url = f"{TODOIST_BASE_URL}{endpoint}"

        with httpx.Client() as client:
            response = client.request(
                method=method,
                url=url,
                headers=self._headers,
                json=json,
                params=params,
                timeout=30.0,
            )
            response.raise_for_status()

            # Some endpoints return empty response (e.g., DELETE)
            if response.status_code == 204 or not response.content:
                return {}
            return response.json()

    def test_connection(self) -> dict[str, Any]:
        """Test the connection to Todoist API.

        Returns:
            Connection status and project count.
        """
        try:
            projects = self.list_projects()
            return {
                "success": True,
                "project_count": len(projects),
                "message": f"Connected to Todoist with {len(projects)} projects",
            }
        except httpx.HTTPStatusError as e:
            logger.error(f"Todoist connection test failed: {e}")
            return {"success": False, "error": str(e)}

    # --- Projects ---

    def list_projects(self) -> list[dict[str, Any]]:
        """List all projects.

        Returns:
            List of project objects.
        """
        result = self._request("GET", "/projects")
        return [self._parse_project(p) for p in result]

    def get_project(self, project_id: str) -> dict[str, Any]:
        """Get a project by ID.

        Args:
            project_id: The project ID.

        Returns:
            Project object.
        """
        result = self._request("GET", f"/projects/{project_id}")
        return self._parse_project(result)

    # --- Tasks ---

    def list_tasks(
        self,
        project_id: str | None = None,
        filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """List active tasks.

        Args:
            project_id: Filter by project.
            filter: Todoist filter string (e.g., "today", "overdue", "@label").

        Returns:
            List of task objects.
        """
        params = {}
        if project_id:
            params["project_id"] = project_id
        if filter:
            params["filter"] = filter

        result = self._request("GET", "/tasks", params=params if params else None)
        return [self._parse_task(t) for t in result]

    def get_task(self, task_id: str) -> dict[str, Any]:
        """Get a task by ID.

        Args:
            task_id: The task ID.

        Returns:
            Task object.
        """
        result = self._request("GET", f"/tasks/{task_id}")
        return self._parse_task(result)

    def create_task(
        self,
        content: str,
        description: str | None = None,
        project_id: str | None = None,
        due_string: str | None = None,
        due_date: str | None = None,
        priority: int = 1,
        labels: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a new task.

        Args:
            content: Task title/content.
            description: Task description.
            project_id: Project to add task to.
            due_string: Natural language due date (e.g., "tomorrow", "next monday").
            due_date: Due date in YYYY-MM-DD format.
            priority: Priority 1-4 (4 is highest/urgent).
            labels: List of label names.

        Returns:
            Created task object.
        """
        body: dict[str, Any] = {"content": content}

        if description:
            body["description"] = description
        if project_id:
            body["project_id"] = project_id
        if due_string:
            body["due_string"] = due_string
        elif due_date:
            body["due_date"] = due_date
        if priority and priority > 1:
            body["priority"] = priority
        if labels:
            body["labels"] = labels

        result = self._request("POST", "/tasks", json=body)
        return self._parse_task(result)

    def update_task(
        self,
        task_id: str,
        content: str | None = None,
        description: str | None = None,
        due_string: str | None = None,
        priority: int | None = None,
        labels: list[str] | None = None,
    ) -> dict[str, Any]:
        """Update a task.

        Args:
            task_id: The task ID.
            content: New task content.
            description: New description.
            due_string: New due date string.
            priority: New priority.
            labels: New labels.

        Returns:
            Updated task object.
        """
        body = {}
        if content is not None:
            body["content"] = content
        if description is not None:
            body["description"] = description
        if due_string is not None:
            body["due_string"] = due_string
        if priority is not None:
            body["priority"] = priority
        if labels is not None:
            body["labels"] = labels

        result = self._request("POST", f"/tasks/{task_id}", json=body)
        return self._parse_task(result)

    def complete_task(self, task_id: str) -> bool:
        """Mark a task as complete.

        Args:
            task_id: The task ID.

        Returns:
            True if successful.
        """
        self._request("POST", f"/tasks/{task_id}/close")
        return True

    def reopen_task(self, task_id: str) -> bool:
        """Reopen a completed task.

        Args:
            task_id: The task ID.

        Returns:
            True if successful.
        """
        self._request("POST", f"/tasks/{task_id}/reopen")
        return True

    def delete_task(self, task_id: str) -> bool:
        """Delete a task.

        Args:
            task_id: The task ID.

        Returns:
            True if successful.
        """
        self._request("DELETE", f"/tasks/{task_id}")
        return True

    # --- Labels ---

    def list_labels(self) -> list[dict[str, Any]]:
        """List all labels.

        Returns:
            List of label objects.
        """
        result = self._request("GET", "/labels")
        return [self._parse_label(l) for l in result]

    # --- Comments ---

    def list_comments(self, task_id: str) -> list[dict[str, Any]]:
        """List comments on a task.

        Args:
            task_id: The task ID.

        Returns:
            List of comment objects.
        """
        result = self._request("GET", "/comments", params={"task_id": task_id})
        return [self._parse_comment(c) for c in result]

    def add_comment(self, task_id: str, content: str) -> dict[str, Any]:
        """Add a comment to a task.

        Args:
            task_id: The task ID.
            content: Comment text.

        Returns:
            Created comment object.
        """
        result = self._request(
            "POST", "/comments", json={"task_id": task_id, "content": content}
        )
        return self._parse_comment(result)

    # --- Parsing Helpers ---

    def _parse_project(self, project: dict) -> dict[str, Any]:
        """Parse a project object."""
        return {
            "id": project["id"],
            "name": project["name"],
            "color": project.get("color"),
            "is_favorite": project.get("is_favorite", False),
            "is_inbox_project": project.get("is_inbox_project", False),
            "view_style": project.get("view_style", "list"),
            "url": project.get("url"),
        }

    def _parse_task(self, task: dict) -> dict[str, Any]:
        """Parse a task object."""
        due = task.get("due")
        due_info = None
        if due:
            due_info = {
                "date": due.get("date"),
                "string": due.get("string"),
                "datetime": due.get("datetime"),
                "is_recurring": due.get("is_recurring", False),
            }

        return {
            "id": task["id"],
            "content": task["content"],
            "description": task.get("description", ""),
            "project_id": task.get("project_id"),
            "priority": task.get("priority", 1),
            "due": due_info,
            "labels": task.get("labels", []),
            "is_completed": task.get("is_completed", False),
            "created_at": task.get("created_at"),
            "url": task.get("url"),
        }

    def _parse_label(self, label: dict) -> dict[str, Any]:
        """Parse a label object."""
        return {
            "id": label["id"],
            "name": label["name"],
            "color": label.get("color"),
            "is_favorite": label.get("is_favorite", False),
        }

    def _parse_comment(self, comment: dict) -> dict[str, Any]:
        """Parse a comment object."""
        return {
            "id": comment["id"],
            "task_id": comment.get("task_id"),
            "content": comment["content"],
            "posted_at": comment.get("posted_at"),
        }
