"""GitHub API client."""

import logging
from datetime import datetime
from typing import Any

from github import Auth, Github
from github.GithubException import GithubException

from ..config import GITHUB_ORG, GITHUB_TOKEN, GITHUB_USERNAME

logger = logging.getLogger(__name__)


class GitHubClient:
    """Client for interacting with GitHub API."""

    def __init__(
        self,
        token: str | None = None,
        username: str | None = None,
        org: str | None = None,
    ):
        """Initialize GitHub client.

        Args:
            token: GitHub personal access token.
            username: GitHub username.
            org: GitHub organization name.
        """
        self.token = token or GITHUB_TOKEN
        self.username = username or GITHUB_USERNAME
        self.org = org or GITHUB_ORG

        if not self.token:
            raise ValueError("GitHub token is required")

        auth = Auth.Token(self.token)
        self._github = Github(auth=auth)
        self._user = None
        self._org = None

    @property
    def user(self):
        """Get authenticated user."""
        if self._user is None:
            self._user = self._github.get_user()
        return self._user

    @property
    def organization(self):
        """Get organization."""
        if self._org is None and self.org:
            try:
                self._org = self._github.get_organization(self.org)
            except GithubException as e:
                logger.warning(f"Could not access organization {self.org}: {e}")
        return self._org

    def list_repos(
        self,
        include_org: bool = True,
        include_private: bool = True,
        max_results: int = 100,
    ) -> list[dict[str, Any]]:
        """List repositories.

        Args:
            include_org: Include organization repos.
            include_private: Include private repos.
            max_results: Maximum number of repos.

        Returns:
            List of repository metadata.
        """
        repos = []

        # Personal repos
        try:
            user_repos = self.user.get_repos(
                type="owner" if not include_private else "all"
            )
            for repo in user_repos:
                if len(repos) >= max_results:
                    break
                repos.append(self._parse_repo(repo))
        except GithubException as e:
            logger.error(f"Error listing user repos: {e}")

        # Organization repos
        if include_org and self.organization:
            try:
                org_repos = self.organization.get_repos(type="all")
                for repo in org_repos:
                    if len(repos) >= max_results:
                        break
                    repos.append(self._parse_repo(repo))
            except GithubException as e:
                logger.error(f"Error listing org repos: {e}")

        return repos[:max_results]

    def get_repo(self, repo_name: str) -> dict[str, Any] | None:
        """Get a repository by name.

        Args:
            repo_name: Full repo name (owner/repo) or just repo name (will try org).

        Returns:
            Repository metadata or None if not found.
        """
        if "/" not in repo_name and self.org:
            repo_name = f"{self.org}/{repo_name}"

        try:
            repo = self._github.get_repo(repo_name)
            return self._parse_repo(repo)
        except GithubException as e:
            if e.status == 404:
                return None
            logger.error(f"Error getting repo {repo_name}: {e}")
            raise

    def get_my_issues(
        self,
        state: str = "open",
        max_results: int = 50,
    ) -> list[dict[str, Any]]:
        """Get issues assigned to the authenticated user.

        Args:
            state: Issue state ("open", "closed", "all").
            max_results: Maximum number of issues.

        Returns:
            List of issue metadata.
        """
        issues = []

        try:
            user_issues = self._github.search_issues(
                f"is:issue assignee:{self.username} state:{state}",
                sort="updated",
                order="desc",
            )

            for issue in user_issues:
                if len(issues) >= max_results:
                    break
                issues.append(self._parse_issue(issue))

        except GithubException as e:
            logger.error(f"Error getting my issues: {e}")

        return issues

    def get_my_prs(
        self,
        state: str = "open",
        max_results: int = 50,
    ) -> list[dict[str, Any]]:
        """Get pull requests created by or assigned to the user.

        Args:
            state: PR state ("open", "closed", "all").
            max_results: Maximum number of PRs.

        Returns:
            List of PR metadata.
        """
        prs = []

        try:
            # PRs authored by user
            author_prs = self._github.search_issues(
                f"is:pr author:{self.username} state:{state}",
                sort="updated",
                order="desc",
            )

            for pr in author_prs:
                if len(prs) >= max_results:
                    break
                prs.append(self._parse_pr(pr))

            # PRs where user is requested reviewer
            if len(prs) < max_results:
                review_prs = self._github.search_issues(
                    f"is:pr review-requested:{self.username} state:{state}",
                    sort="updated",
                    order="desc",
                )

                for pr in review_prs:
                    if len(prs) >= max_results:
                        break
                    pr_data = self._parse_pr(pr)
                    # Avoid duplicates
                    if not any(p["id"] == pr_data["id"] for p in prs):
                        prs.append(pr_data)

        except GithubException as e:
            logger.error(f"Error getting my PRs: {e}")

        return prs

    def get_repo_issues(
        self,
        repo_name: str,
        state: str = "open",
        labels: list[str] | None = None,
        max_results: int = 50,
    ) -> list[dict[str, Any]]:
        """Get issues for a specific repository.

        Args:
            repo_name: Repository name (owner/repo or just repo).
            state: Issue state.
            labels: Filter by labels.
            max_results: Maximum number of issues.

        Returns:
            List of issue metadata.
        """
        if "/" not in repo_name and self.org:
            repo_name = f"{self.org}/{repo_name}"

        try:
            repo = self._github.get_repo(repo_name)
            issues = repo.get_issues(
                state=state,
                labels=labels or [],
                sort="updated",
                direction="desc",
            )

            result = []
            for issue in issues:
                if len(result) >= max_results:
                    break
                # Skip pull requests (they're also returned as issues)
                if issue.pull_request is None:
                    result.append(self._parse_issue(issue))

            return result

        except GithubException as e:
            logger.error(f"Error getting repo issues for {repo_name}: {e}")
            return []

    def search_code(
        self,
        query: str,
        repo: str | None = None,
        max_results: int = 30,
    ) -> list[dict[str, Any]]:
        """Search code across repositories.

        Args:
            query: Search query.
            repo: Limit to specific repo (owner/repo).
            max_results: Maximum number of results.

        Returns:
            List of code search results.
        """
        search_query = query
        if repo:
            if "/" not in repo and self.org:
                repo = f"{self.org}/{repo}"
            search_query = f"{query} repo:{repo}"
        elif self.org:
            search_query = f"{query} org:{self.org}"

        try:
            results = self._github.search_code(search_query)
            code_results = []

            for item in results:
                if len(code_results) >= max_results:
                    break

                code_results.append({
                    "name": item.name,
                    "path": item.path,
                    "repo": item.repository.full_name,
                    "url": item.html_url,
                    "sha": item.sha,
                })

            return code_results

        except GithubException as e:
            logger.error(f"Error searching code: {e}")
            return []

    def search_code_in_repo(
        self,
        repo: str,
        query: str,
        max_results: int = 30,
    ) -> list[dict[str, Any]]:
        """Search code in a specific repository.

        Args:
            repo: Repository in format owner/repo (or repo if org is configured).
            query: Search query text.
            max_results: Maximum number of results.

        Returns:
            List of code search results.
        """
        return self.search_code(query=query, repo=repo, max_results=max_results)

    def create_issue(
        self,
        repo_name: str,
        title: str,
        body: str | None = None,
        labels: list[str] | None = None,
        assignees: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a new issue.

        Args:
            repo_name: Repository name.
            title: Issue title.
            body: Issue body/description.
            labels: Labels to add.
            assignees: Users to assign.

        Returns:
            Created issue metadata.
        """
        if "/" not in repo_name and self.org:
            repo_name = f"{self.org}/{repo_name}"

        try:
            repo = self._github.get_repo(repo_name)
            issue = repo.create_issue(
                title=title,
                body=body or "",
                labels=labels or [],
                assignees=assignees or [],
            )
            return self._parse_issue(issue)

        except GithubException as e:
            logger.error(f"Error creating issue: {e}")
            raise

    def add_issue_comment(
        self,
        repo_name: str,
        issue_number: int,
        body: str,
    ) -> dict[str, Any]:
        """Add a comment to an issue.

        Args:
            repo_name: Repository name.
            issue_number: Issue number.
            body: Comment body.

        Returns:
            Created comment metadata.
        """
        if "/" not in repo_name and self.org:
            repo_name = f"{self.org}/{repo_name}"

        try:
            repo = self._github.get_repo(repo_name)
            issue = repo.get_issue(issue_number)
            comment = issue.create_comment(body)

            return {
                "id": comment.id,
                "body": comment.body,
                "user": comment.user.login,
                "created_at": comment.created_at,
                "url": comment.html_url,
            }

        except GithubException as e:
            logger.error(f"Error adding comment: {e}")
            raise

    def get_recent_commits(
        self,
        repo_name: str | None = None,
        max_results: int = 20,
        since: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Get recent commits.

        Args:
            repo_name: Specific repo, or None for all org repos.
            max_results: Maximum number of commits.
            since: Only commits after this date.

        Returns:
            List of commit metadata.
        """
        commits = []

        if repo_name:
            repos = [repo_name]
        elif self.org:
            repos = [r["full_name"] for r in self.list_repos(max_results=20)]
        else:
            repos = [r["full_name"] for r in self.list_repos(include_org=False, max_results=10)]

        for repo in repos:
            if len(commits) >= max_results:
                break

            try:
                r = self._github.get_repo(repo)
                repo_commits = r.get_commits(
                    author=self.username,
                    since=since,
                )

                for commit in repo_commits:
                    if len(commits) >= max_results:
                        break

                    commits.append({
                        "sha": commit.sha[:7],
                        "message": commit.commit.message.split("\n")[0],
                        "repo": repo,
                        "date": commit.commit.author.date,
                        "url": commit.html_url,
                    })

            except GithubException as e:
                logger.debug(f"Error getting commits for {repo}: {e}")

        commits.sort(key=lambda x: x["date"], reverse=True)
        return commits[:max_results]

    def _parse_repo(self, repo) -> dict[str, Any]:
        """Parse a repository object into a dictionary."""
        return {
            "id": repo.id,
            "name": repo.name,
            "full_name": repo.full_name,
            "description": repo.description,
            "private": repo.private,
            "url": repo.html_url,
            "language": repo.language,
            "stars": repo.stargazers_count,
            "forks": repo.forks_count,
            "open_issues": repo.open_issues_count,
            "created_at": repo.created_at,
            "updated_at": repo.updated_at,
            "pushed_at": repo.pushed_at,
        }

    def _parse_issue(self, issue) -> dict[str, Any]:
        """Parse an issue object into a dictionary."""
        return {
            "id": issue.id,
            "number": issue.number,
            "title": issue.title,
            "body": issue.body or "",
            "state": issue.state,
            "user": issue.user.login,
            "assignees": [a.login for a in issue.assignees],
            "labels": [l.name for l in issue.labels],
            "repo": issue.repository.full_name if hasattr(issue, "repository") else None,
            "url": issue.html_url,
            "created_at": issue.created_at,
            "updated_at": issue.updated_at,
            "closed_at": issue.closed_at,
            "comments": issue.comments,
        }

    def _parse_pr(self, pr) -> dict[str, Any]:
        """Parse a pull request object into a dictionary."""
        # PR objects from search_issues have slightly different attributes
        data = {
            "id": pr.id,
            "number": pr.number,
            "title": pr.title,
            "body": pr.body or "",
            "state": pr.state,
            "user": pr.user.login,
            "url": pr.html_url,
            "created_at": pr.created_at,
            "updated_at": pr.updated_at,
            "closed_at": pr.closed_at,
        }

        if hasattr(pr, "repository"):
            data["repo"] = pr.repository.full_name
        elif hasattr(pr, "repository_url"):
            data["repo"] = pr.repository_url.split("/")[-2] + "/" + pr.repository_url.split("/")[-1]

        return data
