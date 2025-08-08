"""Git watcher agent for DevGuard - monitors git repository changes with advanced multi-repository coordination."""

import asyncio
import logging
import os
import subprocess
import hashlib
import fnmatch
import time
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum

from .base_agent import BaseAgent
from ..core.config import Config, RepositoryConfig
from ..memory.shared_memory import SharedMemory, AgentState
from ..memory.vector_db import VectorDatabase
from ..llm.provider import LLMProvider

logger = logging.getLogger(__name__)


class RepositoryRelationship(Enum):
    """Types of relationships between repositories."""
    INDEPENDENT = "independent"
    DEPENDENCY = "dependency"
    MONOREPO_COMPONENT = "monorepo_component"
    SHARED_LIBRARY = "shared_library"
    MICROSERVICE = "microservice"


@dataclass
class RepositoryGroup:
    """Represents a group of related repositories."""
    name: str
    repositories: List[str]
    relationship_type: RepositoryRelationship
    priority: int = 1
    sync_strategy: str = "parallel"  # parallel, sequential, dependency_order
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class CrossRepoChange:
    """Represents a change that affects multiple repositories."""
    change_id: str
    primary_repository: str
    affected_repositories: List[str]
    change_type: str  # commit, branch, file_change, dependency_update
    impact_level: str  # low, medium, high, critical
    timestamp: datetime
    description: str
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class GitWorkflowEvent:
    """Represents a Git workflow event (merge, PR, tag, etc.)."""
    event_id: str
    repository: str
    event_type: str  # merge, tag, branch_create, branch_delete, pull_request
    source_branch: Optional[str] = None
    target_branch: Optional[str] = None
    commit_hash: Optional[str] = None
    author: Optional[str] = None
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
        if self.metadata is None:
            self.metadata = {}


class GitWatcherAgent(BaseAgent):
    """
    Enhanced Git watcher agent responsible for:
    - Advanced multi-repository monitoring and coordination  
    - Cross-repository change correlation and impact analysis
    - Git workflow automation and integration
    - Repository relationship management and synchronization
    - Batch operations across repository groups
    - Advanced Git features (hooks, remotes, submodules, workflows)
    - Repository health monitoring and alerting
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm_provider = kwargs.get('llm_provider')
        
        # Enhanced multi-repository configuration
        self.repositories: Dict[str, RepositoryConfig] = {}
        self.repository_groups: Dict[str, RepositoryGroup] = {}
        self.repository_relationships: Dict[str, Dict[str, RepositoryRelationship]] = {}
        
        # Enhanced tracking and coordination
        self.file_checksums: Dict[str, Dict[str, str]] = {}  # repo_path -> {file_path: checksum}
        self.last_commit_hashes: Dict[str, str] = {}  # repo_path -> commit_hash
        self.branch_states: Dict[str, Dict[str, str]] = {}  # repo_path -> {branch: commit_hash}
        self.remote_states: Dict[str, Dict[str, str]] = {}  # repo_path -> {remote: url}
        self.cross_repo_changes: List[CrossRepoChange] = []
        self.workflow_events: List[GitWorkflowEvent] = []
        
        # Advanced monitoring configuration
        self.monitoring_active = False
        self.poll_interval = 5.0  # seconds
        self.batch_operation_timeout = 300  # seconds for batch operations
        self.cross_repo_analysis_enabled = True
        self.workflow_automation_enabled = True
        
        # Initialize repositories and groups from config
        if self.config and hasattr(self.config, 'repositories'):
            if isinstance(self.config.repositories, dict):
                # Dictionary format: repo_name -> repo_config
                for repo_name, repo_config in self.config.repositories.items():
                    if isinstance(repo_config, RepositoryConfig):
                        self.repositories[repo_name] = repo_config
                    elif isinstance(repo_config, dict):
                        self.repositories[repo_name] = RepositoryConfig(**repo_config)
            elif isinstance(self.config.repositories, list):
                # List format: list of repo configs - auto-generate names
                for i, repo_config in enumerate(self.config.repositories):
                    repo_name = f"repo_{i+1}"
                    if isinstance(repo_config, RepositoryConfig):
                        self.repositories[repo_name] = repo_config
                    elif isinstance(repo_config, dict):
                        self.repositories[repo_name] = RepositoryConfig(**repo_config)
    
    async def execute(self, state: Any) -> Any:
        """Execute the git watcher agent's main logic."""
        if isinstance(state, dict):
            task = state
        else:
            task = {"type": "monitor_git", "description": str(state)}
        
        return await self.execute_task(task)
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an enhanced git monitoring task with multi-repository support."""
        try:
            self._update_state("busy", task.get("task_id"))
            
            task_type = task.get("type", "monitor_git")
            
            # Enhanced task types for multi-repository operations
            if task_type == "monitor_git":
                result = await self._monitor_repositories()
            elif task_type == "scan_repository":
                result = await self._scan_single_repository(task)
            elif task_type == "detect_changes":
                result = await self._detect_repository_changes(task)
            elif task_type == "start_monitoring":
                result = await self._start_continuous_monitoring()
            elif task_type == "stop_monitoring":
                result = await self._stop_continuous_monitoring()
            elif task_type == "get_repo_status":
                result = await self._get_repository_status(task)
            elif task_type == "analyze_commit":
                result = await self._analyze_commit(task)
            
            # NEW: Multi-repository coordination tasks
            elif task_type == "create_repo_group":
                result = await self._create_repository_group(task)
            elif task_type == "sync_repo_group":
                result = await self._sync_repository_group(task)
            elif task_type == "analyze_cross_repo_impact":
                result = await self._analyze_cross_repository_impact(task)
            elif task_type == "batch_repo_operation":
                result = await self._execute_batch_repository_operation(task)
            elif task_type == "monitor_git_workflows":
                result = await self._monitor_git_workflows(task)
            elif task_type == "sync_branches":
                result = await self._sync_branches_across_repos(task)
            elif task_type == "analyze_repo_health":
                result = await self._analyze_repository_health(task)
            elif task_type == "coordinate_releases":
                result = await self._coordinate_releases(task)
            elif task_type == "manage_dependencies":
                result = await self._manage_cross_repo_dependencies(task)
            elif task_type == "validate_repo_consistency":
                result = await self._validate_repository_consistency(task)
            
            else:
                result = await self._monitor_repositories()  # Default action
            
            self._update_state("idle")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in git watcher task execution: {e}")
            self._update_state("error", error=str(e))
            return {"success": False, "error": str(e)}

    async def _monitor_repositories(self) -> Dict[str, Any]:
        """Monitor all configured repositories for changes."""
        changes_detected = []
        repositories_scanned = 0
        
        for repo_name, repo_config in self.repositories.items():
            try:
                self.logger.debug(f"Monitoring repository: {repo_name} at {repo_config.path}")
                
                repo_changes = await self._scan_repository_for_changes(repo_name, repo_config)
                if repo_changes['has_changes']:
                    changes_detected.append({
                        "repository": repo_name,
                        "path": repo_config.path,
                        "changes": repo_changes
                    })
                    
                    # Log significant changes to memory
                    self.log_observation(
                        f"Changes detected in repository {repo_name}",
                        data={
                            "repository": repo_name,
                            "path": repo_config.path,
                            "change_summary": repo_changes['summary'],
                            "files_changed": len(repo_changes.get('file_changes', []))
                        }
                    )
                
                repositories_scanned += 1
                
            except Exception as e:
                self.logger.error(f"Error monitoring repository {repo_name}: {e}")
                continue
        
        return {
            "success": True,
            "repositories_scanned": repositories_scanned,
            "changes_detected": len(changes_detected),
            "change_details": changes_detected,
            "monitoring_timestamp": datetime.now(timezone.utc).isoformat()
        }

    async def _scan_repository_for_changes(self, repo_name: str, repo_config: RepositoryConfig) -> Dict[str, Any]:
        """Scan a single repository for changes."""
        repo_path = Path(repo_config.path)
        
        if not repo_path.exists():
            return {
                "has_changes": False,
                "error": f"Repository path does not exist: {repo_path}",
                "summary": "Repository not found"
            }
        
        changes = {
            "has_changes": False,
            "git_changes": [],
            "file_changes": [],
            "new_commits": [],
            "branch_changes": {},
            "summary": "No changes detected"
        }
        
        try:
            # Check for Git changes (commits, branch changes)
            git_changes = await self._check_git_changes(repo_path, repo_config)
            changes["git_changes"] = git_changes
            
            # Check for file system changes
            file_changes = await self._check_file_changes(repo_path, repo_config, repo_name)
            changes["file_changes"] = file_changes
            
            # Determine if there are significant changes
            has_changes = (
                len(git_changes) > 0 or 
                len(file_changes) > 0
            )
            
            changes["has_changes"] = has_changes
            
            if has_changes:
                changes["summary"] = self._generate_change_summary(git_changes, file_changes)
            
        except Exception as e:
            self.logger.error(f"Error scanning repository {repo_name}: {e}")
            changes["error"] = str(e)
        
        return changes

    async def _check_git_changes(self, repo_path: Path, repo_config: RepositoryConfig) -> List[Dict[str, Any]]:
        """Check for Git-specific changes (commits, branches, etc.)."""
        git_changes = []
        
        try:
            # Check if it's a Git repository
            if not (repo_path / ".git").exists():
                return git_changes
            
            # Get current branch
            current_branch = await self._run_git_command(repo_path, ["branch", "--show-current"])
            current_branch = current_branch.strip()
            
            # Get latest commit hash
            latest_commit = await self._run_git_command(repo_path, ["rev-parse", "HEAD"])
            latest_commit = latest_commit.strip()
            
            repo_key = str(repo_path)
            last_known_commit = self.last_commit_hashes.get(repo_key)
            
            # Check for new commits
            if last_known_commit and last_known_commit != latest_commit:
                # Get commits between last known and current
                commit_range = f"{last_known_commit}..{latest_commit}"
                commit_log = await self._run_git_command(
                    repo_path, 
                    ["log", "--oneline", "--no-merges", commit_range]
                )
                
                if commit_log.strip():
                    commits = []
                    for line in commit_log.strip().split('\n'):
                        if line.strip():
                            parts = line.split(' ', 1)
                            commit_hash = parts[0]
                            commit_message = parts[1] if len(parts) > 1 else ""
                            commits.append({
                                "hash": commit_hash,
                                "message": commit_message
                            })
                    
                    git_changes.append({
                        "type": "new_commits",
                        "branch": current_branch,
                        "commits": commits,
                        "commit_count": len(commits)
                    })
            
            # Update our record of the latest commit
            self.last_commit_hashes[repo_key] = latest_commit
            
            # Check for uncommitted changes
            status_output = await self._run_git_command(repo_path, ["status", "--porcelain"])
            if status_output.strip():
                staged_files = []
                unstaged_files = []
                untracked_files = []
                
                for line in status_output.strip().split('\n'):
                    if len(line) >= 3:
                        status_code = line[:2]
                        file_path = line[3:]
                        
                        if status_code[0] != ' ':  # Staged changes
                            staged_files.append({
                                "file": file_path,
                                "status": status_code[0]
                            })
                        
                        if status_code[1] != ' ':  # Unstaged changes
                            if status_code[1] == '?':
                                untracked_files.append(file_path)
                            else:
                                unstaged_files.append({
                                    "file": file_path,
                                    "status": status_code[1]
                                })
                
                if staged_files or unstaged_files or untracked_files:
                    git_changes.append({
                        "type": "uncommitted_changes",
                        "staged_files": staged_files,
                        "unstaged_files": unstaged_files,
                        "untracked_files": untracked_files
                    })
            
        except Exception as e:
            self.logger.error(f"Error checking Git changes: {e}")
        
        return git_changes

    async def _check_file_changes(self, repo_path: Path, repo_config: RepositoryConfig, repo_name: str) -> List[Dict[str, Any]]:
        """Check for file system changes based on checksums."""
        file_changes = []
        current_checksums = {}
        
        try:
            # Get all files matching watch patterns
            watched_files = await self._get_watched_files(repo_path, repo_config)
            
            # Calculate checksums for current files
            for file_path in watched_files:
                try:
                    checksum = await self._calculate_file_checksum(file_path)
                    relative_path = str(file_path.relative_to(repo_path))
                    current_checksums[relative_path] = checksum
                except Exception as e:
                    self.logger.warning(f"Error calculating checksum for {file_path}: {e}")
                    continue
            
            # Compare with previous checksums
            repo_key = f"{repo_name}:{repo_path}"
            previous_checksums = self.file_checksums.get(repo_key, {})
            
            # Find new, modified, and deleted files
            current_files = set(current_checksums.keys())
            previous_files = set(previous_checksums.keys())
            
            # New files
            new_files = current_files - previous_files
            for file_path in new_files:
                file_changes.append({
                    "type": "added",
                    "file": file_path,
                    "checksum": current_checksums[file_path]
                })
            
            # Deleted files
            deleted_files = previous_files - current_files
            for file_path in deleted_files:
                file_changes.append({
                    "type": "deleted",
                    "file": file_path,
                    "previous_checksum": previous_checksums[file_path]
                })
            
            # Modified files
            common_files = current_files & previous_files
            for file_path in common_files:
                if current_checksums[file_path] != previous_checksums[file_path]:
                    file_changes.append({
                        "type": "modified",
                        "file": file_path,
                        "checksum": current_checksums[file_path],
                        "previous_checksum": previous_checksums[file_path]
                    })
            
            # Update our checksum record
            self.file_checksums[repo_key] = current_checksums
            
        except Exception as e:
            self.logger.error(f"Error checking file changes: {e}")
        
        return file_changes

    async def _get_watched_files(self, repo_path: Path, repo_config: RepositoryConfig) -> List[Path]:
        """Get all files that should be watched based on configuration."""
        watched_files = []
        
        try:
            # Walk through directory tree
            for root, dirs, files in os.walk(repo_path):
                root_path = Path(root)
                
                # Check scan depth
                depth = len(root_path.relative_to(repo_path).parts)
                if depth > repo_config.scan_depth:
                    dirs.clear()  # Don't recurse deeper
                    continue
                
                # Filter directories based on ignore patterns
                dirs[:] = [d for d in dirs if not self._should_ignore_path(d, repo_config.ignore_patterns)]
                
                for file in files:
                    file_path = root_path / file
                    
                    # Check if file should be ignored
                    if self._should_ignore_path(str(file_path.relative_to(repo_path)), repo_config.ignore_patterns):
                        continue
                    
                    # Check if file matches watch patterns
                    if self._matches_watch_patterns(file, repo_config.watch_files):
                        # Check file size limit
                        try:
                            file_size_mb = file_path.stat().st_size / (1024 * 1024)
                            if file_size_mb <= repo_config.max_file_size_mb:
                                watched_files.append(file_path)
                        except Exception as e:
                            self.logger.warning(f"Error checking file size for {file_path}: {e}")
                            
        except Exception as e:
            self.logger.error(f"Error getting watched files: {e}")
        
        return watched_files

    def _should_ignore_path(self, path: str, ignore_patterns: List[str]) -> bool:
        """Check if a path should be ignored based on patterns."""
        for pattern in ignore_patterns:
            if fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(Path(path).name, pattern):
                return True
        return False

    def _matches_watch_patterns(self, filename: str, watch_patterns: List[str]) -> bool:
        """Check if a filename matches any watch pattern."""
        for pattern in watch_patterns:
            if fnmatch.fnmatch(filename, pattern):
                return True
        return False

    async def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum for a file."""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            self.logger.error(f"Error calculating checksum for {file_path}: {e}")
            return ""

    async def _run_git_command(self, repo_path: Path, args: List[str]) -> str:
        """Run a Git command and return the output."""
        try:
            cmd = ["git"] + args
            result = subprocess.run(
                cmd,
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                return result.stdout
            else:
                self.logger.error(f"Git command failed: {' '.join(cmd)}, Error: {result.stderr}")
                return ""
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"Git command timed out: {' '.join(args)}")
            return ""
        except Exception as e:
            self.logger.error(f"Error running Git command: {e}")
            return ""

    def _generate_change_summary(self, git_changes: List[Dict[str, Any]], file_changes: List[Dict[str, Any]]) -> str:
        """Generate a human-readable summary of changes."""
        summary_parts = []
        
        # Git changes summary
        for change in git_changes:
            if change["type"] == "new_commits":
                commit_count = change["commit_count"]
                branch = change["branch"]
                summary_parts.append(f"{commit_count} new commit(s) on {branch}")
            elif change["type"] == "uncommitted_changes":
                staged = len(change.get("staged_files", []))
                unstaged = len(change.get("unstaged_files", []))
                untracked = len(change.get("untracked_files", []))
                if staged > 0:
                    summary_parts.append(f"{staged} staged file(s)")
                if unstaged > 0:
                    summary_parts.append(f"{unstaged} unstaged file(s)")
                if untracked > 0:
                    summary_parts.append(f"{untracked} untracked file(s)")
        
        # File changes summary
        added = sum(1 for change in file_changes if change["type"] == "added")
        modified = sum(1 for change in file_changes if change["type"] == "modified")
        deleted = sum(1 for change in file_changes if change["type"] == "deleted")
        
        if added > 0:
            summary_parts.append(f"{added} file(s) added")
        if modified > 0:
            summary_parts.append(f"{modified} file(s) modified")
        if deleted > 0:
            summary_parts.append(f"{deleted} file(s) deleted")
        
        return "; ".join(summary_parts) if summary_parts else "No significant changes"

    async def _scan_single_repository(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Scan a specific repository for changes."""
        repo_path = task.get("repo_path")
        if not repo_path:
            return {"success": False, "error": "No repository path provided"}
        
        # Create a temporary repository config
        repo_config = RepositoryConfig(path=repo_path)
        
        try:
            changes = await self._scan_repository_for_changes("single_scan", repo_config)
            return {
                "success": True,
                "repository_path": repo_path,
                "changes": changes,
                "scan_timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _detect_repository_changes(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Detect changes in a specific repository since last check."""
        repo_name = task.get("repo_name")
        if not repo_name or repo_name not in self.repositories:
            return {"success": False, "error": f"Repository '{repo_name}' not found"}
        
        repo_config = self.repositories[repo_name]
        changes = await self._scan_repository_for_changes(repo_name, repo_config)
        
        return {
            "success": True,
            "repository": repo_name,
            "changes": changes,
            "detection_timestamp": datetime.now(timezone.utc).isoformat()
        }

    async def _start_continuous_monitoring(self) -> Dict[str, Any]:
        """Start continuous monitoring of all repositories."""
        if self.monitoring_active:
            return {"success": True, "message": "Monitoring already active"}
        
        self.monitoring_active = True
        asyncio.create_task(self._continuous_monitoring_loop())
        
        return {
            "success": True,
            "message": "Continuous monitoring started",
            "repositories": list(self.repositories.keys()),
            "poll_interval": self.poll_interval
        }

    async def _stop_continuous_monitoring(self) -> Dict[str, Any]:
        """Stop continuous monitoring."""
        self.monitoring_active = False
        
        return {
            "success": True,
            "message": "Continuous monitoring stopped"
        }

    async def _continuous_monitoring_loop(self):
        """Main continuous monitoring loop."""
        self.logger.info("Starting continuous Git monitoring")
        
        while self.monitoring_active:
            try:
                await self._monitor_repositories()
                await asyncio.sleep(self.poll_interval)
            except Exception as e:
                self.logger.error(f"Error in continuous monitoring loop: {e}")
                await asyncio.sleep(self.poll_interval)  # Continue despite errors
        
        self.logger.info("Continuous Git monitoring stopped")

    async def _get_repository_status(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Get the current status of a repository."""
        repo_name = task.get("repo_name")
        if not repo_name or repo_name not in self.repositories:
            return {"success": False, "error": f"Repository '{repo_name}' not found"}
        
        repo_config = self.repositories[repo_name]
        repo_path = Path(repo_config.path)
        
        try:
            status = {
                "repository": repo_name,
                "path": str(repo_path),
                "exists": repo_path.exists(),
                "is_git_repo": (repo_path / ".git").exists() if repo_path.exists() else False
            }
            
            if status["is_git_repo"]:
                # Get Git status information
                current_branch = await self._run_git_command(repo_path, ["branch", "--show-current"])
                latest_commit = await self._run_git_command(repo_path, ["rev-parse", "HEAD"])
                status_output = await self._run_git_command(repo_path, ["status", "--porcelain"])
                
                status.update({
                    "current_branch": current_branch.strip(),
                    "latest_commit": latest_commit.strip()[:7],  # Short hash
                    "has_uncommitted_changes": bool(status_output.strip()),
                    "watched_files_count": len(await self._get_watched_files(repo_path, repo_config))
                })
            
            return {"success": True, "status": status}
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _analyze_commit(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a specific commit for detailed change information."""
        repo_name = task.get("repo_name")
        commit_hash = task.get("commit_hash")
        
        if not repo_name or repo_name not in self.repositories:
            return {"success": False, "error": f"Repository '{repo_name}' not found"}
        
        if not commit_hash:
            return {"success": False, "error": "No commit hash provided"}
        
        repo_config = self.repositories[repo_name]
        repo_path = Path(repo_config.path)
        
        try:
            # Get commit details
            commit_info = await self._run_git_command(
                repo_path, 
                ["show", "--format=%H|%an|%ae|%ad|%s", "--no-patch", commit_hash]
            )
            
            if not commit_info.strip():
                return {"success": False, "error": "Commit not found"}
            
            parts = commit_info.strip().split('|')
            commit_details = {
                "hash": parts[0] if len(parts) > 0 else "",
                "author": parts[1] if len(parts) > 1 else "",
                "email": parts[2] if len(parts) > 2 else "",
                "date": parts[3] if len(parts) > 3 else "",
                "message": parts[4] if len(parts) > 4 else ""
            }
            
            # Get changed files
            changed_files = await self._run_git_command(
                repo_path,
                ["diff-tree", "--no-commit-id", "--name-status", "-r", commit_hash]
            )
            
            file_changes = []
            if changed_files.strip():
                for line in changed_files.strip().split('\n'):
                    if line.strip():
                        parts = line.split('\t', 1)
                        if len(parts) == 2:
                            status, file_path = parts
                            file_changes.append({
                                "status": status,
                                "file": file_path
                            })
            
            return {
                "success": True,
                "commit": commit_details,
                "file_changes": file_changes,
                "analysis_timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ================================
    # ENHANCED MULTI-REPOSITORY METHODS
    # ================================

    async def _create_repository_group(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new repository group for coordinated monitoring."""
        try:
            group_name = task.get("group_name")
            repositories = task.get("repositories", [])
            relationship_type = task.get("relationship_type", RepositoryRelationship.INDEPENDENT.value)
            priority = task.get("priority", 1)
            sync_strategy = task.get("sync_strategy", "parallel")

            if not group_name or not repositories:
                return {"success": False, "error": "Group name and repositories required"}

            # Validate repositories exist
            missing_repos = [repo for repo in repositories if repo not in self.repositories]
            if missing_repos:
                return {"success": False, "error": f"Repositories not found: {missing_repos}"}

            # Create repository group
            group = RepositoryGroup(
                name=group_name,
                repositories=repositories,
                relationship_type=RepositoryRelationship(relationship_type),
                priority=priority,
                sync_strategy=sync_strategy,
                metadata=task.get("metadata", {})
            )

            self.repository_groups[group_name] = group

            self.log_observation(
                f"Created repository group: {group_name}",
                data={
                    "group_name": group_name,
                    "repositories": repositories,
                    "relationship_type": relationship_type,
                    "sync_strategy": sync_strategy
                }
            )

            return {
                "success": True,
                "group_name": group_name,
                "repositories": repositories,
                "relationship_type": relationship_type,
                "sync_strategy": sync_strategy,
                "created_timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _sync_repository_group(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronize repositories within a group based on their sync strategy."""
        try:
            group_name = task.get("group_name")
            if not group_name or group_name not in self.repository_groups:
                return {"success": False, "error": f"Repository group '{group_name}' not found"}

            group = self.repository_groups[group_name]
            sync_results = []

            if group.sync_strategy == "parallel":
                # Execute operations in parallel
                tasks = []
                for repo_name in group.repositories:
                    if repo_name in self.repositories:
                        repo_config = self.repositories[repo_name]
                        tasks.append(self._sync_single_repository(repo_name, repo_config))
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for i, result in enumerate(results):
                    repo_name = group.repositories[i]
                    if isinstance(result, Exception):
                        sync_results.append({
                            "repository": repo_name,
                            "success": False,
                            "error": str(result)
                        })
                    else:
                        sync_results.append({
                            "repository": repo_name,
                            "success": True,
                            "result": result
                        })

            elif group.sync_strategy == "sequential":
                # Execute operations sequentially
                for repo_name in group.repositories:
                    if repo_name in self.repositories:
                        repo_config = self.repositories[repo_name]
                        try:
                            result = await self._sync_single_repository(repo_name, repo_config)
                            sync_results.append({
                                "repository": repo_name,
                                "success": True,
                                "result": result
                            })
                        except Exception as e:
                            sync_results.append({
                                "repository": repo_name,
                                "success": False,
                                "error": str(e)
                            })

            successful_syncs = sum(1 for r in sync_results if r["success"])
            
            return {
                "success": True,
                "group_name": group_name,
                "sync_strategy": group.sync_strategy,
                "total_repositories": len(group.repositories),
                "successful_syncs": successful_syncs,
                "sync_results": sync_results,
                "sync_timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _sync_single_repository(self, repo_name: str, repo_config: RepositoryConfig) -> Dict[str, Any]:
        """Synchronize a single repository (fetch latest changes)."""
        repo_path = Path(repo_config.path)
        
        try:
            # Fetch from all remotes
            fetch_output = await self._run_git_command(repo_path, ["fetch", "--all"])
            
            # Get current branch and upstream status
            current_branch = await self._run_git_command(repo_path, ["branch", "--show-current"])
            current_branch = current_branch.strip()
            
            # Check if branch has upstream
            upstream_output = await self._run_git_command(
                repo_path, ["rev-parse", "--abbrev-ref", f"{current_branch}@{{upstream}}"]
            )
            
            sync_info = {
                "repository": repo_name,
                "current_branch": current_branch,
                "fetch_successful": bool(fetch_output),
                "has_upstream": bool(upstream_output.strip())
            }
            
            if upstream_output.strip():
                # Check if branch is behind upstream
                behind_count = await self._run_git_command(
                    repo_path, ["rev-list", "--count", f"{current_branch}..{upstream_output.strip()}"]
                )
                ahead_count = await self._run_git_command(
                    repo_path, ["rev-list", "--count", f"{upstream_output.strip()}..{current_branch}"]
                )
                
                sync_info.update({
                    "behind_upstream": int(behind_count.strip()) if behind_count.strip() else 0,
                    "ahead_upstream": int(ahead_count.strip()) if ahead_count.strip() else 0
                })
            
            return sync_info
            
        except Exception as e:
            return {"repository": repo_name, "error": str(e)}

    async def _analyze_cross_repository_impact(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how changes in one repository might impact others."""
        try:
            primary_repo = task.get("repository")
            change_type = task.get("change_type", "commit")
            
            if not primary_repo or primary_repo not in self.repositories:
                return {"success": False, "error": f"Repository '{primary_repo}' not found"}

            # Analyze potential impacts
            impact_analysis = {
                "primary_repository": primary_repo,
                "change_type": change_type,
                "potentially_affected": [],
                "high_impact": [],
                "medium_impact": [],
                "low_impact": [],
                "analysis_timestamp": datetime.now(timezone.utc).isoformat()
            }

            # Check repository groups for related repositories
            affected_repos = set()
            for group_name, group in self.repository_groups.items():
                if primary_repo in group.repositories:
                    for repo in group.repositories:
                        if repo != primary_repo:
                            affected_repos.add(repo)
                            
                            # Categorize impact based on relationship type
                            if group.relationship_type == RepositoryRelationship.DEPENDENCY:
                                impact_analysis["high_impact"].append(repo)
                            elif group.relationship_type == RepositoryRelationship.SHARED_LIBRARY:
                                impact_analysis["high_impact"].append(repo)
                            elif group.relationship_type == RepositoryRelationship.MICROSERVICE:
                                impact_analysis["medium_impact"].append(repo)
                            elif group.relationship_type == RepositoryRelationship.MONOREPO_COMPONENT:
                                impact_analysis["medium_impact"].append(repo)
                            else:
                                impact_analysis["low_impact"].append(repo)

            impact_analysis["potentially_affected"] = list(affected_repos)

            # Create cross-repository change record if significant impact
            if affected_repos:
                change_id = f"{primary_repo}_{change_type}_{int(datetime.now().timestamp())}"
                cross_repo_change = CrossRepoChange(
                    change_id=change_id,
                    primary_repository=primary_repo,
                    affected_repositories=list(affected_repos),
                    change_type=change_type,
                    impact_level=self._determine_impact_level(impact_analysis),
                    timestamp=datetime.now(timezone.utc),
                    description=f"{change_type} in {primary_repo} affecting {len(affected_repos)} repositories"
                )
                
                self.cross_repo_changes.append(cross_repo_change)

                # Log to shared memory
                self.log_observation(
                    f"Cross-repository impact detected: {primary_repo}",
                    data={
                        "change_id": change_id,
                        "primary_repository": primary_repo,
                        "affected_count": len(affected_repos),
                        "impact_level": cross_repo_change.impact_level
                    }
                )

            return {
                "success": True,
                **impact_analysis
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _determine_impact_level(self, impact_analysis: Dict[str, Any]) -> str:
        """Determine the overall impact level of a cross-repository change."""
        high_count = len(impact_analysis.get("high_impact", []))
        medium_count = len(impact_analysis.get("medium_impact", []))
        low_count = len(impact_analysis.get("low_impact", []))

        if high_count > 0:
            return "high" if high_count >= 3 else "medium"
        elif medium_count > 2:
            return "medium"
        elif medium_count > 0 or low_count > 3:
            return "low"
        else:
            return "minimal"

    async def _execute_batch_repository_operation(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a batch operation across multiple repositories."""
        try:
            operation = task.get("operation")
            target_repos = task.get("repositories", [])
            operation_params = task.get("params", {})
            parallel = task.get("parallel", True)

            if not operation:
                return {"success": False, "error": "Operation type required"}

            if not target_repos:
                target_repos = list(self.repositories.keys())

            # Filter to valid repositories
            valid_repos = [repo for repo in target_repos if repo in self.repositories]
            if not valid_repos:
                return {"success": False, "error": "No valid repositories found"}

            operation_results = []

            if parallel:
                # Execute in parallel
                tasks = []
                for repo_name in valid_repos:
                    repo_config = self.repositories[repo_name]
                    tasks.append(self._execute_single_repo_operation(
                        repo_name, repo_config, operation, operation_params
                    ))
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for i, result in enumerate(results):
                    repo_name = valid_repos[i]
                    if isinstance(result, Exception):
                        operation_results.append({
                            "repository": repo_name,
                            "success": False,
                            "error": str(result)
                        })
                    else:
                        operation_results.append({
                            "repository": repo_name,
                            "success": True,
                            "result": result
                        })
            else:
                # Execute sequentially
                for repo_name in valid_repos:
                    repo_config = self.repositories[repo_name]
                    try:
                        result = await self._execute_single_repo_operation(
                            repo_name, repo_config, operation, operation_params
                        )
                        operation_results.append({
                            "repository": repo_name,
                            "success": True,
                            "result": result
                        })
                    except Exception as e:
                        operation_results.append({
                            "repository": repo_name,
                            "success": False,
                            "error": str(e)
                        })

            successful_ops = sum(1 for r in operation_results if r["success"])

            return {
                "success": True,
                "operation": operation,
                "total_repositories": len(valid_repos),
                "successful_operations": successful_ops,
                "parallel_execution": parallel,
                "operation_results": operation_results,
                "execution_timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_single_repo_operation(self, repo_name: str, repo_config: RepositoryConfig, 
                                          operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single repository operation."""
        repo_path = Path(repo_config.path)
        
        try:
            if operation == "status":
                # Get repository status
                status_output = await self._run_git_command(repo_path, ["status", "--porcelain"])
                branch_output = await self._run_git_command(repo_path, ["branch", "--show-current"])
                return {
                    "operation": "status",
                    "clean": not bool(status_output.strip()),
                    "current_branch": branch_output.strip(),
                    "status_details": status_output.strip()
                }
            
            elif operation == "fetch":
                # Fetch from remote
                remote = params.get("remote", "origin")
                fetch_output = await self._run_git_command(repo_path, ["fetch", remote])
                return {
                    "operation": "fetch",
                    "remote": remote,
                    "output": fetch_output.strip()
                }
            
            elif operation == "pull":
                # Pull from upstream
                pull_output = await self._run_git_command(repo_path, ["pull"])
                return {
                    "operation": "pull",
                    "output": pull_output.strip()
                }
            
            elif operation == "branch_list":
                # List branches
                local_branches = await self._run_git_command(repo_path, ["branch"])
                remote_branches = await self._run_git_command(repo_path, ["branch", "-r"])
                return {
                    "operation": "branch_list",
                    "local_branches": [b.strip().replace("* ", "") for b in local_branches.split("\n") if b.strip()],
                    "remote_branches": [b.strip() for b in remote_branches.split("\n") if b.strip()]
                }
            
            elif operation == "log":
                # Get commit log
                count = params.get("count", 10)
                log_output = await self._run_git_command(
                    repo_path, ["log", f"--oneline", f"-{count}"]
                )
                return {
                    "operation": "log",
                    "commits": [line.strip() for line in log_output.split("\n") if line.strip()]
                }
                
            else:
                return {"operation": operation, "error": f"Unsupported operation: {operation}"}
                
        except Exception as e:
            return {"operation": operation, "error": str(e)}

    async def _monitor_git_workflows(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor Git workflows (merges, tags, branches) across repositories."""
        try:
            target_repos = task.get("repositories", list(self.repositories.keys()))
            workflow_events = []

            for repo_name in target_repos:
                if repo_name not in self.repositories:
                    continue

                repo_config = self.repositories[repo_name]
                repo_path = Path(repo_config.path)

                try:
                    # Check for recent merges
                    merge_commits = await self._run_git_command(
                        repo_path, ["log", "--merges", "--oneline", "-10"]
                    )
                    
                    if merge_commits.strip():
                        for line in merge_commits.strip().split("\n")[:5]:  # Limit to recent merges
                            if line.strip():
                                parts = line.split(" ", 1)
                                commit_hash = parts[0]
                                message = parts[1] if len(parts) > 1 else ""
                                
                                event = GitWorkflowEvent(
                                    event_id=f"{repo_name}_merge_{commit_hash}",
                                    repository=repo_name,
                                    event_type="merge",
                                    commit_hash=commit_hash,
                                    metadata={"message": message}
                                )
                                workflow_events.append(event)
                    
                    # Check for recent tags
                    tags = await self._run_git_command(repo_path, ["tag", "--sort=-creatordate", "-10"])
                    if tags.strip():
                        for tag in tags.strip().split("\n")[:3]:  # Limit to recent tags
                            if tag.strip():
                                event = GitWorkflowEvent(
                                    event_id=f"{repo_name}_tag_{tag}",
                                    repository=repo_name,
                                    event_type="tag",
                                    metadata={"tag_name": tag}
                                )
                                workflow_events.append(event)

                    # Check for new branches
                    all_branches = await self._run_git_command(repo_path, ["branch", "-a"])
                    current_branches = set()
                    if all_branches.strip():
                        for branch in all_branches.split("\n"):
                            if branch.strip() and not branch.strip().startswith("remotes/origin/HEAD"):
                                branch_name = branch.strip().replace("* ", "").replace("remotes/origin/", "")
                                current_branches.add(branch_name)

                    # Compare with previous branch state
                    repo_key = str(repo_path)
                    previous_branches = set(self.branch_states.get(repo_key, {}).keys())
                    new_branches = current_branches - previous_branches
                    deleted_branches = previous_branches - current_branches

                    for branch in new_branches:
                        event = GitWorkflowEvent(
                            event_id=f"{repo_name}_branch_create_{branch}",
                            repository=repo_name,
                            event_type="branch_create",
                            source_branch=branch,
                            metadata={"branch_name": branch}
                        )
                        workflow_events.append(event)

                    for branch in deleted_branches:
                        event = GitWorkflowEvent(
                            event_id=f"{repo_name}_branch_delete_{branch}",
                            repository=repo_name,
                            event_type="branch_delete",
                            source_branch=branch,
                            metadata={"branch_name": branch}
                        )
                        workflow_events.append(event)

                    # Update branch state
                    branch_commits = {}
                    for branch in current_branches:
                        try:
                            commit = await self._run_git_command(repo_path, ["rev-parse", branch])
                            if commit.strip():
                                branch_commits[branch] = commit.strip()
                        except:
                            continue
                    
                    self.branch_states[repo_key] = branch_commits

                except Exception as e:
                    self.logger.error(f"Error monitoring workflows for {repo_name}: {e}")
                    continue

            # Store workflow events
            self.workflow_events.extend(workflow_events)
            
            # Keep only recent events (last 100)
            self.workflow_events = self.workflow_events[-100:]

            return {
                "success": True,
                "monitored_repositories": len(target_repos),
                "workflow_events_detected": len(workflow_events),
                "workflow_events": [asdict(event) for event in workflow_events],
                "monitoring_timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _sync_branches_across_repos(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronize specific branches across multiple repositories."""
        try:
            target_branch = task.get("branch", "main")
            target_repos = task.get("repositories", list(self.repositories.keys()))
            operation = task.get("operation", "fetch")  # fetch, pull, checkout

            sync_results = []

            for repo_name in target_repos:
                if repo_name not in self.repositories:
                    continue

                repo_config = self.repositories[repo_name]
                repo_path = Path(repo_config.path)

                try:
                    if operation == "fetch":
                        # Fetch the target branch
                        output = await self._run_git_command(
                            repo_path, ["fetch", "origin", f"{target_branch}:{target_branch}"]
                        )
                        sync_results.append({
                            "repository": repo_name,
                            "success": True,
                            "operation": "fetch",
                            "branch": target_branch,
                            "output": output.strip()
                        })

                    elif operation == "checkout":
                        # Checkout the target branch
                        output = await self._run_git_command(repo_path, ["checkout", target_branch])
                        sync_results.append({
                            "repository": repo_name,
                            "success": True,
                            "operation": "checkout",
                            "branch": target_branch,
                            "output": output.strip()
                        })

                    elif operation == "pull":
                        # Checkout and pull the target branch
                        checkout_output = await self._run_git_command(repo_path, ["checkout", target_branch])
                        pull_output = await self._run_git_command(repo_path, ["pull", "origin", target_branch])
                        sync_results.append({
                            "repository": repo_name,
                            "success": True,
                            "operation": "pull",
                            "branch": target_branch,
                            "checkout_output": checkout_output.strip(),
                            "pull_output": pull_output.strip()
                        })

                except Exception as e:
                    sync_results.append({
                        "repository": repo_name,
                        "success": False,
                        "operation": operation,
                        "branch": target_branch,
                        "error": str(e)
                    })

            successful_syncs = sum(1 for r in sync_results if r["success"])

            return {
                "success": True,
                "operation": operation,
                "target_branch": target_branch,
                "total_repositories": len(target_repos),
                "successful_syncs": successful_syncs,
                "sync_results": sync_results,
                "sync_timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _analyze_repository_health(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the health of repositories (commit frequency, branch status, issues)."""
        try:
            target_repos = task.get("repositories", list(self.repositories.keys()))
            health_analysis = []

            for repo_name in target_repos:
                if repo_name not in self.repositories:
                    continue

                repo_config = self.repositories[repo_name]
                repo_path = Path(repo_config.path)

                try:
                    # Analyze commit activity
                    recent_commits = await self._run_git_command(
                        repo_path, ["log", "--oneline", "--since=30 days ago"]
                    )
                    commit_count_30d = len([line for line in recent_commits.split("\n") if line.strip()])

                    # Check repository status
                    status_output = await self._run_git_command(repo_path, ["status", "--porcelain"])
                    has_uncommitted = bool(status_output.strip())

                    # Check branch ahead/behind status
                    current_branch = await self._run_git_command(repo_path, ["branch", "--show-current"])
                    current_branch = current_branch.strip()

                    upstream_status = "unknown"
                    try:
                        upstream = await self._run_git_command(
                            repo_path, ["rev-parse", "--abbrev-ref", f"{current_branch}@{{upstream}}"]
                        )
                        if upstream.strip():
                            behind = await self._run_git_command(
                                repo_path, ["rev-list", "--count", f"{current_branch}..{upstream.strip()}"]
                            )
                            ahead = await self._run_git_command(
                                repo_path, ["rev-list", "--count", f"{upstream.strip()}..{current_branch}"]
                            )
                            
                            behind_count = int(behind.strip()) if behind.strip() else 0
                            ahead_count = int(ahead.strip()) if ahead.strip() else 0
                            
                            if behind_count == 0 and ahead_count == 0:
                                upstream_status = "up_to_date"
                            elif behind_count > 0 and ahead_count == 0:
                                upstream_status = f"behind_{behind_count}"
                            elif behind_count == 0 and ahead_count > 0:
                                upstream_status = f"ahead_{ahead_count}"
                            else:
                                upstream_status = f"diverged_{behind_count}_{ahead_count}"
                    except:
                        upstream_status = "no_upstream"

                    # Calculate health score
                    health_score = 100
                    health_issues = []

                    if commit_count_30d == 0:
                        health_score -= 30
                        health_issues.append("No commits in last 30 days")
                    elif commit_count_30d < 5:
                        health_score -= 10
                        health_issues.append("Low commit activity")

                    if has_uncommitted:
                        health_score -= 15
                        health_issues.append("Uncommitted changes")

                    if "behind" in upstream_status:
                        health_score -= 20
                        health_issues.append("Branch behind upstream")

                    if upstream_status == "no_upstream":
                        health_score -= 10
                        health_issues.append("No upstream tracking")

                    health_analysis.append({
                        "repository": repo_name,
                        "health_score": max(0, health_score),
                        "commit_activity_30d": commit_count_30d,
                        "has_uncommitted_changes": has_uncommitted,
                        "current_branch": current_branch,
                        "upstream_status": upstream_status,
                        "health_issues": health_issues,
                        "status": "healthy" if health_score >= 80 else "attention_needed" if health_score >= 60 else "poor"
                    })

                except Exception as e:
                    health_analysis.append({
                        "repository": repo_name,
                        "health_score": 0,
                        "status": "error",
                        "error": str(e)
                    })

            # Calculate overall health
            avg_health = sum(repo.get("health_score", 0) for repo in health_analysis) / len(health_analysis) if health_analysis else 0
            
            return {
                "success": True,
                "overall_health_score": round(avg_health, 2),
                "repositories_analyzed": len(health_analysis),
                "healthy_repos": len([r for r in health_analysis if r.get("status") == "healthy"]),
                "attention_needed": len([r for r in health_analysis if r.get("status") == "attention_needed"]),
                "poor_health": len([r for r in health_analysis if r.get("status") == "poor"]),
                "repository_health": health_analysis,
                "analysis_timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _coordinate_releases(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate releases across multiple repositories."""
        try:
            release_version = task.get("version")
            target_repos = task.get("repositories", [])
            release_type = task.get("type", "tag")  # tag, branch
            message = task.get("message", f"Release {release_version}")

            if not release_version:
                return {"success": False, "error": "Release version required"}

            if not target_repos:
                return {"success": False, "error": "Target repositories required"}

            coordination_results = []

            for repo_name in target_repos:
                if repo_name not in self.repositories:
                    coordination_results.append({
                        "repository": repo_name,
                        "success": False,
                        "error": "Repository not found"
                    })
                    continue

                repo_config = self.repositories[repo_name]
                repo_path = Path(repo_config.path)

                try:
                    if release_type == "tag":
                        # Create release tag
                        tag_output = await self._run_git_command(
                            repo_path, ["tag", "-a", release_version, "-m", message]
                        )
                        
                        # Push tag if requested
                        if task.get("push", False):
                            push_output = await self._run_git_command(
                                repo_path, ["push", "origin", release_version]
                            )
                            coordination_results.append({
                                "repository": repo_name,
                                "success": True,
                                "release_type": "tag",
                                "version": release_version,
                                "tag_output": tag_output.strip(),
                                "push_output": push_output.strip()
                            })
                        else:
                            coordination_results.append({
                                "repository": repo_name,
                                "success": True,
                                "release_type": "tag",
                                "version": release_version,
                                "tag_output": tag_output.strip()
                            })

                    elif release_type == "branch":
                        # Create release branch
                        branch_output = await self._run_git_command(
                            repo_path, ["checkout", "-b", f"release/{release_version}"]
                        )
                        
                        coordination_results.append({
                            "repository": repo_name,
                            "success": True,
                            "release_type": "branch",
                            "version": release_version,
                            "branch_output": branch_output.strip()
                        })

                except Exception as e:
                    coordination_results.append({
                        "repository": repo_name,
                        "success": False,
                        "release_type": release_type,
                        "version": release_version,
                        "error": str(e)
                    })

            successful_releases = sum(1 for r in coordination_results if r["success"])

            # Create workflow event for release coordination
            event = GitWorkflowEvent(
                event_id=f"release_coordination_{release_version}_{int(datetime.now().timestamp())}",
                repository="multi_repo",
                event_type="coordinate_release",
                metadata={
                    "release_version": release_version,
                    "release_type": release_type,
                    "target_repositories": target_repos,
                    "successful_releases": successful_releases
                }
            )
            self.workflow_events.append(event)

            return {
                "success": True,
                "release_version": release_version,
                "release_type": release_type,
                "total_repositories": len(target_repos),
                "successful_releases": successful_releases,
                "coordination_results": coordination_results,
                "coordination_timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _manage_cross_repo_dependencies(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Manage dependencies between repositories."""
        try:
            operation = task.get("operation", "analyze")  # analyze, update, validate
            target_repos = task.get("repositories", list(self.repositories.keys()))

            dependency_results = []

            for repo_name in target_repos:
                if repo_name not in self.repositories:
                    continue

                repo_config = self.repositories[repo_name]
                repo_path = Path(repo_config.path)

                try:
                    # Look for dependency files
                    dependency_files = []
                    
                    # Python dependencies
                    if (repo_path / "requirements.txt").exists():
                        dependency_files.append("requirements.txt")
                    if (repo_path / "pyproject.toml").exists():
                        dependency_files.append("pyproject.toml")
                    
                    # Node.js dependencies
                    if (repo_path / "package.json").exists():
                        dependency_files.append("package.json")
                    
                    # Analyze dependencies
                    dependencies = {}
                    for dep_file in dependency_files:
                        dep_path = repo_path / dep_file
                        try:
                            with open(dep_path, 'r') as f:
                                content = f.read()
                                dependencies[dep_file] = self._extract_dependencies(dep_file, content)
                        except Exception as e:
                            self.logger.warning(f"Error reading {dep_file}: {e}")

                    dependency_results.append({
                        "repository": repo_name,
                        "success": True,
                        "dependency_files": dependency_files,
                        "dependencies": dependencies,
                        "operation": operation
                    })

                except Exception as e:
                    dependency_results.append({
                        "repository": repo_name,
                        "success": False,
                        "error": str(e)
                    })

            return {
                "success": True,
                "operation": operation,
                "repositories_analyzed": len([r for r in dependency_results if r["success"]]),
                "dependency_analysis": dependency_results,
                "analysis_timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _extract_dependencies(self, filename: str, content: str) -> List[str]:
        """Extract dependencies from dependency files."""
        dependencies = []
        
        try:
            if filename == "requirements.txt":
                for line in content.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Extract package name (before == or >= etc.)
                        pkg_name = line.split('==')[0].split('>=')[0].split('<=')[0].split('!=')[0].strip()
                        if pkg_name:
                            dependencies.append(pkg_name)
            
            elif filename == "package.json":
                import json
                try:
                    pkg_data = json.loads(content)
                    for dep_type in ['dependencies', 'devDependencies']:
                        if dep_type in pkg_data:
                            dependencies.extend(list(pkg_data[dep_type].keys()))
                except json.JSONDecodeError:
                    pass
            
            elif filename == "pyproject.toml":
                # Basic TOML parsing for dependencies
                lines = content.split('\n')
                in_dependencies = False
                for line in lines:
                    line = line.strip()
                    if line.startswith('[tool.poetry.dependencies]') or line.startswith('[project.dependencies]'):
                        in_dependencies = True
                        continue
                    if in_dependencies and line.startswith('['):
                        in_dependencies = False
                        continue
                    if in_dependencies and '=' in line:
                        pkg_name = line.split('=')[0].strip().strip('"')
                        if pkg_name and not pkg_name.startswith('#'):
                            dependencies.append(pkg_name)
        
        except Exception as e:
            self.logger.warning(f"Error parsing {filename}: {e}")
        
        return dependencies

    async def _validate_repository_consistency(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Validate consistency across repositories (naming, structure, configs)."""
        try:
            target_repos = task.get("repositories", list(self.repositories.keys()))
            validation_rules = task.get("rules", {
                "branch_naming": True,
                "file_structure": True,
                "dependency_versions": True,
                "commit_message_format": True
            })

            validation_results = []

            for repo_name in target_repos:
                if repo_name not in self.repositories:
                    continue

                repo_config = self.repositories[repo_name]
                repo_path = Path(repo_config.path)

                validation_issues = []
                validation_score = 100

                try:
                    # Branch naming validation
                    if validation_rules.get("branch_naming", False):
                        branches = await self._run_git_command(repo_path, ["branch", "-a"])
                        for branch_line in branches.split('\n'):
                            branch = branch_line.strip().replace('* ', '').replace('remotes/origin/', '')
                            if branch and not self._validate_branch_name(branch):
                                validation_issues.append(f"Invalid branch name: {branch}")
                                validation_score -= 5

                    # File structure validation
                    if validation_rules.get("file_structure", False):
                        required_files = task.get("required_files", ["README.md", ".gitignore"])
                        for req_file in required_files:
                            if not (repo_path / req_file).exists():
                                validation_issues.append(f"Missing required file: {req_file}")
                                validation_score -= 10

                    # Commit message validation (recent commits)
                    if validation_rules.get("commit_message_format", False):
                        recent_commits = await self._run_git_command(
                            repo_path, ["log", "--format=%s", "-10"]
                        )
                        for commit_msg in recent_commits.split('\n')[:5]:
                            if commit_msg.strip() and not self._validate_commit_message(commit_msg.strip()):
                                validation_issues.append(f"Invalid commit message format: {commit_msg[:50]}...")
                                validation_score -= 3

                    validation_results.append({
                        "repository": repo_name,
                        "validation_score": max(0, validation_score),
                        "validation_issues": validation_issues,
                        "status": "compliant" if validation_score >= 90 else "needs_attention" if validation_score >= 70 else "non_compliant"
                    })

                except Exception as e:
                    validation_results.append({
                        "repository": repo_name,
                        "validation_score": 0,
                        "status": "error",
                        "error": str(e)
                    })

            # Calculate overall compliance
            avg_score = sum(r.get("validation_score", 0) for r in validation_results) / len(validation_results) if validation_results else 0
            
            return {
                "success": True,
                "overall_compliance_score": round(avg_score, 2),
                "repositories_validated": len(validation_results),
                "compliant_repos": len([r for r in validation_results if r.get("status") == "compliant"]),
                "needs_attention": len([r for r in validation_results if r.get("status") == "needs_attention"]),
                "non_compliant": len([r for r in validation_results if r.get("status") == "non_compliant"]),
                "validation_results": validation_results,
                "validation_timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _validate_branch_name(self, branch_name: str) -> bool:
        """Validate branch name format."""
        if not branch_name or branch_name in ["HEAD", "master", "main"]:
            return True
        
        # Basic validation - no spaces, valid characters
        invalid_chars = [' ', '..', '~', '^', ':', '?', '*', '[', '\\']
        return not any(char in branch_name for char in invalid_chars)

    def _validate_commit_message(self, message: str) -> bool:
        """Validate commit message format."""
        if not message:
            return False
        
        # Basic validation - should be descriptive
        return len(message) >= 10 and not message.startswith("fix") or message.startswith("feat") or ":" in message

    def _update_state(self, status: str, task_id: Optional[str] = None, error: Optional[str] = None) -> None:
        """Update agent state."""
        state = AgentState(
            agent_id=self.agent_id,
            status=status,
            current_task=task_id,
            last_heartbeat=datetime.now(timezone.utc),
            metadata={"error": error if error else None}
        )
        self.shared_memory.update_agent_state(state)
    
    def get_capabilities(self) -> List[str]:
        """Get enhanced Git Watcher agent capabilities."""
        return [
            # Core Git monitoring
            "git_monitoring",
            "repository_scanning",
            "change_detection",
            "file_tracking",
            "commit_analysis",
            "branch_monitoring",
            "continuous_monitoring",
            "repository_status",
            "uncommitted_tracking",
            
            # Multi-repository coordination
            "multi_repository_support",
            "repository_group_management",
            "cross_repository_analysis",
            "batch_operations",
            "repository_synchronization",
            
            # Advanced Git features
            "git_workflow_monitoring",
            "branch_synchronization",
            "release_coordination",
            "dependency_management",
            "repository_health_analysis",
            "consistency_validation",
            
            # Workflow automation
            "workflow_events_tracking",
            "automated_syncing",
            "coordinated_releases",
            "cross_repo_impact_analysis",
            
            # Integration capabilities
            "shared_memory_integration",
            "swarm_coordination",
            "task_based_operations",
            "async_processing",
            "comprehensive_logging"
        ]
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status with enhanced monitoring information."""
        return {
            "agent_id": self.agent_id,
            "type": "git_watcher",
            "monitoring_active": self.monitoring_active,
            "repositories_count": len(self.repositories),
            "repositories": list(self.repositories.keys()),
            "repository_groups_count": len(self.repository_groups),
            "repository_groups": list(self.repository_groups.keys()),
            "cross_repo_changes_tracked": len(self.cross_repo_changes),
            "workflow_events_tracked": len(self.workflow_events),
            "poll_interval": self.poll_interval,
            "last_heartbeat": datetime.now(timezone.utc).isoformat(),
            "capabilities": {
                "multi_repository_coordination": True,
                "cross_repository_analysis": True,
                "workflow_automation": True,
                "batch_operations": True,
                "repository_health_monitoring": True
            }
        }
